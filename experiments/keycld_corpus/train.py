"""Full official video loss and paired full-budget state component interventions."""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('WANDB_MODE','disabled')
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental.ode import odeint
from keycld import util
from keycld.losses import loss_fn_step
from keycld.models import predict
from keycld.train import ExperimentBase
from .model import ControlledKeyCLD, rk4_four

if not hasattr(jax,'tree_map'):
    jax.tree_map=jax.tree_util.tree_map


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,data):
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); tmp.replace(path)


def save(path,data):
    tmp=path.with_suffix(path.suffix+'.partial')
    with tmp.open('wb') as handle:
        pickle.dump(jax.device_get(data),handle)
    tmp.replace(path)


def load(root,record,state=False):
    with np.load(root/record['observations']) as data:
        item={k:data[k] for k in data.files}
    item['x']=item['x'].astype(np.float32)/255
    if state:
        # This is a separately named privileged-state diagnostic, never video training.
        with np.load(root/record['state_labels']) as data:
            item['points']=data['x'][:len(item['t'])].reshape(-1,6)
            item['velocity']=data['v'][:len(item['t'])].reshape(-1,6)
    return item


def run(root,arm,seed):
    root=Path(root); out=root/'runs'/('%s_seed%d'%(arm,seed)); out.mkdir(parents=True,exist_ok=True)
    if (out/'training_report.json').exists():
        print('Already complete',out,flush=True); return
    protocol=json.loads((root/'protocol.json').read_text()); manifest=json.loads((root/'dataset_manifest.json').read_text())
    assert seed in protocol['training']['seeds'] and arm in protocol['training']['arms']
    identity={'protocol_sha256':sha(root/'protocol.json'),'dataset_manifest_sha256':sha(root/'dataset_manifest.json'),
              'train_source_sha256':sha(Path(__file__)),'model_adapter_sha256':sha(Path(__file__).with_name('model.py'))}
    start=time.monotonic(); state_arm=arm!='video_euler'
    def status(stage,**values):
        write(out/'status.json',{'stage':stage,'arm':arm,'seed':seed,'pid':os.getpid(),'elapsed_s':time.monotonic()-start,**values})
    status('loading_training_data')
    records=manifest['records']
    train=[load(root,r,state_arm) for r in records if r['split']=='train']
    validation=[load(root,r,state_arm) for r in records if r['split']=='validation']
    # Test/calibration data is never loaded by this training entry point.
    rng=np.random.RandomState(seed); params=ControlledKeyCLD(jnp.asarray(train[0]['geometry'])).init(jax.random.PRNGKey(seed))
    initial_path=out/'initial_params.p'
    paired_path=root/'runs'/('video_euler_seed%d'%seed)/'initial_params.p'
    source_initial=initial_path if initial_path.exists() else paired_path if state_arm and paired_path.exists() else None
    if source_initial is not None:
        with source_initial.open('rb') as handle:
            params=pickle.load(handle)
    initial=params
    experiment=ExperimentBase(num_epochs=40,learning_rate=.0003,batch_size=1,num_hidden_dim=32,
                              num_predicted_steps=4,dynamics_weight=1.,bce_weight=1.,solver='euler')
    experiment.configure_optimizers(params,40*len(train)); tx=experiment.tx; opt_state=experiment.opt_state
    def objective(parameters,item,augment):
        model=ControlledKeyCLD(item['geometry'])
        if not state_arm:
            return loss_fn_step(1.,1.,4,'euler',model,parameters,item,augment)
        x=item['points'][1:-1]; v=item['velocity'][1:-1]
        v=jax.vmap(partial(util.project_velocity,model.constraint_fn))(x,v)
        initial_state=jnp.concatenate([x,v],axis=1)[:-5]
        times=jnp.arange(5)*(item['t'][1]-item['t'][0])
        solver=util.explicit_euler if arm=='state_euler' else rk4_four
        history=jax.vmap(solver,in_axes=(None,0,None,None))(partial(model.ode,parameters),initial_state,times,item['action'])
        target=jnp.stack([x[i:i+5] for i in range(len(initial_state))])
        loss=jnp.sum((history[:,:,:6]-target)**2)
        return loss,{'dynamics':loss}
    gradient=jax.value_and_grad(objective,has_aux=True)
    @jax.jit
    def step(parameters,opt,item,augmentation):
        (loss,aux),grad=gradient(parameters,item,augmentation)
        updates,new_opt=tx.update(grad,opt,parameters)
        return optax.apply_updates(parameters,updates),new_opt,loss,aux
    @jax.jit
    def forecast(parameters,item):
        model=ControlledKeyCLD(item['geometry'])
        if state_arm:
            x=item['points'][1]; v=util.project_velocity(model.constraint_fn,x,item['velocity'][1])
            history=odeint(partial(model.ode,parameters),jnp.r_[x,v],item['t'][1:],item['action'],mxstep=4096)
            return history[:,:6].reshape(-1,3,2)
        maps=model.encoder(parameters,item['x'][:3]); kp,_=util.map_to_keypoints(maps)
        kp=predict(model,parameters,item['t'],kp,item['action'],solver=partial(odeint,mxstep=4096))
        images,_=model.renderer(parameters,kp)
        return images
    mean_image=np.mean([x['x'].mean(axis=0) for x in train],axis=0)
    epsilon=float(np.mean([np.mean((x['x']-mean_image)**2) for x in train]))
    history=[]; updates=0; first_epoch=0
    checkpoint=out/'resume.p'
    if checkpoint.exists():
        with checkpoint.open('rb') as handle:
            saved=pickle.load(handle)
        previous_identity=dict(saved['identity'])
        if previous_identity != identity:
            compatibility=json.loads((root/'source_resume_compatibility.json').read_text())
            pair=[previous_identity['train_source_sha256'],identity['train_source_sha256']]
            assert pair in compatibility['allowed_train_source_pairs'], 'Unregistered source migration'
            previous_identity['train_source_sha256']=identity['train_source_sha256']
            assert previous_identity==identity, 'Dataset, protocol or dynamics adapter changed since checkpoint'

        params=saved['params']; opt_state=saved['opt_state']; rng.set_state(saved['rng'])
        history=saved['history']; updates=saved['updates']; first_epoch=saved['next_epoch']
    else:
        save(out/'initial_params.p',params)
    event_path=out/'execution_segments.json'
    events=json.loads(event_path.read_text()) if event_path.exists() else []
    events.append({'started_unix_s':time.time(),'resume_epoch':first_epoch,'resume_updates':updates,
                   'devices':[str(d) for d in jax.devices()],'source_identity':identity})
    write(event_path,events)
    for epoch in range(first_epoch,40):
        losses=[]; auxes=[]
        for batch,index in enumerate(rng.permutation(len(train))):
            augmentation=rng.randint(0,8,50).astype(np.int32)
            params,opt_state,loss,aux=step(params,opt_state,train[index],augmentation)
            loss=float(loss)
            if not np.isfinite(loss):
                status('failed',epoch=epoch+1,updates=updates,reason='Non-finite full-method loss',training_index=int(index))
                raise RuntimeError('Nonfinite loss; no sequences skipped or algorithm substituted')
            losses.append(loss); auxes.append({k:float(v) for k,v in aux.items()}); updates+=1
            if (batch+1)%150==0:
                status('training',epoch=epoch+1,total_epochs=40,updates=updates,total_updates=40*len(train),batch=batch+1)
        # The final checkpoint is fixed before training. Validation has no role in
        # optimizer updates or checkpoint selection; run it as a separate bounded job.
        errors=[]; vpts=[]; failures=0
        row={'epoch':epoch+1,'updates':updates,'loss':float(np.mean(losses)),
             'loss_components':{k:float(np.mean([a[k] for a in auxes])) for k in auxes[0]},
             'validation_error':float(np.mean(errors)) if errors else None,
             'validation_error_unit':'m point-coordinate RMSE' if state_arm else 'RGB MSE',
             'validation_vpt_index':float(np.mean(vpts)) if vpts else None,
             'validation_nonfinite':None,'validation_status':'deferred_to_frozen_inference',
             'elapsed_s':time.monotonic()-start}
        history.append(row); write(out/'history.json',history)
        save(checkpoint,{'identity':identity,'params':params,'opt_state':opt_state,'rng':rng.get_state(),
                         'history':history,'updates':updates,'next_epoch':epoch+1})
        status('epoch_complete',total_epochs=40,total_updates=40*len(train),**row)
        print(json.dumps(dict(arm=arm,seed=seed,**row)),flush=True)
    save(out/'params.p',params)
    visual_unchanged=all(np.array_equal(a,b) for name in ['encoder','renderer'] for a,b in
                         zip(jax.tree_util.tree_leaves(initial[name]),jax.tree_util.tree_leaves(params[name])))
    if state_arm:
        assert visual_unchanged
    report={**identity,'arm':arm,'seed':seed,'epochs':40,'optimizer_updates':updates,'training_sequences':len(train),
            'validation_sequences':len(validation),'training_assemblies':manifest['assemblies_by_split']['train'],
            'initial_checkpoint_sha256':sha(out/'initial_params.p'),'checkpoint_sha256':sha(out/'params.p'),
            'test_used_for_training_or_selection':False,'visual_weights_unchanged':visual_unchanged,
            'state_supervision':state_arm,'validation_execution':'separate frozen-checkpoint inference job',
            'epsilon_train_only':epsilon,'checkpoint_selection':'final epoch 40',
            'jax_version':jax.__version__,'devices':[str(d) for d in jax.devices()],
            'elapsed_s':time.monotonic()-start}
    write(out/'training_report.json',report); status('training_complete',**report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--arm',choices=['video_euler','state_euler','state_rk4'],required=True)
    parser.add_argument('--seed',type=int,required=True); args=parser.parse_args()
    run(args.root,args.arm,args.seed)
