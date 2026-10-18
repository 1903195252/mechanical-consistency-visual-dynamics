"""Official KeyCLD loss/optimizer configuration, adapted four-bar observations.

The optimizer-state update is explicitly repaired to implement actual Adam.
No simulator state, keypoint labels or independent physics equation enters loss.
"""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault('WANDB_MODE','disabled')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
import numpy as np
import optax
from keycld.losses import loss_fn_step
from keycld.models import predict
from keycld.train import ExperimentBase
from keycld import util
from learned_common import load_model

# Same tree operation, renamed in modern JAX; no numerical algorithm change.
if not hasattr(jax,'tree_map'):jax.tree_map=jax.tree_util.tree_map


def load_observation(root,index):
    with np.load(root/'observations'/f'run_{index:04d}.npz') as data:
        assert set(data.files)=={'x','t','action'}
        return {'x':data['x'].astype(np.float32)/255,'t':data['t'],'action':data['action']}


def epsilon(runs):
    return float(util_epsilon(runs))


def util_epsilon(runs):
    means=[run['x'].mean(axis=0) for run in runs];mean=np.mean(means,axis=0)
    return np.mean([np.mean((mean[None]-run['x'])**2) for run in runs])


def write_status(root,stage,**values):
    path=root/'training_status.json';temporary=path.with_suffix('.json.partial')
    temporary.write_text(json.dumps({'stage':stage,**values},indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root;protocol=json.loads((root/'training_protocol.json').read_text())
    start=time.monotonic();rng=np.random.RandomState(protocol['training']['seed'])
    # Only observation archives are opened; truth directory is never read here.
    train=[load_observation(root,i) for i in range(450)]
    validation=[load_observation(root,i) for i in range(450,500)]
    thresholds={'paper_validation_train_plus_validation':epsilon(train+validation),'independent_test_train_only':epsilon(train)}
    (root/'vpt_thresholds.json').write_text(json.dumps(thresholds,indent=2)+'\n')
    model=load_model(root);params=model.init(jax.random.PRNGKey(protocol['training']['seed']))
    with (root/'initial_params.p').open('wb') as handle:pickle.dump(jax.device_get(params),handle)
    experiment=ExperimentBase(num_epochs=40,learning_rate=.0003,batch_size=1,num_hidden_dim=32,
                             num_predicted_steps=4,dynamics_weight=1.,bce_weight=1.,solver='euler')
    experiment.configure_optimizers(params,40*len(train));optimizer=experiment.tx;state=experiment.opt_state
    objective=partial(loss_fn_step,1.,1.,4,'euler',model)
    loss_grad=jax.value_and_grad(util.reduce_mean(jax.vmap(objective,in_axes=(None,0,0))),has_aux=True)
    @jax.jit
    def step(parameters,optimizer_state,batch,augmentations):
        (loss,aux),gradients=loss_grad(parameters,batch,augmentations)
        updates,new_state=optimizer.update(gradients,optimizer_state,parameters)
        return optax.apply_updates(parameters,updates),new_state,loss,aux
    @jax.jit
    def forecast(parameters,x,t,action):
        maps=model.encoder(parameters,x[:3])
        keypoints,_=util.map_to_keypoints(maps)
        predicted=predict(model,parameters,t,keypoints,action)
        images,_=model.renderer(parameters,predicted)
        return images
    (root/'checkpoints').mkdir(exist_ok=True)
    history=[];updates_done=0
    write_status(root,'training',epoch=0,total_epochs=40,updates=0)
    print('Official loss, Euler training, 40 epochs, 450 independent training runs.',flush=True)
    for epoch in range(40):
        totals=[];parts=[];epoch_start=time.monotonic()
        for i,index in enumerate(rng.permutation(len(train))):
            batch={key:value[None] for key,value in train[index].items()}
            augmentation=rng.randint(0,8,(1,50)).astype(np.int32)
            params,state,loss,aux=step(params,state,batch,augmentation)
            value=float(loss)
            if not np.isfinite(value):
                write_status(root,'failed',epoch=epoch+1,updates=updates_done,reason='Non-finite official training loss')
                raise RuntimeError('Non-finite official training loss; no algorithm substitution performed')
            updates_done+=1;totals.append(value);parts.append({name:float(item) for name,item in aux.items()})
            if (i+1)%100==0:
                write_status(root,'training',epoch=epoch+1,total_epochs=40,batch=i+1,batches=450,updates=updates_done,current_loss=value)
                print('Epoch',epoch+1,'batch',i+1,'loss',round(value,5),flush=True)
        # Exact paper rollout solver, all 50 validation sequences, no gradients.
        predictions=[]
        for run in validation:
            output=np.asarray(forecast(params,run['x'],run['t'],run['action']))
            predictions.append({'x_recon':output})
        vpt=util.calculate_vpt(thresholds['paper_validation_train_plus_validation'],validation,predictions)
        row={'epoch':epoch+1,'updates':updates_done,'loss':float(np.mean(totals)),
             **{name:float(np.mean([part[name] for part in parts])) for name in parts[0]},
             'validation_vpt_mean':float(vpt[0]),'validation_vpt_std':float(vpt[1]),
             'epoch_seconds':time.monotonic()-epoch_start,'elapsed_seconds':time.monotonic()-start,
             'validation_outputs_finite':all(np.isfinite(p['x_recon']).all() for p in predictions)}
        history.append(row)
        with (root/'checkpoints'/f'epoch_{epoch+1:02d}.p').open('wb') as handle:pickle.dump(jax.device_get(params),handle)
        (root/'training_history.json').write_text(json.dumps(history,indent=2)+'\n')
        write_status(root,'training',total_epochs=40,**row)
        print(json.dumps(row),flush=True)
    with (root/'trained_params.p').open('wb') as handle:pickle.dump(jax.device_get(params),handle)
    report={'training_performed':True,'epochs_completed':40,'optimizer_updates':updates_done,'seed':42,
            'parameter_count':sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)),
            'devices':[str(device) for device in jax.devices()],'elapsed_seconds':time.monotonic()-start,
            'optimizer_state_repair':'state returned by Optax is retained between updates; pinned upstream code drops it',
            'test_data_used_for_training_or_selection':False,'checkpoint_selection':'predeclared final epoch 40',
            'trained_checkpoint_sha256':hashlib.sha256((root/'trained_params.p').read_bytes()).hexdigest()}
    (root/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    write_status(root,'training_complete',**report)
    print('Training completed:',json.dumps(report),flush=True)


if __name__=='__main__':main()
