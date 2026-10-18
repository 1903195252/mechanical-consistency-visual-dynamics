"""Post-hoc, paired full-budget Euler / RK4 controls of frozen CON architecture."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import signal
import time
from experiments.runtime import deadline

from .common import OUT, data, jax, jnp, state_model, setup, sha, write, save
import numpy as np
import optax
from src.training.optim import create_learning_rate_fn

EXT = OUT / 'dynamics_extension'
STOP = False

def stop(*unused):
    global STOP
    STOP = True

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def fixed_history(network, parameters, initial, command, solver, intervals=48):
    def field(y):
        return network.apply({'params': parameters}, y, command)
    def interval(y, unused):
        if solver == 'euler':
            y = y + field(y) / 30
        else:
            def substep(unused, y):
                h = 1 / 120
                k1 = field(y)
                k2 = field(y + h*k1/2)
                k3 = field(y + h*k2/2)
                k4 = field(y + h*k3)
                return y + h*(k1 + 2*k2 + 2*k3 + k4)/6
            y = jax.lax.fori_loop(0, 4, substep, y)
        return y, y
    _, history = jax.lax.scan(interval, initial, None, length=intervals)
    return jnp.concatenate([initial[None], history])


def run(solver, seed):
    protocol, manifest = setup(); cfg = protocol['training']
    extension = json.loads((EXT/'protocol.json').read_text())
    assert solver in ('euler', 'rk4') and seed in cfg['seeds']
    arm = f'state_{solver}_con_iae'
    folder = EXT/'runs'/f'{arm}_seed{seed}'; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'training_report.json').exists():
        return
    started = time.monotonic()
    identity = {'protocol':sha(EXT/'protocol.json'), 'parent_protocol':sha(OUT/'protocol.json'),
                'data':sha(OUT/'dataset_manifest.json'), 'train_source':sha(Path(__file__)),
                'common_source':sha(Path(__file__).with_name('common.py'))}
    def status(stage, **values):
        write(folder/'status.json', dict(stage=stage, arm=arm, seed=seed, elapsed_s=time.monotonic()-started, **values))
    status('loading')
    records = [r for r in manifest['records'] if r['split']=='train']
    examples = [data(r, state=True) for r in records]
    arrays = {k:np.stack([x[k] for x in examples]) for k in ('points','velocity','tau')}
    del examples
    network=state_model(); rng=jax.random.PRNGKey(seed); generator=np.random.RandomState(seed)
    parameters=network.init(rng, jnp.zeros(16), jnp.zeros(14), method=network.forward_all_layers)['params']
    save(folder/'initial_params.p', parameters)
    initial_hash = sha(folder/'initial_params.p')
    steps_per_epoch=math.ceil(len(records)/cfg['batch_size'])
    schedule=create_learning_rate_fn(cfg['epochs'], steps_per_epoch, cfg['base_lr'], warmup_epochs=cfg['warmup_epochs'])
    optimizer=optax.adamw(schedule, weight_decay=cfg['weight_decay']); opt_state=optimizer.init(parameters)
    def objective(p, batch):
        def solve(x,v,u):
            y=jnp.r_[x[1],v[1]].astype(jnp.float64)
            return fixed_history(network,p,y,u,solver)[:,:8].astype(jnp.float32)
        history=jax.vmap(solve)(batch['points'],batch['velocity'],batch['tau'])
        mse=jnp.mean((history-batch['points'][:,1:])**2)
        rec=jax.vmap(lambda u:network.apply({'params':p},u,method=network.autoencode_input))(batch['tau'])
        input_error=jnp.mean((rec-batch['tau'])**2)
        return cfg['loss_weights']['mse_z']*mse + cfg['loss_weights']['mse_tau_rec']*input_error
    @jax.jit
    def step(p,opt,b):
        loss,grad=jax.value_and_grad(objective)(p,b)
        updates,opt=optimizer.update(grad,opt,p)
        return optax.apply_updates(p,updates),opt,loss
    checkpoint=folder/'resume.p'; epoch=cursor=updates=samples=0; order=None; history=[]; losses=[]; order_digests=[]
    if checkpoint.exists():
        with checkpoint.open('rb') as handle:
            old=pickle.load(handle)
        assert old['identity']==identity
        parameters,opt_state=old['params'],old['opt_state']; generator.set_state(old['numpy_rng'])
        epoch,cursor,updates,samples,order,history,losses,order_digests=[old[k] for k in ('epoch','cursor','updates','samples','order','history','losses','order_digests')]
    def checkpoint_now():
        save(checkpoint,dict(params=parameters,opt_state=opt_state,numpy_rng=generator.get_state(),epoch=epoch,cursor=cursor,
                            updates=updates,samples=samples,order=order,history=history,losses=losses,order_digests=order_digests,identity=identity))
    status('compiling',updates=updates)
    for current in range(epoch,cfg['epochs']):
        epoch=current
        if order is None:
            order=generator.permutation(len(records)); cursor=0; losses=[]
            order_digests.append(hashlib.sha256(order.astype('<i8').tobytes()).hexdigest())
        while cursor<len(order):
            if STOP or time.time()>=deadline()-90:
                checkpoint_now(); status('paused_at_budget',epoch=epoch+1,updates=updates); return
            indices=order[cursor:cursor+cfg['batch_size']]
            batch={k:jnp.asarray(v[indices]) for k,v in arrays.items()}; tick=time.monotonic()
            parameters,opt_state,loss=step(parameters,opt_state,batch); loss=float(loss)
            if not np.isfinite(loss):
                status('failed',reason='nonfinite full-budget objective',epoch=epoch+1,updates=updates)
                raise RuntimeError('Nonfinite objective; no clipping, skipped records or replacement scores')
            updates+=1; samples+=len(indices); cursor+=len(indices); losses.append(loss)
            status('training',epoch=epoch+1,epochs=cfg['epochs'],updates=updates,target_updates=steps_per_epoch*cfg['epochs'],
                   loss=loss,last_step_s=time.monotonic()-tick,samples_seen=samples)
            if updates%20==0:
                checkpoint_now()
        history.append(dict(epoch=epoch+1,updates=updates,mean_batch_loss=float(np.mean(losses)),samples_seen=samples))
        write(folder/'history.json',history); print(json.dumps(history[-1]),flush=True)
        epoch=current+1; order=None; cursor=0; losses=[]; checkpoint_now()
    save(folder/'params.p',parameters)
    write(folder/'training_report.json',dict(arm=arm,seed=seed,epochs=cfg['epochs'],updates=updates,expected_updates=steps_per_epoch*cfg['epochs'],
          samples_seen=samples,parameters=sum(x.size for x in jax.tree.leaves(parameters)),checkpoint_sha256=sha(folder/'params.p'),
          initial_checkpoint_sha256=initial_hash,epoch_data_order_sha256=order_digests,identity=identity,privileged_state=True,
          solver=solver,substeps=1 if solver=='euler' else 4,elapsed_s=time.monotonic()-started,devices=[str(d) for d in jax.devices()]))
    status('training_complete',epoch=cfg['epochs'],updates=updates)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--solver',choices=['euler','rk4'],required=True); parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args(); run(args.solver,args.seed)
