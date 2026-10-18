"""Full-budget component intervention, explicitly using physical-state labels."""
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
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental.ode import odeint
from keycld import util
from keycld.train import ExperimentBase
from learned_common import load_model
from diagnose import fixed_solver,summary,rmse,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--arm',choices=['euler','rk4_4'],required=True)
    args=parser.parse_args();root=args.root;out=args.out/args.arm;out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((args.out/'state_oracle_protocol.json').read_text())
    if (out/'report.json').exists():raise RuntimeError('Completed arm exists; preserve results')
    sys.path.insert(0,str(root/'source'));sys.path.insert(0,str(root/'source/fourbar'))
    from prepare import points
    from mechanics import closure_jacobian,reduced_acceleration
    model=load_model(root);geometry=json.loads((root/'geometry.json').read_text())
    scale=geometry['half_extent_m'];center=np.asarray(geometry['world_xz_center_m'])
    start=time.monotonic();rng=np.random.RandomState(42)
    def status(stage,**items):
        value={'stage':stage,'arm':args.arm,'elapsed_seconds':time.monotonic()-start,**items}
        write_json(out/'status.json',value);print(json.dumps(value),flush=True)
    def load(index):
        with np.load(root/'truth'/f'run_{index:04d}.npz') as saved:truth={k:saved[k] for k in saved.files}
        c=json.loads((root/'truth'/f'run_{index:04d}.json').read_text())
        with np.load(root/'observations'/f'run_{index:04d}.npz') as saved:t=saved['t'];action=saved['action']
        xy,v,a=points(truth['q'],truth['qdot'],truth['qddot'],c)
        mid=lambda arr:np.concatenate([arr,arr.mean(axis=1,keepdims=True)],axis=1)
        return {'x':((mid(xy)-center)/scale).reshape(50,6).astype(np.float32),
                'v':(mid(v)/scale).reshape(50,6).astype(np.float32),'t':t,'action':action},truth,c
    status('loading_oracle_training_data',epochs=0,updates=0)
    train=[load(i)[0] for i in range(450)];validation=[load(i)[0] for i in range(450,500)]
    with (root/'initial_params.p').open('rb') as handle:params=pickle.load(handle)
    initial_hash=hashlib.sha256((root/'initial_params.p').read_bytes()).hexdigest()
    experiment=ExperimentBase(num_epochs=40,learning_rate=.0003,batch_size=1,num_hidden_dim=32,
                              num_predicted_steps=4,dynamics_weight=1.,bce_weight=1.,solver='euler')
    experiment.configure_optimizers(params,18000);tx=experiment.tx;opt_state=experiment.opt_state
    solver=util.explicit_euler if args.arm=='euler' else fixed_solver('rk4',4)
    def objective(parameters,item):
        # Match official cropping exactly: 48 centred states minus 5 = 43 windows.
        x=item['x'][1:-1];v=item['v'][1:-1]
        v=jax.vmap(partial(util.project_velocity,model.constraint_fn))(x,v)
        initial=jnp.concatenate([x,v],axis=1)[:-5]
        times=jnp.arange(5)*(item['t'][1]-item['t'][0])
        history=jax.vmap(solver,in_axes=(None,0,None,None))(partial(model.ode,parameters),initial,times,item['action'])
        target=jnp.stack([x[i:i+5] for i in range(43)])
        return jnp.sum((history[:,:,:6]-target)**2)
    value_grad=jax.value_and_grad(objective)
    @jax.jit
    def step(parameters,state,item):
        loss,gradient=value_grad(parameters,item)
        updates,new_state=tx.update(gradient,state,parameters)
        return optax.apply_updates(parameters,updates),new_state,loss
    @jax.jit
    def forecast(parameters,item):
        x=item['x'][1];v=util.project_velocity(model.constraint_fn,x,item['v'][1])
        y=jnp.concatenate([x,v])
        history=odeint(partial(model.ode,parameters),y,item['t'][1:],item['action'])
        return history[:,:6].reshape(49,3,2)
    history=[];updates=0
    for epoch in range(40):
        losses=[]
        for j,index in enumerate(rng.permutation(450)):
            params,opt_state,loss=step(params,opt_state,train[index]);loss=float(loss)
            if not np.isfinite(loss):
                status('failed',epoch=epoch+1,updates=updates,reason='Non-finite training loss',index=int(index))
                raise RuntimeError('Non-finite state-oracle loss; no skipped runs or solver substitution')
            losses.append(loss);updates+=1
            if (j+1)%150==0:status('training',epoch=epoch+1,updates=updates,batch=j+1,total_epochs=40)
        errors=[];failures=0
        for item in validation:
            output=np.asarray(forecast(params,item));valid=np.isfinite(output).all()
            if valid:errors.append(rmse(output[2:,:2],item['x'][3:].reshape(47,3,2)[:,:2])*scale)
            else:failures+=1
        row={'epoch':epoch+1,'updates':updates,'dynamics_loss':float(np.mean(losses)),
             'validation_position_rmse_m':summary(errors),'validation_nonfinite_runs':failures,
             'elapsed_seconds':time.monotonic()-start}
        history.append(row);write_json(out/'history.json',history);status('epoch_complete',**row)
    with (out/'params.p').open('wb') as handle:pickle.dump(jax.device_get(params),handle)
    # No test data has been loaded until the predeclared final checkpoint exists.
    acceleration=jax.jit(jax.vmap(model.euler_lagrange,in_axes=(None,0,0,None)))
    rows=[]
    for index in range(500,550):
        item,truth,c=load(index);output=np.asarray(forecast(params,item));finite=bool(np.isfinite(output).all())
        target=item['x'][3:].reshape(47,3,2)
        jacobians=[];alpha=[]
        for q in truth['q']:
            jac=closure_jacobian(q,c);gamma=np.r_[1.,np.linalg.solve(jac[:,1:],-jac[:,0])]
            _,unit_v,_=points(q[None],gamma[None],np.zeros((1,3)),c)
            unit_v=np.concatenate([unit_v,unit_v.mean(axis=1,keepdims=True)],axis=1)
            jacobians.append(unit_v[0]/scale)
            alpha.append(reduced_acceleration(q[0],0.,float(item['action'][0]),c))
        tangent=np.asarray(jacobians);alpha=np.asarray(alpha)
        static=np.asarray(acceleration(params,item['x'],np.zeros((50,6),dtype=np.float32),item['action'])).reshape(50,3,2)
        prediction=np.sum(tangent*static,axis=(1,2))/np.sum(tangent**2,axis=(1,2))
        rows.append({'index':index,'finite':finite,'position_rmse_m':rmse(output[2:,:2],target[:,:2])*scale if finite else None,
                     'zero_velocity_crank_acceleration_relative_error':float(np.linalg.norm(prediction-alpha)/np.linalg.norm(alpha)),
                     'zero_velocity_crank_acceleration_rmse_rad_s2':rmse(prediction,alpha)})
    # Visual parameters must be untouched by a pure dynamics-component objective.
    with (root/'initial_params.p').open('rb') as handle:initial=pickle.load(handle)
    visual_unchanged=all(np.array_equal(a,b) for name in ['encoder','renderer'] for a,b in
                         zip(jax.tree_util.tree_leaves(initial[name]),jax.tree_util.tree_leaves(params[name])))
    assert visual_unchanged
    report={'protocol':protocol,'arm':args.arm,'epochs_completed':40,'optimizer_updates':updates,
            'initial_checkpoint_sha256':initial_hash,'checkpoint_sha256':hashlib.sha256((out/'params.p').read_bytes()).hexdigest(),
            'visual_weights_unchanged':visual_unchanged,'test_used_for_selection':False,'test_count':50,
            'finite_test_runs':sum(r['finite'] for r in rows),
            'summary':{key:summary([r[key] for r in rows]) for key in ['position_rmse_m','zero_velocity_crank_acceleration_relative_error','zero_velocity_crank_acceleration_rmse_rad_s2']},
            'rows':rows,'elapsed_seconds':time.monotonic()-start,'seed':42,'publication_baseline':False}
    write_json(out/'report.json',report);status('complete',epochs=40,updates=updates)


if __name__=='__main__':main()
