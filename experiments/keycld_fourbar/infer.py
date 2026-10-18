"""Run the official KeyCLD modules with four-bar geometry and random weights.

This is a functional diagnostic, never a trained-model accuracy experiment.
Only model_input.npz and geometry.json are available to this process.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('WANDB_MODE','disabled')
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from keycld import util
from keycld.models import KeyCLD,predict


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();root=args.directory;start=time.monotonic()
    with np.load(root/'model_input.npz') as saved:
        data={name:saved[name] for name in saved.files}
    g=json.loads((root/'geometry.json').read_text())
    assert data['observed_x'].shape==(3,64,64,3)
    assert not any(name in data for name in ['q','positions','velocities','future_x'])
    A=jnp.asarray(g['A_normalized']);D=jnp.asarray(g['D_normalized']);lengths=jnp.asarray(g['link_lengths_normalized'])
    def constraint_fn(state):
        B,C=state.reshape(2,2)
        return jnp.array([jnp.sum((B-A)**2)-lengths[0]**2,
                          jnp.sum((C-B)**2)-lengths[1]**2,
                          jnp.sum((C-D)**2)-lengths[2]**2])
    model=KeyCLD(2,3,32,True,constraint_fn,image_size=64)
    params=model.init(jax.random.PRNGKey(42))
    inputs=jnp.asarray(data['observed_x']);times=jnp.asarray(data['time'])
    action_time=jnp.asarray(data['action_time']);actions=jnp.asarray(data['actions'])
    def action_at(t):
        return jnp.stack([jnp.interp(t,action_time,actions[:,i]) for i in range(3)])
    def solver(f,state,t,unused_action):
        def interval(carry,interval):
            left,right=interval;dt=(right-left)/4
            def substep(i,y):
                time=left+dt*i
                k1=f(y,time,action_at(time));k2=f(y+dt*k1/2,time+dt/2,action_at(time+dt/2))
                k3=f(y+dt*k2/2,time+dt/2,action_at(time+dt/2));k4=f(y+dt*k3,time+dt,action_at(time+dt))
                return y+dt*(k1+2*k2+2*k3+k4)/6
            end=jax.lax.fori_loop(0,4,substep,carry)
            return end,end
        _,history=jax.lax.scan(interval,state,(t[:-1],t[1:]))
        return jnp.concatenate([state[None],history])
    @jax.jit
    def forecast(parameters):
        maps=model.encoder(parameters,inputs)
        observed,_=util.map_to_keypoints(maps)
        history_store=[]
        def capture_solver(f,state,t,action):
            history=solver(f,state,t,action)
            history_store.append(history)
            return history
        predicted=predict(model,parameters,times,observed,actions[0],solver=capture_solver)
        rendered,_=model.renderer(parameters,predicted)
        # Retain the very same states used by the official prediction function.
        history=history_store[0]
        states=jnp.concatenate([jnp.concatenate([observed[0].ravel(),history[0,4:]])[None],history])
        acceleration=jax.vmap(lambda y,t:model.euler_lagrange(parameters,y[:4],y[4:],action_at(t)))(states,times)
        return observed,predicted,rendered,states[:,4:].reshape(-1,2,2),acceleration.reshape(-1,2,2)
    print('Devices:',jax.devices(),flush=True)
    print('Compiling official KeyCLD four-bar forecast, random initialization, no optimizer steps...',flush=True)
    observed,predicted,images,velocities,accelerations=jax.device_get(forecast(params))
    result=dict(time=data['time'],observed_keypoints=observed,keypoints_normalized=predicted,
                keypoint_velocity_normalized_s=velocities,keypoint_acceleration_normalized_s2=accelerations,x_recon=images)
    finite={name:bool(np.isfinite(value).all()) for name,value in result.items()}
    np.savez_compressed(root/'prediction.npz',**result)
    report={'purpose':'input-and-scoring diagnostic only','research_accuracy_result':False,
            'weights':'random initialization, seed 42, zero optimizer updates','training_performed':False,
            'official_source_commit':'426c99a53d597e81044e4ce77214cd3aa6b92b64',
            'num_keypoints':2,'num_action_dimensions':3,'num_hidden_dimensions':32,
            'input_observed_frames':3,'prediction_frames':len(times),'geometry_given':True,
            'physics_parameters_given':False,'friction_model':'absent in the official learned force architecture',
            'parameter_count':int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))),
            'solver':'RK4, four substeps per video interval; actual time-varying actions',
            'jax_version':jax.__version__,'devices':[str(device) for device in jax.devices()],
            'finite_outputs':finite,'elapsed_seconds':round(time.monotonic()-start,3)}
    (root/'inference_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
