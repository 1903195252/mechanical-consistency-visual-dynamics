"""Separate given-constraint curvature from learned generalized driving force.

True-coordinate probes are diagnostic interventions, not video-only scores.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint

from keycld import util
from learned_common import load_model
from train_paper import load_observation


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    root,out=args.root,args.out
    sys.path.insert(0,str(root/'source'));sys.path.insert(0,str(root/'source/fourbar'))
    from prepare import points
    from mechanics import closure_jacobian,closure_curvature,reduced_acceleration
    from diagnose import rmse,summary,write_json
    base=json.loads((out/'report.json').read_text())
    assert hashlib.sha256((root/'trained_params.p').read_bytes()).hexdigest()==base['checkpoint_sha256']
    with (root/'trained_params.p').open('rb') as handle:trained=pickle.load(handle)
    with (root/'initial_params.p').open('rb') as handle:initial=pickle.load(handle)
    no_force=dict(trained)
    for name in ['potential_energy','input_matrix']:
        no_force[name]=jax.tree_util.tree_map(jnp.zeros_like,trained[name])
    conditions={'trained':trained,'given_constraints_zero_force':no_force,'random_initialization':initial}
    model=load_model(root);geometry=json.loads((root/'geometry.json').read_text())
    scale=geometry['half_extent_m'];center=np.asarray(geometry['world_xz_center_m'])
    acceleration=jax.jit(jax.vmap(model.euler_lagrange,in_axes=(None,0,0,None)))
    @jax.jit
    def rollout(parameters,x,v,t,action):
        state=jnp.concatenate([x.ravel(),v.ravel()])
        history=odeint(partial(model.ode,parameters),state,t[1:],action)
        return history[:,:6].reshape(-1,3,2)

    rows=[]
    for index in range(500,550):
        with np.load(root/'truth'/f'run_{index:04d}.npz') as saved:truth={k:saved[k] for k in saved.files}
        c=json.loads((root/'truth'/f'run_{index:04d}.json').read_text());run=load_observation(root,index)
        xy,v,a=points(truth['q'],truth['qdot'],truth['qddot'],c)
        mid=lambda array:np.concatenate([array,array.mean(axis=1,keepdims=True)],axis=1)
        x=(mid(xy)-center)/scale;v=mid(v)/scale;a=mid(a)/scale
        tangent=[];curvature=[];static_alpha=[];moving_alpha=[]
        for q,omega in zip(truth['q'],truth['qdot'][:,0]):
            jac=closure_jacobian(q,c);gamma=np.r_[1.,np.linalg.solve(jac[:,1:],-jac[:,0])]
            gamma2=np.r_[0.,np.linalg.solve(jac[:,1:],-closure_curvature(q,gamma,c))]
            _,unit_v,unit_a=points(q[None],gamma[None],gamma2[None],c)
            tangent.append(mid(unit_v)[0]/scale);curvature.append(mid(unit_a)[0]/scale)
            static_alpha.append(reduced_acceleration(q[0],0.,float(run['action'][0]),c))
            moving_alpha.append(reduced_acceleration(q[0],float(omega),float(run['action'][0]),c))
        tangent=np.asarray(tangent);curvature=np.asarray(curvature)
        static_alpha=np.asarray(static_alpha);moving_alpha=np.asarray(moving_alpha)
        static_a=tangent*static_alpha[:,None,None]
        values={}
        for name,parameters in conditions.items():
            moving=np.asarray(acceleration(parameters,x.reshape(50,6).astype(np.float32),
                                           v.reshape(50,6).astype(np.float32),run['action'])).reshape(50,3,2)
            static=np.asarray(acceleration(parameters,x.reshape(50,6).astype(np.float32),
                                           np.zeros((50,6),dtype=np.float32),run['action'])).reshape(50,3,2)
            infer=lambda predicted,curvature_term:np.sum(tangent*(predicted-curvature_term),axis=(1,2))/np.sum(tangent**2,axis=(1,2))
            alpha=infer(moving,curvature*truth['qdot'][:,0,None,None]**2)
            alpha_static=infer(static,0.)
            values[name]={'full_point_acceleration_relative_error':float(np.linalg.norm(moving-a)/np.linalg.norm(a)),
                          'moving_crank_acceleration_relative_error':float(np.linalg.norm(alpha-moving_alpha)/np.linalg.norm(moving_alpha)),
                          'zero_velocity_crank_acceleration_relative_error':float(np.linalg.norm(alpha_static-static_alpha)/np.linalg.norm(static_alpha)),
                          'zero_velocity_crank_acceleration_rmse_rad_s2':rmse(alpha_static,static_alpha),
                          'zero_velocity_point_acceleration_relative_error':float(np.linalg.norm(static-static_a)/np.linalg.norm(static_a))}
            if name!='random_initialization':
                forecast=np.asarray(rollout(parameters,x[1].astype(np.float32),v[1].astype(np.float32),run['t'],run['action']))
                valid=bool(np.isfinite(forecast).all())
                values[name]['oracle_initialization_rollout_finite']=valid
                values[name]['oracle_initialization_position_rmse_m']=rmse(forecast[2:,:2],x[3:,:2])*scale if valid else None
        rows.append({'index':index,'conditions':values})
    fields=list(rows[0]['conditions']['trained'])
    result={'purpose':'Given curvature versus learned driving-force diagnostic on exact physical states',
            'run_count':50,'checkpoint_sha256':base['checkpoint_sha256'],
            'intervention':'Zero only potential and input-network weights; keep learned mass and all constraints. Also set velocity to zero at the same poses.',
            'semantic_limit':'Exact physical coordinates are outside the learned encoder semantics; these are capability probes, not causal video-only dynamics scores.',
            'oracle_initialization':'Exact physical x and v at the middle context timestamp; keep the learned equation, never substitute the correct physical equation.',
            'summary':{name:{field:summary([r['conditions'][name][field] for r in rows]) for field in fields if field in rows[0]['conditions'][name]} for name in conditions},
            'rows':rows,'training_performed':False}
    write_json(out/'force_probe.json',result)
    print(json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
