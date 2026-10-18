"""Independent force-equation scoring of physical-coordinate diagnostic models."""
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
from keycld.models import KeyCLD
from learned_common import load_model
from diagnose import summary,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();root,out=args.root,args.out
    sys.path.insert(0,str(root/'source'));sys.path.insert(0,str(root/'source/fourbar'))
    from prepare import points
    from score import metrics
    model=load_model(root);g=json.loads((root/'geometry.json').read_text());scale=g['half_extent_m'];center=np.asarray(g['world_xz_center_m'])
    @jax.jit
    def forecast(parameters,x,v,t,action):
        initial=jnp.concatenate([x[1].ravel(),util.project_velocity(model.constraint_fn,x[1].ravel(),v[1].ravel())])
        history=odeint(partial(model.ode,parameters),initial,t[1:],action)
        all_state=jnp.concatenate([jnp.concatenate([x[0].ravel(),v[0].ravel()])[None],history])
        acceleration=jax.vmap(lambda y:model.euler_lagrange(parameters,y[:6],y[6:],action))(all_state)
        return all_state[:,:6].reshape(50,3,2),all_state[:,6:].reshape(50,3,2),acceleration.reshape(50,3,2)
    results={};examples={}
    for arm in ['euler','rk4_4']:
        report=json.loads((out/arm/'report.json').read_text())
        assert report['epochs_completed']==40 and report['optimizer_updates']==18000
        assert hashlib.sha256((out/arm/'params.p').read_bytes()).hexdigest()==report['checkpoint_sha256']
        with (out/arm/'params.p').open('rb') as handle:params=pickle.load(handle)
        rows=[]
        for index in range(500,550):
            with np.load(root/'truth'/f'run_{index:04d}.npz') as saved:truth={k:saved[k] for k in saved.files}
            c=json.loads((root/'truth'/f'run_{index:04d}.json').read_text())
            with np.load(root/'observations'/f'run_{index:04d}.npz') as saved:action=saved['action'];t=saved['t']
            xy,v,a=points(truth['q'],truth['qdot'],truth['qddot'],c)
            mid=lambda arr:np.concatenate([arr,arr.mean(axis=1,keepdims=True)],axis=1)
            x=(mid(xy)-center)/scale;v=mid(v)/scale
            predicted,pred_v,pred_a=jax.device_get(forecast(params,x.astype(np.float32),v.astype(np.float32),t,action))
            finite=all(np.isfinite(array).all() for array in [predicted,pred_v,pred_a])
            row={'index':index,'finite':bool(finite)}
            if finite:
                actions=np.tile([action[0],*c['external_force_xz_N']],(50,1))
                values,_=metrics(predicted[:,:2]*scale+center,pred_v[:,:2]*scale,pred_a[:,:2]*scale,
                                 truth['time'],actions,{'keypoints_world_xz':xy,'q':truth['q']},c)
                row['physics']=values
            rows.append(row)
            if index==500:
                examples[arm+'_predicted_world_xz']=predicted[:,:2]*scale+center
                examples['reference_world_xz']=xy;examples['time']=truth['time']
        keys=['keypoint_position_rmse_m','native_link_length_rmse_m','native_link_length_max_error_m',
              'independent_tangent_force_residual_rmse_Nm','angular_closure_max_m']
        results[arm]={'run_count':50,'finite_runs':sum(r['finite'] for r in rows),
                      'summary':{key:summary([r.get('physics',{}).get(key) for r in rows]) for key in keys},
                      'geometry_consistent_future_frames':sum(r.get('physics',{}).get('geometry_consistent_future_frames',0) for r in rows),
                      'total_future_frames':50*47,'rows':rows}
        matrix=model.mass_matrix(params,jnp.zeros(6))
        diagonal=np.diag(np.asarray(matrix))
        results[arm]['mass_parameter_audit']={
            'coordinate_mass_coefficients':diagonal[::2].tolist(),
            'max_to_min_ratio':float(diagonal.max()/diagonal.min()) if diagonal.min()>0 else None,
            'pinv_identity_max_abs_error':float(np.max(np.abs(np.asarray(jnp.linalg.pinv(matrix)@matrix)-np.eye(6)))),
            'note':'Coefficients are in normalized learned coordinates and have a mass/force scale gauge; not identified SI masses.'}
    # Unique critical-path audit of the analytical constraint-level issue:
    # double only the three given length constants, keeping state and weights.
    with (root/'trained_params.p').open('rb') as handle:baseline=pickle.load(handle)
    original_constraint=model.constraint_fn
    offset=jnp.r_[jnp.asarray(g['link_lengths_normalized']),jnp.zeros(2)]
    def doubled_lengths(state):return original_constraint(state)-offset
    altered=KeyCLD(3,3,32,True,doubled_lengths,image_size=64)
    with np.load(root/'truth/run_0500.npz') as saved:q=saved['q'];qd=saved['qdot'];qa=saved['qddot']
    c=json.loads((root/'truth/run_0500.json').read_text())
    xy,v,_=points(q,qd,qa,c);mid=lambda arr:np.concatenate([arr,arr.mean(axis=1,keepdims=True)],axis=1)
    x=((mid(xy)-center)/scale).reshape(50,6).astype(np.float32);v=(mid(v)/scale).reshape(50,6).astype(np.float32)
    with np.load(root/'observations/run_0500.npz') as saved:action=saved['action']
    original_a=jax.vmap(model.euler_lagrange,in_axes=(None,0,0,None))(baseline,x,v,action)
    altered_a=jax.vmap(altered.euler_lagrange,in_axes=(None,0,0,None))(baseline,x,v,action)
    jacobian=jax.vmap(jax.jacobian(original_constraint))(x)
    altered_jacobian=jax.vmap(jax.jacobian(doubled_lengths))(x)
    hessian=jax.vmap(jax.hessian(original_constraint))(x)
    altered_hessian=jax.vmap(jax.hessian(doubled_lengths))(x)
    metadata_audit={'index':500,'states_checked':50,'given_link_length_factor':2.,
                    'jacobian_max_abs_change':float(np.max(np.abs(np.asarray(jacobian-altered_jacobian)))),
                    'hessian_max_abs_change':float(np.max(np.abs(np.asarray(hessian-altered_hessian)))),
                    'acceleration_max_abs_change_m_s2':float(np.max(np.abs(np.asarray(original_a-altered_a))))*scale,
                    'position_constraint_max_abs_change_m':float(np.max(np.abs(np.asarray(jax.vmap(original_constraint)(x)-jax.vmap(doubled_lengths)(x)))))*scale,
                    'interpretation':'Only the equality level constants change. Their derivatives and the acceleration field do not. Correct physical lengths require a consistent initial state; anchors and topology still affect derivatives.'}
    assert metadata_audit['jacobian_max_abs_change']==0 and metadata_audit['hessian_max_abs_change']==0
    assert metadata_audit['acceleration_max_abs_change_m_s2']==0
    write_json(out/'physics_scores.json',{'conditions':results,'diagnostic_only':True,'length_metadata_audit':metadata_audit,
                'semantics':'True physical state labels supplied for training and initialization; independent SI equations used only by evaluator.',
                'geometry_tolerance_m':.01,'geometry_gate_note':'Legacy diagnostic tolerance; raw geometry errors are reported, not a publication acceptance threshold.'})
    np.savez_compressed(out/'trajectory_example.npz',**examples)
    print(json.dumps({arm:row['summary'] for arm,row in results.items()}),flush=True)


if __name__=='__main__':main()
