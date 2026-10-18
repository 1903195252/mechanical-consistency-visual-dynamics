"""Explicit geometry-conditioned crank readout, shared by both frozen models.

Only predicted B and its derivatives plus public geometry enter this map.
This intervention is never substituted for the original raw model trajectory.
"""
import numpy as np
from experiments.keycld_corpus.physics_score import MODULES
from experiments.keycld_corpus.prepare import point_states


def reconstruct(x,v,a,c):
    if not all(np.isfinite(z).all() for z in (x,v,a)):
        raise ValueError('nonfinite_native_prediction')
    r=x[:,0]-[0,c['pivot_height_m']];u=v[:,0];b=a[:,0];square=np.sum(r*r,axis=1)
    if np.any(square<1e-6):
        raise ValueError('predicted_crank_direction_singular')
    cross=lambda p,q:p[:,0]*q[:,1]-p[:,1]*q[:,0]
    theta=np.arctan2(r[:,1],r[:,0]);omega=cross(r,u)/square
    alpha=cross(r,b)/square-2*np.sum(r*u,axis=1)*omega/square
    module=MODULES[c['category']]
    q=np.array([module.inverse_kinematics(t,c) for t in theta])
    J=np.array([module.closure_jacobian(p,c) for p in q]);chart=J[:,:,1:]
    if np.any(np.linalg.cond(chart)>1e10):raise ValueError('closure_coordinate_singular')
    gamma=np.column_stack([np.ones(len(q)),np.linalg.solve(chart,-J[:,:,0,None])[:,:,0]])
    curvature=np.array([module.closure_curvature(p,n,c) for p,n in zip(q,gamma)])
    gamma_prime=np.column_stack([np.zeros(len(q)),np.linalg.solve(chart,-curvature[:,:,None])[:,:,0]])
    qd=gamma*omega[:,None];qdd=gamma*alpha[:,None]+gamma_prime*omega[:,None]**2
    return point_states(q,qd,qdd,c),dict(q=q,qdot=qd,qddot=qdd)
