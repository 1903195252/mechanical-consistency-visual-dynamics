"""Independent SI mechanics for 4R and 3R+1P; no simulator or learned model imports."""
import numpy as np
from mechanics import (properties,payload_inertia,inverse_kinematics,closure_jacobian,
                       closure_curvature,direction,tangent,JOINT_RATE_MAP)


def is_slider(c):return c.get('mechanism')=='crank_slider'


def inverse(theta,c):
    if not is_slider(c):return inverse_kinematics(theta,c)
    l1,l2=c['lengths_m'][:2]
    horizontal=np.sqrt(l2*l2-(l1*np.sin(theta))**2)
    return np.array([theta,np.arctan2(-l1*np.sin(theta),horizontal),l1*np.cos(theta)+horizontal])


def closure(q,c):
    l1,l2,l3=c['lengths_m']
    endpoint=l1*direction(q[0])+l2*direction(q[1])
    if is_slider(c):return endpoint-np.array([q[2],0.])
    return endpoint-l3*direction(q[2])-np.array([c['ground_length_m'],0.])


def jacobian(q,c):
    if not is_slider(c):return closure_jacobian(q,c)
    l1,l2=c['lengths_m'][:2]
    return np.column_stack([l1*tangent(q[0]),l2*tangent(q[1]),np.array([-1.,0.])])


def curvature(q,v,c):
    if not is_slider(c):return closure_curvature(q,v,c)
    l1,l2=c['lengths_m'][:2]
    return -l1*direction(q[0])*v[0]**2-l2*direction(q[1])*v[1]**2


def derivatives(q,c):
    J=jacobian(q,c);gamma=np.r_[1.,np.linalg.solve(J[:,1:],-J[:,0])]
    gamma2=np.r_[0.,np.linalg.solve(J[:,1:],-curvature(q,gamma,c))]
    return gamma,gamma2


def bodies(q,v,c):
    """Mass-linear M,c,G basis, using specified uniform shapes and COMs."""
    l1,l2,l3=c['lengths_m'];masses,inertias=properties(c);zero=np.zeros(2)
    J=[np.column_stack([l1/2*tangent(q[0]),zero,zero]),
       np.column_stack([l1*tangent(q[0]),l2/2*tangent(q[1]),zero])]
    k=[-l1/2*direction(q[0])*v[0]**2,
       -l1*direction(q[0])*v[0]**2-l2/2*direction(q[1])*v[1]**2]
    if is_slider(c):
        J.append(np.column_stack([zero,zero,np.array([1.,0.])]))
        k.append(zero);masses[2]=c['slider_mass_kg']
    else:
        J.append(np.column_stack([zero,zero,l3/2*tangent(q[2])]))
        k.append(-l3/2*direction(q[2])*v[2]**2)
    along=c['payload_along_coupler_m'];normal=c['payload_normal_offset_m']
    J.append(np.column_stack([l1*tangent(q[0]),along*tangent(q[1])-normal*direction(q[1]),zero]))
    k.append(-l1*direction(q[0])*v[0]**2-(along*direction(q[1])+normal*tangent(q[1]))*v[1]**2)
    ratios=[inertias[0,1]/masses[0],inertias[1,1]/masses[1],
            0. if is_slider(c) else inertias[2,1]/masses[2],
            payload_inertia(c)[1]/c['payload_mass_kg']]
    axes=[0,1,2,1];mass=np.r_[masses,c['payload_mass_kg']];components=[]
    gravity=np.array([0.,-c['gravity_m_s2']])
    for index,(Jbody,centripetal,ratio,axis) in enumerate(zip(J,k,ratios,axes)):
        rotational=np.zeros((3,3));rotational[axis,axis]=ratio
        components.append((Jbody.T@Jbody+rotational,Jbody.T@centripetal,-Jbody.T@gravity))
    return mass,components


def mass_bias(q,v,c):
    mass,components=bodies(q,v,c)
    M=sum(m*part[0] for m,part in zip(mass,components))
    M[0,0]+=c['motor_armature_kg_m2']
    C=sum(m*part[1] for m,part in zip(mass,components))
    G=sum(m*part[2] for m,part in zip(mass,components))
    return M,C,G


def rate_map(c):
    return np.array([[1.,0.,0.],[-1.,1.,0.],[0.,1.,0.],[0.,0.,1.]]) if is_slider(c) else JOINT_RATE_MAP


def friction(qv,c):
    R=rate_map(c);rates=R@qv;b=np.asarray(c['viscous_friction_Nm_s_rad']);fc=np.asarray(c['coulomb_friction_Nm'])
    return -R.T@(b*rates+fc*np.tanh(rates/c['coulomb_smoothing_rad_s']))


def external(q,c):
    l1=c['lengths_m'][0];along=c['force_application_along_coupler_m']
    J=np.column_stack([l1*tangent(q[0]),along*tangent(q[1]),np.zeros(2)])
    return J.T@np.asarray(c['external_force_xz_N'])


def terms(q,v,a,tau,c):
    gamma,_=derivatives(q,c);M,C,G=mass_bias(q,v,c)
    return {'inertia_Nm':float(gamma@(M@a)), 'velocity_quadratic_Nm':float(gamma@C),
            'gravity_Nm':float(gamma@G), 'friction_Nm':float(-gamma@friction(v,c)),
            'external_Nm':float(gamma@external(q,c)), 'motor_Nm':float(tau)}


def acceleration(theta,omega,tau,c):
    q=inverse(theta,c);gamma,gamma2=derivatives(q,c);v=gamma*omega
    M,C,G=mass_bias(q,v,c)
    Q=np.array([tau,0.,0.])+friction(v,c)+external(q,c)
    return float(gamma@(Q-C-G-M@gamma2*omega**2)/(gamma@M@gamma))


def point_states(q,v,a,c):
    l1,l2=c['lengths_m'][:2];h=c['pivot_height_m']
    B=l1*direction(q[0])+[0,h];C=B+l2*direction(q[1])
    vB=l1*tangent(q[0])*v[0];vC=vB+l2*tangent(q[1])*v[1]
    aB=l1*(tangent(q[0])*a[0]-direction(q[0])*v[0]**2)
    aC=aB+l2*(tangent(q[1])*a[1]-direction(q[1])*v[1]**2)
    mid=lambda u,w:np.stack([u,w,(u+w)/2])
    return mid(B,C),mid(vB,vC),mid(aB,aC)


def decode(x,v,a,c):
    A=np.array([0.,c['pivot_height_m']]);D=np.array([c['ground_length_m'],c['pivot_height_m']])
    r=np.stack([x[0]-A,x[1]-x[0],x[1]-D]);rv=np.stack([v[0],v[1]-v[0],v[1]]);ra=np.stack([a[0],a[1]-a[0],a[1]])
    radii=np.linalg.norm(r,axis=1)
    if np.any(radii[:2]<.001) or (not is_slider(c) and radii[2]<.001):return None
    cross=lambda u,w:u[:,0]*w[:,1]-u[:,1]*w[:,0]
    safe=np.maximum(radii**2,1e-30);q=np.arctan2(r[:,1],r[:,0]);qd=cross(r,rv)/safe
    qdd=cross(r,ra)/safe-2*np.sum(r*rv,axis=1)*cross(r,rv)/safe**2
    if is_slider(c):q[2]=x[1,0];qd[2]=v[1,0];qdd[2]=a[1,0]
    return q,qd,qdd


def geometry_error(x,c):
    A=np.array([0.,c['pivot_height_m']]);D=np.array([c['ground_length_m'],c['pivot_height_m']])
    third=x[1,1]-c['pivot_height_m'] if is_slider(c) else np.linalg.norm(x[1]-D)-c['lengths_m'][2]
    return np.r_[np.linalg.norm(x[0]-A)-c['lengths_m'][0],np.linalg.norm(x[1]-x[0])-c['lengths_m'][1],third,x[2]-(x[0]+x[1])/2]


def regressor(q,v,a,c):
    gamma,_=derivatives(q,c);mass,components=bodies(q,v,c)
    columns=[float(gamma@(part[0]@a+part[1]+part[2])) for part in components]
    R=rate_map(c);rates=R@v;s=R@gamma
    columns.extend(s*rates);columns.extend(s*np.tanh(rates/c['coulomb_smoothing_rad_s']))
    known_rotor=c['motor_armature_kg_m2']*a[0]
    return np.asarray(columns),known_rotor,float(gamma@external(q,c)),np.r_[mass,c['viscous_friction_Nm_s_rad'],c['coulomb_friction_Nm']]
