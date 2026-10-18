"""Force-driven spatial UPS simulation with independent Newton--Euler audits."""
import copy
import hashlib
from pathlib import Path
import time
import mujoco
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation

from . import stewart
from .catalog import write_json,OUTPUT


def source_digest():
    files=[Path(__file__),Path(stewart.__file__),Path(__file__).with_name('spatial_solids.py'),Path(__file__).with_name('solids.py')]
    fingerprint=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    folder=OUTPUT/'source_versions'/('physics_'+fingerprint);folder.mkdir(parents=True,exist_ok=True)
    for p in files:
        target=folder/p.name
        if not target.exists():target.write_bytes(p.read_bytes())
    return fingerprint


def motion_commands(times,c,condition):
    frequency={'low_speed':.18,'rated':.38,'limit_or_load':.65}[condition]
    phase=2*np.pi*frequency*np.asarray(times);ramp=c['actuator']['ramp_s']
    envelope=.5*(1-np.cos(np.pi*np.clip(np.asarray(times)/ramp,0,1)))
    translation=np.column_stack([.018*np.sin(phase),.014*np.sin(.8*phase+.4),.020*np.sin(1.1*phase+.7)])*envelope[:,None]
    angles=np.column_stack([.06*np.sin(.7*phase),.055*np.sin(.9*phase+.5),.09*np.sin(.6*phase+.2)])*envelope[:,None]
    R=Rotation.from_euler('xyz',angles).as_matrix()
    positions=translation+np.array([0.,0.,c['platform_height_m']])
    endpoints=positions[:,None,:]+np.einsum('tij,lj->tli',R,np.asarray(c['platform_anchors_m']))
    lengths=np.linalg.norm(endpoints-np.asarray(c['base_anchors_m'])[None],axis=-1)
    return lengths-np.asarray(c['initial_leg_lengths_m']),{'frequency_Hz':frequency,'ramp_s':ramp,
              'translation_amplitudes_m':[.018,.014,.020],'rotation_amplitudes_rad':[.06,.055,.09],
              'description':'Smooth six-axis reference used to compute commanded prismatic displacements; actual body poses are simulated.'}


def segment_distances(starts,ends):
    """Closest distances of nonadjacent finite leg axes (15 pairs)."""
    result=[]
    for i in range(6):
        for j in range(i+1,6):
            u=ends[i]-starts[i];v=ends[j]-starts[j];w=starts[i]-starts[j]
            aa=u@u;bb=u@v;cc=v@v;dd=u@w;ee=v@w;denom=aa*cc-bb*bb
            s=np.clip((bb*ee-cc*dd)/denom,0,1) if denom>1e-16 else 0.
            t=np.clip((bb*s+ee)/cc,0,1);s=np.clip((bb*t-dd)/aa,0,1);t=np.clip((bb*s+ee)/cc,0,1)
            result.append(float(np.linalg.norm(w+s*u-t*v)))
    return result


def simulate(assembly,condition,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);fingerprint=source_digest()
    c=copy.deepcopy(assembly['config'])
    if condition=='limit_or_load':c['payload_mass_kg']*=3.
    parts=stewart.inventory(c);bodies=stewart.solid_properties(parts);xml,q0=stewart.mjcf(c,bodies)
    (directory/'model.xml').write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);s=c['simulation'];dt=s['timestep_s']
    steps=round(s['duration_s']/dt);command_times=np.arange(2*steps+1)*(dt/2)
    command_positions,command_metadata=motion_commands(command_times,c,condition)
    command_velocities=np.gradient(command_positions,dt/2,axis=0,edge_order=2)
    actuation=np.zeros((24,6));slide_indices=np.arange(6)*3+8;actuation[slide_indices,np.arange(6)]=1
    _,J0=stewart.closure_jacobian(q0,c);_,_,Vh=np.linalg.svd(J0,full_matrices=True);null=Vh[18:].T
    _,bias0,_,_=stewart.mass_bias_energy(q0,np.zeros(24),c,bodies)
    feedforward=np.linalg.solve(null.T@actuation,null.T@(bias0-stewart.external(c)))
    if abs(feedforward).max()>=c['actuator']['force_limit_N']*.8:raise ValueError('Insufficient holding-force margin')
    act=c['actuator'];Qexternal=stewart.external(c)
    def command_index(t):return int(np.clip(round(t/(dt/2)),0,len(command_times)-1))
    def control(m,d):
        index=command_index(d.time)
        d.ctrl[:]=np.clip(feedforward+act['position_gain_N_m']*(command_positions[index]-d.qpos[9::3])+
                         act['velocity_gain_N_s_m']*(command_velocities[index]-d.qvel[8::3]),
                         -act['force_limit_N'],act['force_limit_N'])
        d.qfrc_applied[:]=stewart.friction(d.qvel,c)+Qexternal
    write_json(directory/'physical_parameters.json',{'config':c,'parts':parts,'bodies':bodies,'operating_condition':condition,
                                                     'feedforward_force_N':feedforward.tolist(),'motion_reference':command_metadata})
    write_json(directory/'controller_inputs.json',{'controller_calibration':act,'feedforward_force_N':feedforward.tolist(),
                                                  'known_external_force_world_N':c['external_force_world_N'],
                                                  'command_units':'six prismatic displacements in metres and velocities in metres/second',
                                                  'planned_command_metadata':command_metadata})
    mujoco.set_mjcb_control(control);records=[];video=[];ranks=set();mass_error=bias_error=jac_error=pose_error=0.
    dense=np.zeros((24,24));start=time.monotonic();minimum_leg_gap=np.inf;minimum_piston_margin=np.inf
    record_stride=round(1/(s['record_hz']*dt));video_stride=round(1/(s['fps']*dt))
    force_indices=np.r_[0:3,slide_indices];torque_indices=np.array([i for i in range(24) if i not in force_indices])
    try:
        for step in range(steps+1):
            if step%record_stride==0:
                mujoco.mj_forward(model,data);q=data.qpos.copy();v=data.qvel.copy();a=data.qacc.copy()
                kin=stewart.kinematics(q,v,c,bodies);M,bias,T,V=stewart.mass_bias_energy(q,v,c,bodies,kin)
                residual,J=stewart.closure_jacobian(q,c);rank=np.linalg.matrix_rank(J);ranks.add(int(rank))
                mujoco.mj_fullM(model,dense,data.qM)
                mass_error=max(mass_error,float(abs(M-dense).max()));bias_error=max(bias_error,float(abs(bias-data.qfrc_bias).max()))
                jac_error=max(jac_error,float(abs(J-data.efc_J.reshape(-1,24)).max()))
                for name,k in kin.items():
                    body_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
                    pose_error=max(pose_error,float(np.linalg.norm(k['position']-data.xpos[body_id])))
                Qfriction=stewart.friction(v,c);Qmotor=actuation@data.ctrl
                equation=M@a+bias-Qfriction-Qexternal-Qmotor-data.qfrc_constraint
                index=command_index(data.time)
                tip=np.array([kin['rod_%d'%i]['position'] for i in range(6)])
                minimum_leg_gap=min(minimum_leg_gap,min(segment_distances(np.asarray(c['base_anchors_m']),tip))-.042)
                L=np.asarray(c['initial_leg_lengths_m']);piston_bottom=.45*L+q[9::3]
                minimum_piston_margin=min(minimum_piston_margin,float((piston_bottom-.064).min()),
                                          float((.66*L-piston_bottom-.008).min()))
                records.append((data.time,q,v,a,data.ctrl.copy(),command_positions[index].copy(),command_velocities[index].copy(),
                                T,V,residual,J@v,equation,data.qfrc_constraint.copy(),data.efc_force.copy(),
                                float(Qmotor@v),float(Qfriction@v),float(Qexternal@v),float(data.qfrc_constraint@v)))
            if step%video_stride==0 and step<steps:
                mujoco.mj_forward(model,data);video.append((data.time,data.xpos.copy(),data.xquat.copy()))
            if step<steps:mujoco.mj_step(model,data)
    finally:mujoco.set_mjcb_control(None)
    names=['time','qpos','qvel','qacc','actual_actuator_force_N','command_prismatic_displacement_m','command_prismatic_velocity_m_s',
           'kinetic_J','potential_J','closure_m','closure_velocity_m_s','equation_residual','constraint_generalized_force',
           'ball_constraint_force_N','actuator_power_W','friction_power_W','external_power_W','constraint_power_W']
    arrays={name:np.asarray([row[i] for row in records]) for i,name in enumerate(names)}
    np.savez_compressed(directory/'physics.npz',**arrays)
    np.savez_compressed(directory/'commands.npz',time=arrays['time'],
                        commanded_prismatic_displacement_m=arrays['command_prismatic_displacement_m'],
                        commanded_prismatic_velocity_m_s=arrays['command_prismatic_velocity_m_s'])
    body_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or 'ground' for i in range(model.nbody)]
    np.savez_compressed(directory/'render_poses.npz',time=np.asarray([row[0] for row in video]),body_names=np.asarray(body_names),
                        body_position_m=np.asarray([row[1] for row in video]),body_quaternion_wxyz=np.asarray([row[2] for row in video]))
    E=arrays['kinetic_J']+arrays['potential_J'];power=sum(arrays[k] for k in ['actuator_power_W','friction_power_W','external_power_W','constraint_power_W'])
    energy_error=E-E[0]-cumulative_trapezoid(power,arrays['time'],initial=0)
    rotations=Rotation.from_quat(arrays['qpos'][:,[4,5,6,3]]).as_rotvec()
    metrics={'independent_dof':24-min(ranks),'constraint_ranks':sorted(ranks),'tree_dof':24,
             'maximum_loop_closure_m':float(np.linalg.norm(arrays['closure_m'].reshape(-1,6,3),axis=-1).max()),
             'independent_mass_matrix_max_abs':mass_error,'independent_bias_force_max_abs':bias_error,
             'independent_constraint_jacobian_max_abs':jac_error,'independent_body_position_max_error_m':pose_error,
             'maximum_equation_force_residual_N':float(abs(arrays['equation_residual'][:,force_indices]).max()),
             'maximum_equation_torque_residual_Nm':float(abs(arrays['equation_residual'][:,torque_indices]).max()),
             'maximum_work_energy_error_J':float(abs(energy_error).max()),
             'minimum_conservative_leg_clearance_m':float(minimum_leg_gap),'minimum_piston_end_clearance_m':float(minimum_piston_margin),
             'platform_translation_range_m':np.ptp(arrays['qpos'][:,:3],axis=0).tolist(),
             'platform_rotation_vector_range_rad':np.ptp(rotations,axis=0).tolist(),
             'maximum_prismatic_tracking_error_m':float(abs(arrays['qpos'][:,9::3]-arrays['command_prismatic_displacement_m']).max()),
             'maximum_actual_force_N':float(abs(arrays['actual_actuator_force_N']).max()),
             'maximum_friction_power_W':float(arrays['friction_power_W'].max()),'video_frames':len(video),
             'record_samples':len(records),'finite':all(np.isfinite(value).all() for value in arrays.values())}
    thresholds={'maximum_loop_closure_m':1e-4,'independent_mass_matrix_max_abs':1e-10,'independent_bias_force_max_abs':1e-8,
                'independent_constraint_jacobian_max_abs':1e-10,'maximum_equation_force_residual_N':1e-7,
                'maximum_equation_torque_residual_Nm':1e-7,'maximum_work_energy_error_J':.02,
                'maximum_prismatic_tracking_error_m':.012,'independent_body_position_max_error_m':1e-10}
    checks={name:metrics[name]<limit for name,limit in thresholds.items()}
    checks.update(six_degrees_of_freedom=ranks=={18},finite=metrics['finite'],complete_video_support=len(video)==120,
                  dissipative_friction=metrics['maximum_friction_power_W']<=1e-9,
                  leg_clearance=minimum_leg_gap>.002,piston_stroke_clearance=minimum_piston_margin>.008,
                  translation_excites_three_axes=min(metrics['platform_translation_range_m'])>.002,
                  rotation_excites_three_axes=min(metrics['platform_rotation_vector_range_rad'])>.005)
    report={'condition':condition,'passed':all(checks.values()),'checks':checks,'thresholds':thresholds,'metrics':metrics,
            'assembly_physical_digest':assembly['physical_digest'],'generator_source_digest':fingerprint,
            'mujoco_version':mujoco.__version__,'seconds':time.monotonic()-start,
            'scope':'Independent spatial Jacobians and Newton--Euler mass/bias equations, explicit simulator constraint forces/work, unprojected poses.',
            'command_interface':'Planned prismatic displacements, velocities, force bias and known external force; future measured forces remain evaluator-only.'}
    write_json(directory/'physics_validation.json',report)
    if not report['passed']:raise RuntimeError('Stewart physics validation failed: '+str({k:v for k,v in checks.items() if not v}))
    return report
