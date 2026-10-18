"""Torque-driven Delta simulation with independent spatial tree dynamics."""
import copy
import hashlib
from pathlib import Path
import time
import mujoco
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation
from . import delta
from .spatial_tree import SpatialTree
from .catalog import OUTPUT,write_json


def source_digest():
    files=[Path(__file__),Path(delta.__file__)]+[Path(__file__).with_name(name) for name in
           ['spatial_tree.py','joint_hardware.py','spatial_solids.py','gear_pair.py','gear_geometry.py','solids.py']]
    fingerprint=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest();folder=OUTPUT/'source_versions'/('physics_'+fingerprint);folder.mkdir(parents=True,exist_ok=True)
    for path in files:
        if not (folder/path.name).exists():(folder/path.name).write_bytes(path.read_bytes())
    return fingerprint


def commands(times,c,condition):
    frequency={'low_speed':.18,'rated':.36,'limit_or_load':.58}[condition]
    phase=2*np.pi*frequency*times;envelope=.5*(1-np.cos(np.pi*np.clip(times/c['actuator']['ramp_s'],0,1)))
    positions=np.column_stack([.024*np.sin(phase),.020*np.sin(.9*phase+.5),.022*np.sin(1.1*phase+.8)])*envelope[:,None]
    positions+=np.array([0,0,c['platform_height_m']]);angles=[];L=c['upper_beam_length_m'];off=c['upper_joint_offset_m'];forearm=c['forearm_length_m']
    for phi in c['leg_azimuths_rad']:
        local=positions@delta.rz(phi)-np.array([c['base_radius_m']-c['platform_anchor_radius_m'],0,c['base_height_m']])
        A=local[:,0]*L+local[:,2]*off;B=local[:,0]*off-local[:,2]*L
        ratio=(np.sum(local*local,axis=1)+L*L+off*off-forearm*forearm)/(2*np.hypot(A,B))
        if abs(ratio).max()>=1:raise ValueError('Delta motion reference exits workspace')
        center=np.arctan2(B,A);spread=np.arccos(ratio)
        options=np.column_stack([center-spread,center+spread])-c['upper_reference_angle_rad']
        options=(options+np.pi)%(2*np.pi)-np.pi
        angles.append(options[np.arange(len(times)),np.argmin(abs(options),axis=1)])
    angles=np.column_stack(angles)
    if abs(angles).max()>.6:raise ValueError('Delta shoulder command exits reviewed workspace')
    return angles,positions,{'frequency_Hz':frequency,'cartesian_amplitudes_m':[.024,.020,.022]}


def simulate(assembly,condition,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);fingerprint=source_digest();c=copy.deepcopy(assembly['config'])
    if condition=='limit_or_load':c['payload_mass_kg']*=2.
    parts=delta.inventory(c);bodies=delta.solid_properties(parts);xml,_=delta.mjcf(c,bodies);(directory/'model.xml').write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);tree=SpatialTree.from_mujoco(model).use_body_calibration(bodies)
    drives=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'drive_%d'%i) for i in range(3)]
    drive_q=np.array([model.jnt_qposadr[j] for j in drives]);drive_v=np.array([model.jnt_dofadr[j] for j in drives])
    passive_v=np.array([model.jnt_dofadr[j] for j in range(model.njnt) if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE and j not in drives])
    B=np.zeros((model.nv,3));B[drive_v,np.arange(3)]=1
    states=tree.forward(data.qpos,data.qvel);initial_residual,J0=delta.closure(states,tree,c);rank=int(np.linalg.matrix_rank(J0))
    if rank!=18 or model.nv-rank!=3 or abs(initial_residual).max()>1e-10:raise ValueError('Delta initial mobility/closure differs from specification')
    _,_,Vh=np.linalg.svd(J0,full_matrices=True);null=Vh[rank:].T;_,bias,_,_=tree.dynamics(data.qpos,data.qvel,states)
    feedforward=np.linalg.solve(null.T@B,null.T@bias);act=c['actuator']
    if abs(feedforward).max()>.8*act['torque_limit_Nm']:raise ValueError('Delta static torque exceeds actuator margin')
    s=c['simulation'];dt=s['timestep_s'];steps=round(s['duration_s']/dt);times=np.arange(2*steps+1)*dt/2
    command,reference,description=commands(times,c,condition);velocity=np.gradient(command,dt/2,axis=0,edge_order=2)
    def friction(v):
        out=np.zeros(model.nv)
        out[drive_v]=-c['drive_viscosity_Nm_s_rad']*v[drive_v]-c['drive_coulomb_Nm']*np.tanh(v[drive_v]/.03)
        out[passive_v]=-c['passive_viscosity_Nm_s_rad']*v[passive_v]-c['passive_coulomb_Nm']*np.tanh(v[passive_v]/.03)
        return out
    def index(t):return int(np.clip(round(t/(dt/2)),0,len(times)-1))
    def control(m,d):
        k=index(d.time);d.ctrl[:]=np.clip(feedforward+act['position_gain_Nm_rad']*(command[k]-d.qpos[drive_q])+
                                        act['velocity_gain_Nm_s_rad']*(velocity[k]-d.qvel[drive_v]),-act['torque_limit_Nm'],act['torque_limit_Nm'])
        d.qfrc_applied[:]=friction(d.qvel)
    write_json(directory/'physical_parameters.json',{'config':c,'parts':parts,'bodies':bodies,'tree_calibration':tree.specification,
                                                    'condition':condition,'reference_motion':description,'feedforward_torque_Nm':feedforward.tolist()})
    write_json(directory/'controller_inputs.json',{'controller_calibration':act,'feedforward_torque_Nm':feedforward.tolist(),
                                                  'command_units':'Three shoulder joint offsets in rad and angular velocities in rad/s'})
    mujoco.set_mjcb_control(control);records=[];videos=[];ranks=set();dense=np.zeros((model.nv,model.nv));mass_error=bias_error=jac_error=pose_error=0.
    record_stride=round(1/(s['record_hz']*dt));video_stride=round(1/(s['fps']*dt));start=time.monotonic()
    platform_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'platform')
    try:
        for step in range(steps+1):
            if step%record_stride==0:
                mujoco.mj_forward(model,data);q=data.qpos.copy();v=data.qvel.copy();a=data.qacc.copy()
                states=tree.forward(q,v);M,bias,T,V=tree.dynamics(q,v,states);residual,J=delta.closure(states,tree,c);ranks.add(int(np.linalg.matrix_rank(J)))
                mujoco.mj_fullM(model,dense,data.qM);mass_error=max(mass_error,float(abs(M-dense).max()))
                bias_error=max(bias_error,float(abs(bias-data.qfrc_bias).max()));jac_error=max(jac_error,float(abs(J-data.efc_J.reshape(-1,model.nv)).max()))
                pose_error=max(pose_error,max(float(np.linalg.norm(states[i]['position']-data.xpos[i])) for i in range(1,model.nbody)))
                Qfr=friction(v);Qdrive=B@data.ctrl;equation=M@a+bias-Qfr-Qdrive-data.qfrc_constraint;k=index(data.time)
                records.append((data.time,q,v,a,data.ctrl.copy(),command[k].copy(),velocity[k].copy(),reference[k].copy(),
                                T,V,residual,J@v,equation,data.qfrc_constraint.copy(),
                                float(Qdrive@v),float(Qfr@v),float(data.qfrc_constraint@v),
                                data.xpos[platform_id].copy(),data.xquat[platform_id].copy()))
            if step%video_stride==0 and step<steps:
                mujoco.mj_forward(model,data);videos.append((data.time,data.xpos.copy(),data.xquat.copy()))
            if step<steps:mujoco.mj_step(model,data)
    finally:mujoco.set_mjcb_control(None)
    names=['time','qpos','qvel','qacc','actual_shoulder_torque_Nm','commanded_shoulder_angle_rad','commanded_shoulder_velocity_rad_s',
           'reference_platform_position_m','kinetic_J','potential_J','closure_m','closure_velocity_m_s','equation_residual',
           'constraint_generalized_force','actuator_power_W','friction_power_W','constraint_power_W','platform_position_m','platform_quaternion_wxyz']
    arrays={name:np.asarray([r[i] for r in records]) for i,name in enumerate(names)}
    np.savez_compressed(directory/'physics.npz',**arrays)
    np.savez_compressed(directory/'commands.npz',time=arrays['time'],shoulder_angle_command_rad=arrays['commanded_shoulder_angle_rad'],
                        shoulder_velocity_command_rad_s=arrays['commanded_shoulder_velocity_rad_s'])
    body_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or 'ground' for i in range(model.nbody)]
    np.savez_compressed(directory/'render_poses.npz',time=np.asarray([r[0] for r in videos]),body_names=np.asarray(body_names),
                        body_position_m=np.asarray([r[1] for r in videos]),body_quaternion_wxyz=np.asarray([r[2] for r in videos]))
    E=arrays['kinetic_J']+arrays['potential_J'];power=arrays['actuator_power_W']+arrays['friction_power_W']+arrays['constraint_power_W']
    balance=E-E[0]-cumulative_trapezoid(power,arrays['time'],initial=0)
    rotation=Rotation.from_quat(arrays['platform_quaternion_wxyz'][:,[1,2,3,0]]).as_rotvec()
    metrics={'tree_dof':model.nv,'constraint_ranks':sorted(ranks),'independent_dof':model.nv-min(ranks),
             'maximum_loop_closure_m':float(np.linalg.norm(arrays['closure_m'].reshape(-1,6,3),axis=-1).max()),
             'independent_mass_matrix_max_abs':mass_error,'independent_bias_force_max_abs':bias_error,
             'independent_constraint_jacobian_max_abs':jac_error,'independent_body_position_max_error_m':pose_error,
             'maximum_force_equation_residual_N':float(abs(arrays['equation_residual'][:,:3]).max()),
             'maximum_torque_equation_residual_Nm':float(abs(arrays['equation_residual'][:,3:]).max()),
             'maximum_work_energy_error_J':float(abs(balance).max()),'maximum_platform_rotation_rad':float(np.linalg.norm(rotation,axis=1).max()),
             'platform_translation_range_m':np.ptp(arrays['platform_position_m'],axis=0).tolist(),
             'maximum_cartesian_tracking_error_m':float(np.linalg.norm(arrays['platform_position_m']-arrays['reference_platform_position_m'],axis=1).max()),
             'maximum_shoulder_offset_rad':float(abs(arrays['qpos'][:,drive_q]).max()),
             'maximum_friction_power_W':float(arrays['friction_power_W'].max()),'maximum_drive_torque_Nm':float(abs(arrays['actual_shoulder_torque_Nm']).max()),
             'video_frames':len(videos),'record_samples':len(records),'finite':all(np.isfinite(x).all() for x in arrays.values())}
    thresholds={'maximum_loop_closure_m':1e-4,'independent_mass_matrix_max_abs':1e-10,'independent_bias_force_max_abs':1e-8,
                'independent_constraint_jacobian_max_abs':1e-10,'independent_body_position_max_error_m':1e-10,
                'maximum_force_equation_residual_N':1e-7,'maximum_torque_equation_residual_Nm':1e-7,
                'maximum_work_energy_error_J':.02,'maximum_platform_rotation_rad':1e-3,'maximum_cartesian_tracking_error_m':.015,
                'maximum_shoulder_offset_rad':.6}
    checks={name:metrics[name]<limit for name,limit in thresholds.items()}
    checks.update(three_independent_dof=ranks=={18},spatial_translation_excitation=min(metrics['platform_translation_range_m'])>.008,
                  dissipative_friction=metrics['maximum_friction_power_W']<=1e-9,finite=metrics['finite'],complete_video_support=len(videos)==120)
    report={'passed':all(checks.values()),'condition':condition,'checks':checks,'thresholds':thresholds,'metrics':metrics,
            'assembly_physical_digest':assembly['physical_digest'],'generator_source_digest':fingerprint,'mujoco_version':mujoco.__version__,
            'seconds':time.monotonic()-start,'scope':'Independent spatial Newton--Euler dynamics using per-part aggregate inertias, six closed forearm constraints, raw simulator poses and explicitly recorded constraint work.'}
    write_json(directory/'physics_validation.json',report)
    if not report['passed']:raise RuntimeError('Delta dynamics failed: '+str({k:v for k,v in checks.items() if not v}))
    return report
