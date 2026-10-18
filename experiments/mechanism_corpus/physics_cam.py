"""Contact-driven cam trajectories with independently assembled contact wrenches."""
import copy
import hashlib
from pathlib import Path
import time
import mujoco
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from . import cam_follower,gear_pair,gear_geometry
from .catalog import OUTPUT,write_json


def source_digest():
    files=[Path(__file__),Path(cam_follower.__file__),Path(gear_pair.__file__),Path(gear_geometry.__file__),Path(__file__).with_name('solids.py')]
    fingerprint=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest();folder=OUTPUT/'source_versions'/('physics_'+fingerprint)
    folder.mkdir(parents=True,exist_ok=True)
    for path in files:
        if not (folder/path.name).exists():(folder/path.name).write_bytes(path.read_bytes())
    return fingerprint


def contact_wrenches(model,data,c):
    Q=np.zeros(2);Qnormal=np.zeros(2);contacts=[]
    for i in range(data.ncon):
        contact=data.contact[i]
        if contact.efc_address<0:continue
        local=np.zeros(6);mujoco.mj_contactForce(model,data,i,local)
        frame=np.asarray(contact.frame).reshape(3,3);force=frame.T@local[:3];moment=frame.T@local[3:]
        normal=frame[0]*local[0];point=np.asarray(contact.pos)
        for geom,sign in [(contact.geom1,-1.),(contact.geom2,1.)]:
            body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[geom]))
            if body=='cam':
                lever=point-np.array([0,0,c['pivot_height_m']])
                Q[0]-=sign*(np.cross(lever,force)[1]+moment[1]);Qnormal[0]-=sign*np.cross(lever,normal)[1]
            elif body=='follower':Q[1]+=sign*force[2];Qnormal[1]+=sign*normal[2]
        contacts.append((point.copy(),force.copy(),float(local[0]),float(contact.dist),int(contact.geom1),int(contact.geom2)))
    return Q,Qnormal,contacts


def sliding_forces(model,data,c,contacts):
    forces=[];generalized=np.zeros(2)
    for point,normal_vector,normal_magnitude,distance,g1,g2 in contacts:
        if normal_magnitude<=0:
            forces.append(np.zeros(3));continue
        normal=normal_vector/normal_magnitude;velocities=[]
        for geom in [g1,g2]:
            body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[geom]))
            if body=='cam':
                r=point-np.array([0,0,c['pivot_height_m']]);velocities.append(data.qvel[0]*np.array([-r[2],0,r[0]]))
            elif body=='follower':velocities.append(np.array([0.,0.,data.qvel[1]]))
            else:velocities.append(np.zeros(3))
        relative=velocities[1]-velocities[0];tangent=relative-normal*(normal@relative);speed=np.linalg.norm(tangent)
        force=(-c['contact_friction_coefficient']*normal_magnitude*np.tanh(speed/c['sliding_regularization_speed_m_s'])*tangent/speed
               if speed>1e-14 else np.zeros(3))
        forces.append(force)
        for geom,sign in [(g1,-1.),(g2,1.)]:
            body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[geom]))
            if body=='cam':generalized[0]-=sign*np.cross(point-np.array([0,0,c['pivot_height_m']]),force)[1]
            elif body=='follower':generalized[1]+=sign*force[2]
    return forces,generalized


def apply_sliding_wrenches(model,data,contacts,forces):
    data.xfrc_applied[:]=0
    for contact,force in zip(contacts,forces):
        point,g1,g2=contact[0],contact[4],contact[5]
        for geom,sign in [(g1,-1.),(g2,1.)]:
            body=int(model.geom_bodyid[geom]);F=sign*force
            data.xfrc_applied[body,:3]+=F
            data.xfrc_applied[body,3:]+=np.cross(point-data.xipos[body],F)


def simulate(assembly,condition,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);fingerprint=source_digest()
    c=copy.deepcopy(assembly['config']);target={'low_speed':1.,'rated':3.,'limit_or_load':5.}[condition]
    load=c['preload_force_N']*(2.5 if condition=='limit_or_load' else 1.)
    bodies=assembly['bodies'];xml,_=cam_follower.mjcf(c,bodies);(directory/'model.xml').write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);act=c['actuator'];s=c['simulation']
    def ramp(t):return .5*(1-np.cos(np.pi*np.clip(t/act['ramp_s'],0,1)))
    if s['integrator']!='implicitfast':raise ValueError('Explicit contact-friction coupling uses the implicitfast single-step integrator')
    mujoco.set_mjcb_control(None)
    maximum_friction_error=0.;maximum_coupling_iterations=0
    def forward():
        nonlocal maximum_friction_error,maximum_coupling_iterations
        data.ctrl[0]=np.clip(act['speed_gain_Nm_s_rad']*(target*ramp(data.time)-data.qvel[0]),-act['torque_limit_Nm'],act['torque_limit_Nm'])
        data.ctrl[1]=-load;data.qfrc_applied[:]=cam_follower.friction(data.qvel,c);data.xfrc_applied[:]=0
        mujoco.mj_forward(model,data)
        for iteration in range(12):
            _,_,contacts=contact_wrenches(model,data,c);forces,Qt=sliding_forces(model,data,c,contacts)
            apply_sliding_wrenches(model,data,contacts,forces);mujoco.mj_forward(model,data)
            Qn,_,updated=contact_wrenches(model,data,c);expected,_=sliding_forces(model,data,c,updated)
            error=max((float(np.linalg.norm(f-g)) for f,g in zip(forces,expected)),default=0.) if len(forces)==len(expected) else np.inf
            if error<1e-10:
                maximum_friction_error=max(maximum_friction_error,error);maximum_coupling_iterations=max(maximum_coupling_iterations,iteration+1)
                return Qn,Qt,updated,forces
        raise RuntimeError('Normal/sliding friction coupling did not converge')
    records=[];video=[];all_contacts=[];start=time.monotonic();dense=np.zeros((2,2));mass_error=bias_error=wrench_error=0.
    try:
        def static_balance(displacement):
            data.qpos[1]=displacement;forward();return float(data.qacc[1])
        data.qpos[1]=brentq(static_balance,-.0001,.0001,xtol=1e-14)
        forward();initial=data.qpos.copy();initial_acceleration=float(abs(data.qacc).max())
        write_json(directory/'physical_parameters.json',{'config':c,'parts':assembly['parts'],'bodies':bodies,
                  'operating_condition':condition,'actual_preload_force_N':load,'initial_equilibrium_qpos':initial.tolist(),
                  'initialization':'Static equilibrium solved before recording; recorded poses are not projected or repaired.'})
        write_json(directory/'controller_inputs.json',{'commanded_preload_force_N':-load,
                  'command_units':'Input speed in rad/s; downward follower actuator force in N.'})
        steps=round(s['duration_s']/s['timestep_s']);record_stride=round(1/(s['record_hz']*s['timestep_s']));video_stride=round(1/(s['fps']*s['timestep_s']))
        for step in range(steps+1):
            Qnormal,Qsliding,contacts,tangent_forces=forward();Qcontact=Qnormal+Qsliding
            if step%record_stride==0:
                q=data.qpos.copy();v=data.qvel.copy();a=data.qacc.copy()
                M,G,T,V=cam_follower.mass_bias_energy(q,v,c,bodies);Qfriction=cam_follower.friction(v,c)
                mujoco.mj_fullM(model,dense,data.qM)
                mass_error=max(mass_error,float(abs(M-dense).max()));bias_error=max(bias_error,float(abs(G-data.qfrc_bias).max()))
                wrench_error=max(wrench_error,float(abs(Qnormal-data.qfrc_constraint).max()))
                residual=M@a+G-data.ctrl-Qfriction-Qcontact;gap,J=cam_follower.gap_and_jacobian(q,c)
                normal_force=sum(contact[2] for contact in contacts)
                records.append((data.time,q,v,a,data.ctrl.copy(),target*ramp(data.time),T,V,gap,float(J@v),residual,
                                Qcontact,Qnormal,normal_force,len(contacts),
                                float(data.ctrl[0]*v[0]),float(data.ctrl[1]*v[1]),float(Qfriction@v),
                                float(Qnormal@v),float((Qcontact-Qnormal)@v)))
                all_contacts.append([(p,normal+sliding,N,dist,g1,g2,normal,sliding)
                                     for (p,normal,N,dist,g1,g2),sliding in zip(contacts,tangent_forces)])
            if step%video_stride==0 and step<steps:
                video.append((data.time,data.xpos.copy(),data.xquat.copy()))
            if step<steps:mujoco.mj_step(model,data)
    finally:mujoco.set_mjcb_control(None)
    names=['time','qpos','qvel','qacc','actual_actuator_efforts','speed_command_rad_s','kinetic_J','potential_J',
           'normal_gap_m','normal_gap_velocity_m_s','equation_residual','contact_generalized_force',
           'normal_generalized_force','summed_normal_contact_force_N','active_contact_count',
           'motor_power_W','preload_power_W','bearing_and_guide_friction_power_W','normal_contact_power_W','contact_sliding_power_W']
    arrays={name:np.asarray([r[i] for r in records]) for i,name in enumerate(names)}
    maximum_contacts=max(len(row) for row in all_contacts)
    contact_positions=np.zeros((len(records),maximum_contacts,3));contact_forces=np.zeros_like(contact_positions)
    normal_world=np.zeros_like(contact_positions);sliding_world=np.zeros_like(contact_positions)
    contact_normals=np.zeros((len(records),maximum_contacts));contact_distances=np.zeros_like(contact_normals)
    contact_geoms=np.full((len(records),maximum_contacts,2),-1,dtype=np.int32)
    for i,contacts in enumerate(all_contacts):
        for j,(position,force,normal,distance,g1,g2,normal_vector,sliding_vector) in enumerate(contacts):
            contact_positions[i,j]=position;contact_forces[i,j]=force;contact_normals[i,j]=normal;contact_distances[i,j]=distance;contact_geoms[i,j]=[g1,g2]
            normal_world[i,j]=normal_vector;sliding_world[i,j]=sliding_vector
    arrays.update(contact_position_world_m=contact_positions,force_on_geom2_world_N=contact_forces,
                  normal_force_on_geom2_world_N=normal_world,sliding_force_on_geom2_world_N=sliding_world,
                  normal_contact_force_N=contact_normals,contact_distance_m=contact_distances,contact_geom_ids=contact_geoms)
    np.savez_compressed(directory/'physics.npz',**arrays)
    np.savez_compressed(directory/'commands.npz',time=arrays['time'],speed_command_rad_s=arrays['speed_command_rad_s'],
                        follower_force_command_N=np.full(len(records),-load))
    body_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or 'ground' for i in range(model.nbody)]
    np.savez_compressed(directory/'render_poses.npz',time=np.asarray([r[0] for r in video]),body_names=np.asarray(body_names),
                        body_position_m=np.asarray([r[1] for r in video]),body_quaternion_wxyz=np.asarray([r[2] for r in video]))
    E=arrays['kinetic_J']+arrays['potential_J'];power=sum(arrays[name] for name in ['motor_power_W','preload_power_W',
                                     'bearing_and_guide_friction_power_W','normal_contact_power_W','contact_sliding_power_W'])
    error=E-E[0]-cumulative_trapezoid(power,arrays['time'],initial=0);after=arrays['time']>=.75
    metrics={'tree_dof':2,'active_normal_constraint_rank':1,'independent_dof_in_contact':1,
             'initial_equilibrium_max_acceleration':initial_acceleration,'minimum_normal_contact_force_N':float(arrays['summed_normal_contact_force_N'].min()),
             'contact_fraction':float(np.mean(arrays['active_contact_count']>0)),'maximum_contact_count':maximum_contacts,
             'maximum_penetration_m':float(np.maximum(-arrays['normal_gap_m'],0).max()),
             'maximum_open_gap_m':float(arrays['normal_gap_m'].max()),
             'maximum_equation_torque_residual_Nm':float(abs(arrays['equation_residual'][:,0]).max()),
             'maximum_equation_force_residual_N':float(abs(arrays['equation_residual'][:,1]).max()),
             'independent_mass_matrix_max_abs':mass_error,'independent_bias_force_max_abs':bias_error,
             'contact_wrench_reconstruction_max_abs':wrench_error,'maximum_work_energy_error_J':float(abs(error).max()),
             'maximum_regularized_friction_force_error_N':maximum_friction_error,'maximum_friction_coupling_iterations':maximum_coupling_iterations,
             'maximum_contact_sliding_power_W':float(arrays['contact_sliding_power_W'].max()),
             'maximum_bearing_and_guide_friction_power_W':float(arrays['bearing_and_guide_friction_power_W'].max()),
             'achieved_mean_cam_speed_rad_s':float(arrays['qvel'][after,0].mean()),
             'follower_stroke_m':float(np.ptp(arrays['qpos'][:,1])),'video_frames':len(video),'record_samples':len(records),
             'finite':all(np.isfinite(value).all() for value in arrays.values())}
    thresholds={'initial_equilibrium_max_acceleration':1e-5,'maximum_penetration_m':2e-6,'maximum_open_gap_m':2e-6,
                'maximum_equation_torque_residual_Nm':1e-7,'maximum_equation_force_residual_N':1e-7,
                'independent_mass_matrix_max_abs':1e-10,'independent_bias_force_max_abs':1e-8,
                'contact_wrench_reconstruction_max_abs':1e-7,'maximum_regularized_friction_force_error_N':1e-9,'maximum_work_energy_error_J':.01}
    checks={key:metrics[key]<limit for key,limit in thresholds.items()}
    checks.update(maintained_contact=metrics['contact_fraction']==1. and metrics['minimum_normal_contact_force_N']>0,
                  dissipative_sliding=metrics['maximum_contact_sliding_power_W']<=1e-8,
                  dissipative_bearings=metrics['maximum_bearing_and_guide_friction_power_W']<=1e-9,
                  attained_speed=.65*target<metrics['achieved_mean_cam_speed_rad_s']<1.35*target,
                  finite=metrics['finite'],complete_video_support=len(video)==120)
    report={'passed':all(checks.values()),'condition':condition,'checks':checks,'thresholds':thresholds,'metrics':metrics,
            'assembly_physical_digest':assembly['physical_digest'],'generator_source_digest':fingerprint,'mujoco_version':mujoco.__version__,
            'seconds':time.monotonic()-start,'coordinate_units':['rad','m'],'generalized_effort_units':['Nm','N'],
            'scope':'MuJoCo frictionless normal contact plus explicit regularized Coulomb world forces. Force coupling converges before each implicitfast step. Independent solid moments, full contact wrenches, friction work and energy balance; no prescribed follower motion.'}
    write_json(directory/'physics_validation.json',report)
    if not report['passed']:raise RuntimeError('Cam/contact validation failed: '+str({k:v for k,v in checks.items() if not v}))
    return report
