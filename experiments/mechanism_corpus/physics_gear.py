"""Independent angular dynamics and work balance for ideal rigid gear meshes."""
import copy
import hashlib
from pathlib import Path
import time
import mujoco
import numpy as np
from scipy.integrate import cumulative_trapezoid
from . import gear_pair,gear_geometry
from .catalog import OUTPUT,write_json


def source_digest():
    files=[Path(__file__),Path(gear_pair.__file__),Path(gear_geometry.__file__),Path(__file__).with_name('solids.py')]
    fingerprint=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    folder=OUTPUT/'source_versions'/('physics_'+fingerprint);folder.mkdir(parents=True,exist_ok=True)
    for p in files:
        if not (folder/p.name).exists():(folder/p.name).write_bytes(p.read_bytes())
    return fingerprint


def simulate(assembly,condition,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);fingerprint=source_digest()
    c=copy.deepcopy(assembly['config']);target={'low_speed':1.,'rated':3.,'limit_or_load':5.}[condition]
    load=c['nominal_output_load_torque_Nm']*(3. if condition=='limit_or_load' else 1.)
    bodies=assembly['bodies'];xml,_=gear_pair.mjcf(c,bodies);(directory/'model.xml').write_text(xml)
    write_json(directory/'physical_parameters.json',{'config':c,'parts':assembly['parts'],'bodies':bodies,
                                                    'operating_condition':condition,'output_load_torque_Nm':load})
    write_json(directory/'controller_inputs.json',{'known_output_load_torque_Nm':load,
             'load_command_policy':'Positive output-shaft torque with the same smooth start ramp; opposes the nominal output rotation.',
             'command_units':'rad/s input speed setpoint and Nm output load torque'})
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);act=c['actuator'];s=c['simulation']
    def ramp(t):return .5*(1-np.cos(np.pi*np.clip(t/act['ramp_s'],0,1)))
    def control(m,d):
        d.ctrl[0]=np.clip(act['speed_gain_Nm_s_rad']*(target*ramp(d.time)-d.qvel[0]),-act['torque_limit_Nm'],act['torque_limit_Nm'])
        d.qfrc_applied[:]=gear_pair.friction(d.qvel,c)+[0,load*ramp(d.time)]
    mujoco.set_mjcb_control(control);start=time.monotonic();steps=round(s['duration_s']/s['timestep_s'])
    record_stride=round(1/(s['record_hz']*s['timestep_s']));video_stride=round(1/(s['fps']*s['timestep_s']))
    J=np.array([-c['signed_speed_ratio'],1.]);gamma=np.array([1.,c['signed_speed_ratio']]);dense=np.zeros((2,2))
    records=[];video=[];mass_error=bias_error=jac_error=0.
    try:
        for step in range(steps+1):
            if step%record_stride==0:
                mujoco.mj_forward(model,data);q=data.qpos.copy();v=data.qvel.copy();a=data.qacc.copy()
                M,G,T,V=gear_pair.mass_bias_energy(q,v,c,bodies);friction=gear_pair.friction(v,c)
                Qload=np.array([0.,load*ramp(data.time)]);Qmotor=np.array([data.ctrl[0],0.])
                residual=M@a+G-Qmotor-friction-Qload
                mujoco.mj_fullM(model,dense,data.qM)
                mass_error=max(mass_error,float(abs(M-dense).max()));bias_error=max(bias_error,float(abs(G-data.qfrc_bias).max()))
                jac_error=max(jac_error,float(abs(J-data.efc_J.reshape(-1,2)[0]).max()))
                records.append((data.time,q,v,a,float(data.ctrl[0]),target*ramp(data.time),load*ramp(data.time),T,V,
                                float(J@q),float(J@v),float(J@a),float(gamma@residual),
                                (residual-data.qfrc_constraint).copy(),data.qfrc_constraint.copy(),
                                float(Qmotor@v),float(friction@v),float(Qload@v),float(data.qfrc_constraint@v)))
            if step%video_stride==0 and step<steps:
                mujoco.mj_forward(model,data);video.append((data.time,data.xpos.copy(),data.xquat.copy()))
            if step<steps:mujoco.mj_step(model,data)
    finally:mujoco.set_mjcb_control(None)
    names=['time','qpos','qvel','qacc','actual_motor_torque_Nm','speed_command_rad_s','output_load_command_Nm',
           'kinetic_J','potential_J','mesh_closure_rad','mesh_velocity_residual_rad_s','mesh_acceleration_residual_rad_s2',
           'motor_equivalent_residual_Nm','full_equation_residual_Nm','mesh_generalized_torque_Nm',
           'motor_power_W','friction_power_W','load_power_W','constraint_power_W']
    arrays={name:np.asarray([row[i] for row in records]) for i,name in enumerate(names)}
    np.savez_compressed(directory/'physics.npz',**arrays)
    np.savez_compressed(directory/'commands.npz',time=arrays['time'],speed_command_rad_s=arrays['speed_command_rad_s'],
                        output_load_command_Nm=arrays['output_load_command_Nm'])
    body_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or 'ground' for i in range(model.nbody)]
    np.savez_compressed(directory/'render_poses.npz',time=np.asarray([r[0] for r in video]),body_names=np.asarray(body_names),
                        body_position_m=np.asarray([r[1] for r in video]),body_quaternion_wxyz=np.asarray([r[2] for r in video]))
    E=arrays['kinetic_J']+arrays['potential_J'];power=sum(arrays[name] for name in ['motor_power_W','friction_power_W','load_power_W','constraint_power_W'])
    error=E-E[0]-cumulative_trapezoid(power,arrays['time'],initial=0)
    after=arrays['time']>=.75
    metrics={'independent_dof':1,'tree_dof':2,'constraint_rank':1,'maximum_mesh_closure_rad':float(abs(arrays['mesh_closure_rad']).max()),
             'maximum_motor_equivalent_residual_Nm':float(abs(arrays['motor_equivalent_residual_Nm']).max()),
             'maximum_full_equation_residual_Nm':float(abs(arrays['full_equation_residual_Nm']).max()),
             'independent_mass_matrix_max_abs':mass_error,'independent_bias_force_max_abs':bias_error,
             'independent_constraint_jacobian_max_abs':jac_error,'maximum_work_energy_error_J':float(abs(error).max()),
             'achieved_mean_input_speed_rad_s':float(arrays['qvel'][after,0].mean()),
             'maximum_speed_ratio_residual_rad_s':float(abs(arrays['mesh_velocity_residual_rad_s']).max()),
             'maximum_friction_power_W':float(arrays['friction_power_W'].max()),'maximum_output_load_power_W':float(arrays['load_power_W'].max()),
             'video_frames':len(video),'record_samples':len(records),'finite':all(np.isfinite(value).all() for value in arrays.values())}
    thresholds={'maximum_mesh_closure_rad':1e-5,'maximum_motor_equivalent_residual_Nm':1e-8,'maximum_full_equation_residual_Nm':1e-8,
                'independent_mass_matrix_max_abs':1e-10,'independent_bias_force_max_abs':1e-8,
                'independent_constraint_jacobian_max_abs':1e-10,'maximum_work_energy_error_J':.005}
    checks={name:metrics[name]<limit for name,limit in thresholds.items()}
    checks.update(finite=metrics['finite'],attained_operating_speed=.65*target<metrics['achieved_mean_input_speed_rad_s']<1.35*target,
                  complete_video_support=len(video)==120,dissipative_friction=metrics['maximum_friction_power_W']<=1e-9,
                  resisting_output_load=metrics['maximum_output_load_power_W']<=1e-9,
                  constraint_rank=int(np.linalg.matrix_rank(data.efc_J.reshape(-1,2)))==1)
    report={'passed':all(checks.values()),'condition':condition,'checks':checks,'thresholds':thresholds,'metrics':metrics,
            'assembly_physical_digest':assembly['physical_digest'],'generator_source_digest':fingerprint,'mujoco_version':mujoco.__version__,
            'seconds':time.monotonic()-start,'scope':'Ideal conjugate rigid gear constraint; independently derived shaft inertias, gravity, mesh torques, friction and work-energy balance.',
            'command_interface':'Input speed setpoint and output load-torque program are public. Realized feedback motor torque is evaluator-only.'}
    write_json(directory/'physics_validation.json',report)
    if not report['passed']:raise RuntimeError('Gear dynamics validation failed: '+str({k:v for k,v in checks.items() if not v}))
    return report
