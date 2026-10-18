"""Torque-driven MuJoCo four-bar and independent equation-based audit."""

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from geometry import link_mesh

from mechanics import (TREE_TO_ABSOLUTE, properties, payload_inertia, inverse_kinematics, closure,
                       closure_jacobian, closure_curvature, mass_bias_energy,
                       friction, coupler_point, drive_torque, reduced_acceleration)

ROOT = Path(__file__).resolve().parents[2]


def vector(values):
    return ' '.join(f'{float(value):.15g}' for value in values)


def rotation(angle):
    return vector([np.cos(angle/2), 0, -np.sin(angle/2), 0])


def make_model(config):
    q0 = inverse_kinematics(config['initial_crank_angle_rad'], config)
    lengths = config['lengths_m']
    masses, inertias = properties(config)
    root = ET.Element('mujoco', model=config['name'])
    ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='false')
    simulation = config['simulation']
    option = ET.SubElement(root, 'option', timestep=str(simulation['timestep_s']), gravity=vector([0,0,-config['gravity_m_s2']]),
                           integrator='implicitfast', solver='Newton', iterations='100', tolerance='1e-12', jacobian='dense')
    ET.SubElement(option, 'flag', energy='enable')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(simulation['width']), offheight=str(simulation['height']))
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', contype='0', conaffinity='0')
    ET.SubElement(default, 'joint', type='hinge', axis='0 -1 0', damping='0', frictionloss='0', limited='false')
    assets = ET.SubElement(root, 'asset')
    for i, length in enumerate(lengths):
        vertices, faces = link_mesh(length, config['bar_height_m'], config['bar_depth_m'], config['joint_bore_radius_m'])
        ET.SubElement(assets, 'mesh', name='link_mesh_'+str(i),
                      vertex=vector(np.ravel(vertices)), face=' '.join(str(j) for face in faces for j in face), smoothnormal='false')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='2 2 0.01', rgba='0.88 0.9 0.93 1')
    ET.SubElement(world, 'geom', name='ground_link', type='box', pos=vector([config['ground_length_m']/2,0.055,config['pivot_height_m']]),
                  size=vector([config['ground_length_m']/2+0.045,0.02,0.016]), rgba='0.22 0.25 0.3 1')
    for x in [0,config['ground_length_m']]:
        ET.SubElement(world,'geom',type='box',pos=vector([x,0.065,config['pivot_height_m']/2]),size=vector([0.03,0.025,config['pivot_height_m']/2]),rgba='0.25 0.28 0.33 1')
    ET.SubElement(world, 'light', pos='0.1 -0.5 1.4', dir='0 0 -1', diffuse='0.9 0.9 0.9')
    camera = config['camera']; position=np.array(camera['position_m']); target=np.array(camera['target_m'])
    z=(position-target)/np.linalg.norm(position-target)
    x=np.cross([0,0,1],z); x=x/np.linalg.norm(x); y=np.cross(z,x)
    ET.SubElement(world,'camera',name='main',pos=vector(position),xyaxes=vector(np.r_[x,y]),fovy=str(camera['vertical_fov_deg']))
    colors = [[0.04,0.55,0.82,1],[0.98,0.48,0.08,1],[0.15,0.68,0.43,1]]
    a = ET.SubElement(world,'body',name='crank',pos=vector([0,0,config['pivot_height_m']]),quat=rotation(q0[0]))
    ET.SubElement(a,'joint',name='A',armature=str(config['motor_armature_kg_m2']))
    b = ET.SubElement(a,'body',name='coupler',pos=vector([lengths[0],0,0]),quat=rotation(q0[1]-q0[0]))
    ET.SubElement(b,'joint',name='B')
    d = ET.SubElement(world,'body',name='rocker',pos=vector([config['ground_length_m'],0,config['pivot_height_m']]),quat=rotation(q0[2]))
    ET.SubElement(d,'joint',name='D')
    for i,body in enumerate([a,b,d]):
        com=[lengths[i]/2,config['visual_and_com_y_m'][i],0]
        ET.SubElement(body,'inertial',pos=vector(com),mass=str(masses[i]),diaginertia=vector(inertias[i]))
        ET.SubElement(body,'geom',name=['crank_bar','coupler_bar','rocker_bar'][i],type='mesh',
                      mesh='link_mesh_'+str(i),pos=vector([0,config['visual_and_com_y_m'][i],0]),rgba=vector(colors[i]))
    payload = ET.SubElement(b,'body',name='payload',pos=vector([config['payload_along_coupler_m'],config['payload_y_m'],config['payload_normal_offset_m']]))
    ET.SubElement(payload,'inertial',pos='0 0 0',mass=str(config['payload_mass_kg']),diaginertia=vector(payload_inertia(config)))
    ET.SubElement(payload,'geom',type='box',size=vector(np.asarray(config['payload_size_m'])/2),rgba='0.09 0.12 0.15 1')
    ET.SubElement(a,'site',name='A_site',pos='0 0 0',size='0.009',rgba='0.95 0.9 0.1 1')
    ET.SubElement(b,'site',name='B_site',pos='0 0 0',size='0.009',rgba='0.95 0.9 0.1 1')
    ET.SubElement(b,'site',name='C_coupler',pos=vector([lengths[1],0,0]),size='0.009',rgba='0.95 0.9 0.1 1')
    ET.SubElement(d,'site',name='C_rocker',pos=vector([lengths[2],0,0]),size='0.006',rgba='1 0.1 0.1 1')
    ET.SubElement(d,'site',name='D_site',pos='0 0 0',size='0.009',rgba='0.95 0.9 0.1 1')
    ET.SubElement(b,'site',name='force_point',pos=vector([config['force_application_along_coupler_m'],0,0]),size='0.004',rgba='1 0 0 1')
    equality=ET.SubElement(root,'equality')
    # anchor is local to body1. Reference geometry starts exactly closed.
    ET.SubElement(equality,'connect',name='C_loop',body1='coupler',body2='rocker',anchor=vector([lengths[1],0,0]),
                  solref=vector([simulation['equality_solref_s'],1]),
                  solimp=vector([simulation['equality_impedance'],simulation['equality_impedance'],0.001,0.5,2]))
    actuator=ET.SubElement(root,'actuator')
    ET.SubElement(actuator,'motor',name='crank_motor',joint='A',gear='1',ctrllimited='true',ctrlrange=vector([-config['drive']['torque_limit_Nm'],config['drive']['torque_limit_Nm']]))
    ET.indent(root, space='  ') if hasattr(ET,'indent') else None
    return ET.tostring(root,encoding='unicode'),q0


def simulate(config, output, render):
    output.mkdir(parents=True,exist_ok=True)
    xml,q0=make_model(config)
    (output/'model.xml').write_text(xml)
    masses,inertias=properties(config)
    metadata=copy.deepcopy(config)
    metadata['moving_bar_mass_kg']=masses.tolist()
    metadata['moving_bar_diagonal_inertia_kg_m2']=inertias.tolist()
    metadata['payload_diagonal_inertia_kg_m2']=payload_inertia(config).tolist()
    metadata['initial_absolute_angles_rad']=q0.tolist()
    metadata['physical_assumptions']=['uniform capsule plates with two through bores; analytic mass and inertia, pivot spacing excludes round ends','four ideal revolute joints with explicit smooth Coulomb plus viscous friction','connector sleeves, washers, fasteners and payload mounting plate are massless visual idealizations','fixed ground and fixed motor housing','no contact/collision forces','uniform cuboid rigid payload; analytic COM inertia from local X/Y/Z dimensions','finite-compliance MuJoCo loop closure, audited without projecting states','force/torque-driven motion; no per-frame kinematic overwrite','render-only edge finishing does not change nominal solid moments']
    (output/'physical_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    model=mujoco.MjModel.from_xml_string(xml); data=mujoco.MjData(model)
    assert model.nv==3 and model.nu==1
    s=config['simulation']; dt=s['timestep_s']; steps=round(s['duration_s']/dt)
    record_stride=round(1/(s['record_hz']*dt)); video_stride=round(1/(s['video_fps']*dt))
    assert abs(record_stride*dt*s['record_hz']-1)<1e-10
    assert abs(video_stride*dt*s['video_fps']-1)<1e-10
    renderer=mujoco.Renderer(model,s['height'],s['width']) if render else None
    writer=None
    if render:
        import cv2
        writer=cv2.VideoWriter(str(output/'mujoco_rgb.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),s['video_fps'],(s['width'],s['height']))
        if not writer.isOpened(): raise RuntimeError('Cannot open video writer')
    record=[]; frames=[]
    point_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'force_point')
    body_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'coupler')
    force=np.array([config['external_force_xz_N'][0],0,config['external_force_xz_N'][1]])
    mass_dense=np.zeros((3,3))
    max_mass_difference=0.; max_bias_difference=0.
    for step in range(steps+1):
        q=q0+TREE_TO_ABSOLUTE@data.qpos; v=TREE_TO_ABSOLUTE@data.qvel
        tau=drive_torque(data.time,v[0],config)
        qfriction,_,_=friction(v,config)
        data.ctrl[0]=tau
        data.qfrc_applied[:]=TREE_TO_ABSOLUTE.T@qfriction
        mujoco.mj_forward(model,data)
        mujoco.mj_applyFT(model,data,force,np.zeros(3),data.site_xpos[point_id],body_id,data.qfrc_applied)
        mujoco.mj_forward(model,data)
        if step%record_stride==0:
            acceleration=TREE_TO_ABSOLUTE@data.qacc
            mass,coriolis,gravity,kinetic,potential=mass_bias_energy(q,v,config)
            jacobian=closure_jacobian(q,config)
            _,point_jacobian=coupler_point(q,config,config['force_application_along_coupler_m'])
            external=point_jacobian.T@force[[0,2]]
            residual=mass@acceleration+coriolis+gravity-np.array([tau,0,0])-qfriction-external
            reaction=np.linalg.lstsq(jacobian.T,residual,rcond=None)[0]
            tangent_vector=np.cross(jacobian[0],jacobian[1]); tangent_vector/=np.linalg.norm(tangent_vector)
            dynamic_residual=float(tangent_vector@residual)
            mujoco.mj_fullM(model,mass_dense,data.qM)
            max_mass_difference=max(max_mass_difference,float(np.max(np.abs(mass_dense-TREE_TO_ABSOLUTE.T@mass@TREE_TO_ABSOLUTE))))
            max_bias_difference=max(max_bias_difference,float(np.max(np.abs(data.qfrc_bias-TREE_TO_ABSOLUTE.T@(coriolis+gravity)))))
            record.append((data.time,q.copy(),v.copy(),acceleration.copy(),data.qpos.copy(),tau,kinetic,potential,
                           closure(q,config),jacobian@v,jacobian@acceleration+closure_curvature(q,v,config),reaction,
                           dynamic_residual,tau*v[0],float(qfriction@v),float(external@v),float(reaction@(jacobian@v))))
        if step%video_stride==0:
            frames.append((data.time,q.copy(),data.qpos.copy()))
            if renderer:
                renderer.update_scene(data,camera='main'); image=renderer.render()
                import cv2
                writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                if step==0:cv2.imwrite(str(output/'mujoco_preview.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        if step<steps:mujoco.mj_step(model,data)
    if renderer:renderer.close();writer.release()
    fields=['time','q','qdot','qddot','q_mujoco','motor_torque','kinetic_energy','potential_energy','closure_position',
            'closure_velocity','closure_acceleration','constraint_reaction','dynamics_tangent_residual','motor_power',
            'friction_power','external_power','constraint_power']
    arrays={name:np.array([row[i] for row in record]) for i,name in enumerate(fields)}
    np.savez_compressed(output/'ground_truth.npz',**arrays)
    np.savez_compressed(output/'video_trajectory.npz',time=np.array([r[0] for r in frames]),q=np.array([r[1] for r in frames]),q_mujoco=np.array([r[2] for r in frames]))
    np.savez_compressed(output/'actions.npz',time=arrays['time'],motor_torque_Nm=arrays['motor_torque'],
                        external_force_xz_N=np.broadcast_to(force[[0,2]],(len(record),2)))
    with (output/'trajectory.csv').open('w',newline='') as handle:
        csvwriter=csv.writer(handle);csvwriter.writerow(['time_s','crank_rad','coupler_rad','rocker_rad','crank_velocity_rad_s','coupler_velocity_rad_s','rocker_velocity_rad_s','motor_torque_Nm','closure_error_m','tangent_dynamics_residual_Nm'])
        for i,t in enumerate(arrays['time']):
            csvwriter.writerow([t,*arrays['q'][i],*arrays['qdot'][i],arrays['motor_torque'][i],np.linalg.norm(arrays['closure_position'][i]),arrays['dynamics_tangent_residual'][i]])
    return arrays,dict(mass_matrix_max_abs_error=max_mass_difference,bias_force_max_abs_error=max_bias_difference),len(frames)


def audit(config,output,arrays,engine_checks,frame_count):
    time=arrays['time']; torque=arrays['motor_torque']
    def dynamics(t,state):return [state[1],reduced_acceleration(state[0],state[1],np.interp(t,time,torque),config)]
    independent=solve_ivp(dynamics,[time[0],time[-1]],[config['initial_crank_angle_rad'],0.],t_eval=time,
                          method='DOP853',rtol=2e-9,atol=1e-11,max_step=0.002)
    assert independent.success,independent.message
    reference=np.array([inverse_kinematics(theta,config) for theta in independent.y[0]])
    np.savez_compressed(output/'independent_reference.npz',time=time,q=reference,crank_velocity=independent.y[1])
    angle_error=arrays['q']-reference
    # Long open-loop replay can amplify tiny integration/compliance errors.
    # Also audit fixed 50-ms prediction windows from measured initial states.
    window_errors=[]
    window_size=max(1,round(0.05/(time[1]-time[0])))
    for start in range(0,len(time)-1,window_size):
        end=min(start+window_size,len(time)-1)
        window=solve_ivp(dynamics,[time[start],time[end]],
                         [arrays['q'][start,0],arrays['qdot'][start,0]],
                         t_eval=time[start:end+1],method='DOP853',rtol=2e-9,atol=1e-11,max_step=0.002)
        assert window.success,window.message
        window_reference=np.array([inverse_kinematics(theta,config) for theta in window.y[0]])
        window_errors.append(np.max(np.abs(arrays['q'][start:end+1]-window_reference)))
    energy=arrays['kinetic_energy']+arrays['potential_energy']
    power=arrays['motor_power']+arrays['friction_power']+arrays['external_power']+arrays['constraint_power']
    balance=energy-energy[0]-cumulative_trapezoid(power,time,initial=0)
    lengths=sorted([config['ground_length_m'],*config['lengths_m']])
    report={
        'case':output.name,'source':'MuJoCo torque-driven trajectory; independent NumPy/SciPy Lagrangian validation',
        'mujoco_version':mujoco.__version__,'duration_s':float(time[-1]),'physics_record_count':len(time),'video_frame_count':frame_count,
        'grashof_margin_m':float(lengths[1]+lengths[2]-lengths[0]-lengths[3]),
        'tree_dof':3,'independent_closure_constraint_rank':2,'mechanism_dof':1,
        'closure_position_max_m':float(np.max(np.linalg.norm(arrays['closure_position'],axis=1))),
        'closure_velocity_max_m_s':float(np.max(np.linalg.norm(arrays['closure_velocity'],axis=1))),
        'closure_acceleration_max_m_s2':float(np.max(np.linalg.norm(arrays['closure_acceleration'],axis=1))),
        'tangent_dynamics_residual_max_Nm':float(np.max(np.abs(arrays['dynamics_tangent_residual']))),
        'energy_work_balance_max_J':float(np.max(np.abs(balance))),
        'constraint_work_total_J':float(np.trapz(arrays['constraint_power'],time)),
        'friction_dissipated_energy_J':float(-np.trapz(arrays['friction_power'],time)),
        'independent_replayed_torque_angle_rmse_rad':float(np.sqrt(np.mean(angle_error**2))),
        'independent_replayed_torque_angle_max_error_rad':float(np.max(np.abs(angle_error))),
        'independent_replayed_torque_window_s':0.05,
        'independent_replayed_torque_window_max_error_rad':float(max(window_errors)),
        'crank_revolutions':float((arrays['q'][-1,0]-arrays['q'][0,0])/(2*np.pi)),
        **engine_checks,
        'limitations':['MuJoCo equality constraints have finite compliance; closure residuals are reported, not hidden.',
                       'Friction uses a smooth Coulomb approximation and does not model exact static sticking.',
                       'Physical parameters are synthetic specified values, not measured real hardware.',
                       'This is a basic single-loop 4R mechanism, not a claim of coverage of complex multi-loop mechanisms.']}
    thresholds={'closure_position_max_m':2e-5,'tangent_dynamics_residual_max_Nm':1e-8,
                'mass_matrix_max_abs_error':1e-10,'bias_force_max_abs_error':1e-9,
                'energy_work_balance_max_J':2e-3,'independent_replayed_torque_window_max_error_rad':0.002,
                'independent_replayed_torque_angle_max_error_rad':0.02}
    report['acceptance_thresholds']=thresholds
    report['checks']={key:report[key]<limit for key,limit in thresholds.items()}
    report['checks']['positive_grashof_margin']=report['grashof_margin_m']>0
    report['checks']['friction_never_adds_energy']=bool(np.max(arrays['friction_power'])<1e-10)
    report['checks']['finite_trajectory']=all(bool(np.isfinite(x).all()) for x in arrays.values())
    report['passed']=all(report['checks'].values())
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    instructions={'task':'Predict future motion of the planar four-bar from past visual observations and applied torque/force history.',
                  'description':'Blue crank, orange coupler and green rocker form a closed loop with the fixed ground. Actuation is a torque at the left pivot.',
                  'video':'mujoco_rgb.mp4','actions':'actions.npz','privileged_ground_truth':'ground_truth.npz',
                  'split_rule':'Provide only the observed video prefix and specified future actions to the model. Future ground truth and physical audit metadata are evaluator-only unless a separate parameter-given track is declared.'}
    (output/'interaction.json').write_text(json.dumps(instructions,indent=2)+'\n')
    plot(output,arrays,reference,balance)
    print(json.dumps(report,indent=2),flush=True)
    return report


def plot(output,arrays,reference,balance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    time=arrays['time']; fig,ax=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    for i,name in enumerate(['crank','coupler','rocker']):
        ax[0,0].plot(time,arrays['q'][:,i],label=name)
        ax[0,0].plot(time,reference[:,i],'--',alpha=.6)
    ax[0,0].set_title('MuJoCo angles / independent reference (dashed)');ax[0,0].set_ylabel('rad');ax[0,0].legend()
    ax[0,1].plot(time,arrays['motor_torque']);ax[0,1].set_title('Applied motor torque');ax[0,1].set_ylabel('N m')
    ax[1,0].semilogy(time,np.maximum(np.linalg.norm(arrays['closure_position'],axis=1),1e-15));ax[1,0].set_title('Measured loop closure error');ax[1,0].set_ylabel('m')
    ax[1,1].plot(time,arrays['dynamics_tangent_residual']);ax[1,1].set_title('Independent tangent dynamics residual');ax[1,1].set_ylabel('N m')
    ax[2,0].plot(time,balance);ax[2,0].set_title('Mechanical energy minus accounted work');ax[2,0].set_ylabel('J')
    for key in ['motor_power','friction_power','external_power','constraint_power']:ax[2,1].plot(time,arrays[key],label=key)
    ax[2,1].set_title('Power accounting');ax[2,1].set_ylabel('W');ax[2,1].legend(fontsize=8)
    for axis in ax.flat:axis.set_xlabel('time (s)');axis.grid(alpha=.2)
    fig.savefig(output/'physics_diagnostics.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=['nominal','heavy_payload','high_friction'],default='nominal')
    parser.add_argument('--all-cases',action='store_true')
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--duration',type=float)
    args=parser.parse_args()
    base=json.loads((Path(__file__).parent/'config.json').read_text())
    if args.duration is not None:base['simulation']['duration_s']=args.duration
    cases=['nominal','heavy_payload','high_friction'] if args.all_cases else [args.case]
    reports=[]
    for case in cases:
        config=copy.deepcopy(base)
        if case=='heavy_payload':config['payload_mass_kg']=0.25
        if case=='high_friction':
            config['viscous_friction_Nm_s_rad']=[3*x for x in config['viscous_friction_Nm_s_rad']]
            config['coulomb_friction_Nm']=[3*x for x in config['coulomb_friction_Nm']]
        output=ROOT/'output'/'fourbar'/case
        arrays,checks,count=simulate(config,output,args.render)
        reports.append(audit(config,output,arrays,checks,count))
    if not all(r['passed'] for r in reports):raise SystemExit('Physical validation failed; inspect validation.json')


if __name__=='__main__':main()
