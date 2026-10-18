"""Paired, independently audited target-condition RGB interaction trajectories."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np


def initialize(assets):sys.path.insert(0,str(assets))


def conditions(base):
    from mechanics import properties
    names=['nominal','viscous','coulomb','mixed_friction','heavy_payload','dense_bars','geometry_scaled','crank_slider']
    configs={}
    for name in names:
        c=copy.deepcopy(base);c['mechanism']='fourbar';c['condition']=name
        b=base['viscous_friction_Nm_s_rad'];fc=base['coulomb_friction_Nm']
        c['viscous_friction_Nm_s_rad']=b.copy() if name in ['viscous','mixed_friction'] else [0.]*4
        c['coulomb_friction_Nm']=fc.copy() if name in ['coulomb','mixed_friction'] else [0.]*4
        if name=='heavy_payload':c['payload_mass_kg']=.25
        if name=='dense_bars':c['density_kg_m3']*=1.5
        if name=='geometry_scaled':
            c['lengths_m']=[.8*x for x in c['lengths_m']];c['ground_length_m']*=.8
            c['payload_along_coupler_m']*=.8;c['force_application_along_coupler_m']*=.8
        if name=='crank_slider':
            c['mechanism']='crank_slider';c['slider_mass_kg']=float(properties(base)[0][2])
            c['slider_size_m']=[.055,.024,.044]
        c['simulation'].update(width=64,height=64,video_fps=100,record_hz=100,duration_s=.5,
                               timestep_s=1/96000,equality_solref_s=.000025)
        c['camera']={'position_m':[.2,-10.,.37],'target_m':[.2,0.,.37],
                     'vertical_fov_deg':math.degrees(2*math.atan(.43/10))}
        configs[name]=c
    return configs


def xml_model(c):
    from simulate import make_model,vector,rotation
    from extended_mechanics import inverse,is_slider
    xml,q0=make_model(c);tree=ET.fromstring(xml)
    camera=tree.find('.//camera[@name="main"]');camera.set('xyaxes','1 0 0 0 0 1')
    if is_slider(c):
        q0=inverse(c['initial_crank_angle_rad'],c)
        tree.find('.//body[@name="crank"]').set('quat',rotation(q0[0]))
        tree.find('.//body[@name="coupler"]').set('quat',rotation(q0[1]-q0[0]))
        slider=tree.find('.//body[@name="rocker"]');slider.clear()
        slider.set('name','slider');slider.set('pos',vector([q0[2],0.,c['pivot_height_m']]))
        ET.SubElement(slider,'joint',name='S',type='slide',axis='1 0 0')
        size=np.asarray(c['slider_size_m']);m=c['slider_mass_kg'];inertia=m*(np.sum(size**2)-size**2)/12
        ET.SubElement(slider,'inertial',pos=vector([0.,c['visual_and_com_y_m'][2],0.]),mass=str(m),diaginertia=vector(inertia))
        ET.SubElement(slider,'geom',type='box',pos=vector([0.,c['visual_and_com_y_m'][2],0.]),size=vector(size/2),rgba='0.15 0.68 0.43 1')
        tree.find('.//connect[@name="C_loop"]').set('body2','slider')
        rail=tree.find('.//geom[@name="ground_link"]');rail.set('pos',vector([.28,.055,c['pivot_height_m']-.016]));rail.set('size',vector([.29,.02,.016]))
    return ET.tostring(tree,encoding='unicode'),q0


def generate(task):
    name,index,config,root=task;root=Path(root)
    import mujoco
    from extended_mechanics import derivatives,jacobian,closure,mass_bias,friction,terms,external
    from mechanics import TREE_TO_ABSOLUTE
    rng=np.random.RandomState(index);action=rng.uniform(-2.,2.,3);action[1:]=0.
    if rng.rand()<.2:action[:]=0.
    theta=float(rng.uniform(-np.pi,np.pi));omega=float(rng.uniform(-.1,.1))
    c=copy.deepcopy(config);c['initial_crank_angle_rad']=theta;c['initial_crank_velocity_rad_s']=omega
    c['drive']={'type':'constant_open_loop_torque','torque_Nm':float(action[0]),'torque_limit_Nm':2.}
    xml,q0=xml_model(c);model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    assert model.nv==3 and model.nu==1
    S=TREE_TO_ABSOLUTE;gamma,_=derivatives(q0,c);data.qvel[:]=np.linalg.solve(S,gamma*omega);data.ctrl[0]=action[0]
    renderer=mujoco.Renderer(model,64,64);options=mujoco.MjvOption();options.sitegroup[:]=0
    l1=c['lengths_m'][0];along=c['force_application_along_coupler_m'];Fx,Fz=c['external_force_xz_N']
    has_friction=bool(np.any(c['viscous_friction_Nm_s_rad']) or np.any(c['coulomb_friction_Nm']))
    def applied():
        q1=q0[0]+data.qpos[0];q2=q0[1]+data.qpos[0]+data.qpos[1]
        f1=l1*(-math.sin(q1)*Fx+math.cos(q1)*Fz);f2=along*(-math.sin(q2)*Fx+math.cos(q2)*Fz)
        data.qfrc_applied[:]=[f1+f2,f2,0.]
        if has_friction:data.qfrc_applied[:]+=S.T@friction(S@data.qvel,c)
    records=[];images=[];max_closure=max_force=max_mass=max_bias=0.;dense=np.zeros((3,3))
    for frame in range(50):
        for _ in range(960):applied();mujoco.mj_step(model,data)
        applied();mujoco.mj_forward(model,data)
        q=q0+S@data.qpos;v=S@data.qvel;a=S@data.qacc;M,C,G=mass_bias(q,v,c)
        gamma,_=derivatives(q,c)
        residual=gamma@(M@a+C+G-np.array([action[0],0.,0.])-friction(v,c)-external(q,c))
        mujoco.mj_fullM(model,dense,data.qM)
        max_closure=max(max_closure,float(np.linalg.norm(closure(q,c))))
        max_force=max(max_force,abs(float(residual)))
        max_mass=max(max_mass,float(np.max(np.abs(dense-S.T@M@S))))
        max_bias=max(max_bias,float(np.max(np.abs(data.qfrc_bias-S.T@(C+G)))))
        renderer.update_scene(data,camera='main',scene_option=options);images.append(renderer.render().copy())
        records.append((data.time,q.copy(),v.copy(),a.copy()))
    renderer.close()
    assert max_closure<2e-5 and max_force<1e-8 and max_mass<1e-10 and max_bias<1e-9,(name,index,max_closure,max_force,max_mass,max_bias)
    observations=root/'observations'/name;truth=root/'truth'/name
    observations.mkdir(parents=True,exist_ok=True);truth.mkdir(parents=True,exist_ok=True)
    path=observations/f'run_{index:04d}.npz'
    np.savez_compressed(path,x=np.asarray(images,dtype=np.uint8),t=np.array([r[0] for r in records],dtype=np.float32),action=action.astype(np.float32))
    np.savez_compressed(truth/f'run_{index:04d}.npz',time=np.array([r[0] for r in records]),q=np.array([r[1] for r in records]),
                        qdot=np.array([r[2] for r in records]),qddot=np.array([r[3] for r in records]))
    (truth/f'run_{index:04d}.json').write_text(json.dumps(c,indent=2)+'\n')
    if index==1000:(root/f'{name}_model.xml').write_text(xml)
    return {'condition':name,'index':index,'initial_angle_rad':theta,'initial_speed_rad_s':omega,'torque_Nm':float(action[0]),
            'closure_max_m':max_closure,'motor_torque_residual_max_Nm':max_force,'mass_error_max':max_mass,'bias_error_max':max_bias,
            'observation_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
