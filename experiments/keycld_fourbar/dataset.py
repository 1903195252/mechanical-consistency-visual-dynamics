"""Generate the paper's random, constant-action image sequences for four-bar."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np


def initialize(assets):sys.path.insert(0,str(assets))


def generate(task):
    index,split,base,root=task;root=Path(root)
    import mujoco
    from simulate import make_model,vector
    from mechanics import TREE_TO_ABSOLUTE,closure,closure_jacobian,mass_bias_energy,friction,coupler_point
    rng=np.random.RandomState(index)
    action=rng.uniform(-2.,2.,3);action[1:]=0.
    if rng.rand()<.2:action[:]=0.
    torque=float(action[0])
    c=copy.deepcopy(base)
    c['initial_crank_angle_rad']=float(rng.uniform(-np.pi,np.pi));initial_speed=float(rng.uniform(-.1,.1))
    c['initial_crank_velocity_rad_s']=initial_speed
    c['drive']={'type':'constant_open_loop_torque','torque_Nm':torque,'torque_limit_Nm':2.}
    if split=='heavy':c['payload_mass_kg']=.25
    if split=='friction':
        c['viscous_friction_Nm_s_rad']=[3*x for x in base['original_viscous_friction']]
        c['coulomb_friction_Nm']=[3*x for x in base['original_coulomb_friction']]
    else:
        c['viscous_friction_Nm_s_rad']=[0.]*4;c['coulomb_friction_Nm']=[0.]*4
    # Simulator/force generation uses SI parameters; the learner never opens them.
    xml,q0=make_model(c);tree=ET.fromstring(xml)
    camera=tree.find('.//camera[@name="main"]')
    center=np.array([c['ground_length_m']/2,c['pivot_height_m']+.05]);half=.43
    c['camera']={'position_m':[float(center[0]),-10.,float(center[1])],
                 'target_m':[float(center[0]),0.,float(center[1])],
                 'vertical_fov_deg':math.degrees(2*math.atan(half/10))}
    camera.set('pos',vector([center[0],-10.,center[1]]));camera.set('xyaxes','1 0 0 0 0 1')
    camera.set('fovy',str(math.degrees(2*math.atan(half/10))))
    actuator=tree.find('actuator')
    for name,joint in [('coupler_motor','B'),('rocker_motor','D')]:
        ET.SubElement(actuator,'motor',name=name,joint=joint,gear='1',ctrllimited='true',ctrlrange='-2 2')
    model=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'));data=mujoco.MjData(model)
    J=closure_jacobian(q0,c);derivative=np.r_[1.,np.linalg.solve(J[:,1:],-J[:,0])]
    velocity=derivative*initial_speed
    data.qvel[:]=np.linalg.solve(TREE_TO_ABSOLUTE,velocity);data.ctrl[:]=action
    renderer=mujoco.Renderer(model,64,64);options=mujoco.MjvOption();options.sitegroup[:]=0
    step_count=round(.01/model.opt.timestep)
    force=np.array(c['external_force_xz_N']);has_friction=split=='friction'
    l1=c['lengths_m'][0];along=c['force_application_along_coupler_m']
    def applied():
        q1=q0[0]+data.qpos[0];q2=q0[1]+data.qpos[0]+data.qpos[1]
        f1=l1*(-math.sin(q1)*force[0]+math.cos(q1)*force[1])
        f2=along*(-math.sin(q2)*force[0]+math.cos(q2)*force[1])
        data.qfrc_applied[:]=[f1+f2,f2,0.]
        if has_friction:
            f,_,_=friction(TREE_TO_ABSOLUTE@data.qvel,c)
            data.qfrc_applied[:]+=TREE_TO_ABSOLUTE.T@f
    images=[];records=[];max_closure=0.;max_residual=0.;max_mass=0.;max_bias=0.
    dense=np.zeros((3,3))
    for frame in range(50):
        for _ in range(step_count):applied();mujoco.mj_step(model,data)
        applied();mujoco.mj_forward(model,data)
        q=q0+TREE_TO_ABSOLUTE@data.qpos;v=TREE_TO_ABSOLUTE@data.qvel;a=TREE_TO_ABSOLUTE@data.qacc
        M,C,G,kinetic,potential=mass_bias_energy(q,v,c);f,_,_=friction(v,c)
        _,JF=coupler_point(q,c,along);jac=closure_jacobian(q,c)
        tangent=np.cross(jac[0],jac[1]);tangent/=np.linalg.norm(tangent)
        residual=tangent@(M@a+C+G-np.array([torque,0,0])-f-JF.T@force)
        max_closure=max(max_closure,float(np.linalg.norm(closure(q,c))))
        max_residual=max(max_residual,abs(float(residual)))
        mujoco.mj_fullM(model,dense,data.qM)
        max_mass=max(max_mass,float(np.max(np.abs(dense-TREE_TO_ABSOLUTE.T@M@TREE_TO_ABSOLUTE))))
        max_bias=max(max_bias,float(np.max(np.abs(data.qfrc_bias-TREE_TO_ABSOLUTE.T@(C+G)))))
        renderer.update_scene(data,camera='main',scene_option=options);images.append(renderer.render().copy())
        records.append((data.time,q.copy(),v.copy(),a.copy(),kinetic,potential))
    renderer.close()
    assert max_closure<2e-5 and max_residual<1e-8 and max_mass<1e-10 and max_bias<1e-9,(index,max_closure,max_residual,max_mass,max_bias)
    times=np.array([r[0] for r in records]);images=np.asarray(images,dtype=np.uint8)
    assert np.isfinite(np.asarray([r[1] for r in records])).all()
    observation=root/'observations'/f'run_{index:04d}.npz'
    np.savez_compressed(observation,x=images,t=times.astype(np.float32),action=np.array([torque,0,0],dtype=np.float32))
    truth=root/'truth'/f'run_{index:04d}.npz'
    np.savez_compressed(truth,time=times,q=np.array([r[1] for r in records]),qdot=np.array([r[2] for r in records]),
                        qddot=np.array([r[3] for r in records]),kinetic=np.array([r[4] for r in records]),potential=np.array([r[5] for r in records]))
    (root/'truth'/f'run_{index:04d}.json').write_text(json.dumps(c,indent=2)+'\n')
    return {'index':index,'split':split,'torque_Nm':float(torque),'initial_angle_rad':c['initial_crank_angle_rad'],
            'initial_speed_rad_s':initial_speed,'closure_max_m':max_closure,'force_residual_max_Nm':max_residual,
            'mass_matrix_error_max':max_mass,'bias_error_max':max_bias,
            'observation_sha256':hashlib.sha256(observation.read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True);parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();start=time.monotonic()
    for name in ['observations','truth']:(args.root/name).mkdir(parents=True,exist_ok=True)
    base=json.loads((args.assets/'config.json').read_text())
    protocol=json.loads((args.root/'training_protocol.json').read_text())
    base['original_viscous_friction']=base['viscous_friction_Nm_s_rad']
    base['original_coulomb_friction']=base['coulomb_friction_Nm']
    base['simulation']['width']=64;base['simulation']['height']=64;base['simulation']['video_fps']=100
    base['simulation']['duration_s']=.5;base['simulation']['record_hz']=100
    base['simulation']['timestep_s']=protocol['dataset']['physics_timestep_s']
    base['simulation']['equality_solref_s']=protocol['dataset']['equality_solref_s']
    ids=[(i,'train' if i<450 else 'validation' if i<500 else 'test') for i in range(550)]
    ids+=[(i,'heavy') for i in range(600,620)]+[(i,'friction') for i in range(700,720)]
    tasks=[(i,split,base,str(args.root)) for i,split in ids]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(str(args.assets),)) as pool:
        for future in as_completed([pool.submit(generate,task) for task in tasks]):
            results.append(future.result())
            if len(results)%20==0:print('Generated and physics-checked',len(results),'of',len(tasks),flush=True)
    results.sort(key=lambda r:r['index'])
    (args.root/'dataset_manifest.json').write_text(json.dumps({'runs':results,'elapsed_seconds':time.monotonic()-start,
                                                           'train_images_only':True,'physics_checked_per_run':True},indent=2)+'\n')
    center=np.array([base['ground_length_m']/2,base['pivot_height_m']+.05]);half=.43*63/64
    geometry={'keypoint_order':['B','C','midpoint'],'half_extent_m':half,'world_xz_center_m':center.tolist(),
              'A_normalized':((np.array([0,base['pivot_height_m']])-center)/half).tolist(),
              'D_normalized':((np.array([base['ground_length_m'],base['pivot_height_m']])-center)/half).tolist(),
              'link_lengths_normalized':(np.asarray(base['lengths_m'])/half).tolist(),
              'camera_note':'parallel-plane perspective camera at 10 m; planar Y=0 calibration and pixel-centre convention',
              'observation_count':3,'video_fps':100}
    (args.root/'geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    print('Dataset complete:',len(results),'independent trajectories',flush=True)


if __name__=='__main__':main()
