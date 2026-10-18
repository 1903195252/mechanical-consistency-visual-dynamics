"""Rectify four-bar video and isolate model inputs from evaluator-only truth."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/keycld_fourbar'


def points(q,v,a,c):
    l1,l2,_=c['lengths_m'];height=c['pivot_height_m']
    e=lambda theta:np.stack([np.cos(theta),np.sin(theta)],axis=-1)
    n=lambda theta:np.stack([-np.sin(theta),np.cos(theta)],axis=-1)
    B=l1*e(q[:,0])+[0,height];C=B+l2*e(q[:,1])
    vB=l1*n(q[:,0])*v[:,0,None];vC=vB+l2*n(q[:,1])*v[:,1,None]
    aB=l1*(n(q[:,0])*a[:,0,None]-e(q[:,0])*v[:,0,None]**2)
    aC=aB+l2*(n(q[:,1])*a[:,1,None]-e(q[:,1])*v[:,1,None]**2)
    return np.stack([B,C],axis=1),np.stack([vB,vC],axis=1),np.stack([aB,aC],axis=1)


def main():
    import cv2
    OUT.mkdir(exist_ok=True,parents=True)
    case=ROOT/'output/fourbar/nominal'
    c=json.loads((case/'physical_metadata.json').read_text())
    cam=json.loads((case/'blender_camera.json').read_text())
    video=case/'blender_rgb.mp4';cap=cv2.VideoCapture(str(video))
    K=np.asarray(cam['K_pixels']);R=np.asarray(cam['R_world_to_camera_cv']);t=np.asarray(cam['t_world_to_camera_cv_m'])
    size=64;half=.43;center=np.array([c['ground_length_m']/2,c['pivot_height_m']+.05])
    H=K@np.column_stack([R[:,0],R[:,2],t]) # rectification plane Y=0
    pixel_to_world=np.array([[2*half/(size-1),0,center[0]-half],
                             [0,-2*half/(size-1),center[1]+half],[0,0,1.]])
    mapping=H@pixel_to_world
    images=[]
    while True:
        okay,frame=cap.read()
        if not okay:break
        rectified=cv2.warpPerspective(frame,mapping,(size,size),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_CONSTANT,borderValue=(128,128,128))
        images.append(cv2.cvtColor(rectified,cv2.COLOR_BGR2RGB))
    cap.release();images=np.asarray(images,dtype=np.float32)/255
    with np.load(case/'video_trajectory.npz') as data:
        times=data['time'].copy();q=data['q'].copy()
    assert len(images)==len(times)==121
    with np.load(case/'actions.npz') as data:
        action_times=data['time'].copy()
        actions=np.column_stack([data['motor_torque_Nm'],data['external_force_xz_N']])
    observed_count=3
    np.savez_compressed(OUT/'model_input.npz',observed_x=images[:observed_count],time=times,
                        observed_time=times[:observed_count],action_time=action_times,actions=actions)
    # Geometry/calibration-given track. No mass, inertia, friction or future state.
    geometry={'keypoint_order':['B','C'],'normalized_coordinate_convention':'x right, y up in [-1,1]',
              'world_xz_center_m':center.tolist(),'half_extent_m':half,
              'A_normalized':((np.array([0,c['pivot_height_m']])-center)/half).tolist(),
              'D_normalized':((np.array([c['ground_length_m'],c['pivot_height_m']])-center)/half).tolist(),
              'link_lengths_normalized':(np.asarray(c['lengths_m'])/half).tolist(),
              'world_xz_to_source_pixel_homography':H.tolist(),
              'rectified_pixel_to_world_xz':pixel_to_world.tolist(),
              'action_channels':['motor_torque_Nm','external_force_x_N','external_force_z_N'],
              'observation_count':observed_count,'video_fps':c['simulation']['video_fps'],
              'assumption':'Calibrated planar rectification at Y=0; out-of-plane plate/connector parallax is retained.'}
    (OUT/'geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    # This file is evaluator-only and is NEVER uploaded to the inference process.
    with np.load(case/'ground_truth.npz') as data:
        interpolate=lambda name:np.column_stack([np.interp(times,data['time'],data[name][:,i]) for i in range(3)])
        velocity=interpolate('qdot');acceleration=interpolate('qddot')
    xy,xy_v,xy_a=points(q,velocity,acceleration,c)
    np.savez_compressed(OUT/'evaluation_reference.npz',time=times,x=images,q=q,qdot=velocity,qddot=acceleration,
                        keypoints_world_xz=xy,keypoint_velocity_m_s=xy_v,keypoint_acceleration_m_s2=xy_a)
    preview=cv2.cvtColor((np.concatenate(images[:3],axis=1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUT/'observed_prefix.png'),cv2.resize(preview,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))
    manifest={'purpose':'KeyCLD input/scoring diagnostic; no training',
              'track':'given geometry and camera calibration; RGB video prefix plus known future actions; no text encoder',
              'observed_frames':3,'future_frames':118,'image_shape':list(images[:3].shape),
              'source_video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
              'model_input_files':['model_input.npz','geometry.json'],
              'evaluator_only_files':['evaluation_reference.npz','../fourbar/nominal/physical_metadata.json'],
              'notes':['No future images or states in model_input.npz.',
                       'KeyCLD original action is constant per run; adapter supplies the actual time-varying actions through an RK4 solver.',
                       'Three existing clips do not constitute a held-out training/evaluation dataset.']}
    (OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
