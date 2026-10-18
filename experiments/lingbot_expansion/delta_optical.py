"""Image-tracked CAD reconstruction for Delta; no mechanical motion prior.

Surface points are anchored once to the public CAD reset. Future points are
tracked solely in RGB. Least-squares camera reprojection reconstructs platform
translation; the closure intervention remains explicit, and image fit errors
and tracking losses are retained rather than replaced by command trajectories.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from .common import ROOT,OUT,write,sha
from .delta_instrument import rows,reference,DIRECTORY

OPTICAL=DIRECTORY/'optical'


def raymap(row):
    import mujoco
    folder=ROOT/row['evaluator_directory'];assembly=folder.parents[1]
    output=OPTICAL/'raymaps'/(row['assembly_id']+'.npz');output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():return output
    model=mujoco.MjModel.from_xml_path(str(folder/'model.xml'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    nominal={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i):dict(position=data.xpos[i].tolist(),quaternion=data.xquat[i].tolist()) for i in range(1,model.nbody)}
    public=json.loads((ROOT/row['public_inputs']).read_text())
    task=dict(blend=str(assembly/'assembly.blend'),nominal_bodies=nominal,
        camera=public['observed_camera_frames'][0],width=256,output=str(output))
    spec=output.with_suffix('.json');write(spec,task)
    env=os.environ.copy();env['OMP_NUM_THREADS']='2'
    with output.with_suffix('.log').open('w') as log:
        subprocess.run(['/opt/blender-5.1.1-linux-x64/blender-launcher','-b','-t','2','--python',
            str(ROOT/'experiments/lingbot_expansion/delta_raymap.py'),'--',str(spec)],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    return output


def images(row):
    cap=cv2.VideoCapture(str(ROOT/row['video']));frames=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(cv2.resize(frame,(256,256),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY))
    finally:cap.release()
    if len(frames)!=120:raise ValueError('Incomplete RGB sequence')
    return np.stack(frames)


def project_surface(p,local,owner,c,camera):
    from experiments.mechanism_corpus.delta import inverse_kinematics,rz,ry
    angles=inverse_kinematics(p,c)
    world=local.copy()
    world[owner==0]+=p
    for i,phi in enumerate(c['leg_azimuths_rad']):
        mask=owner==i+1
        world[mask]=local[mask]@(rz(phi)@ry(c['upper_reference_angle_rad']+angles[i])).T+[
            c['base_radius_m']*np.cos(phi),c['base_radius_m']*np.sin(phi),c['base_height_m']]
    camera_xyz=world@np.asarray(camera['R_world_to_camera']).T+camera['t_world_to_camera_m']
    return camera_xyz[:,:2]*256/camera['scale_m']+127.5


def track_rgb(row,frames):
    """Measured feature tracks, independently of any reconstructed 3D pose."""
    cv2.setNumThreads(1)
    with np.load(raymap(row)) as d:owner_map,local_map=d['owner'],d['local']
    pixels=[]
    for body in range(4):
        corners=cv2.goodFeaturesToTrack(frames[0],maxCorners=80,qualityLevel=.003,minDistance=2,
            mask=np.uint8(owner_map==body)*255,blockSize=3)
        if corners is not None:pixels.extend(corners[:,0])
    indices=np.rint(np.asarray(pixels,dtype=np.float32)).astype(int)
    owners=owner_map[indices[:,1],indices[:,0]];local=local_map[indices[:,1],indices[:,0]].astype(float)
    start=indices.astype(np.float32).reshape(-1,1,2);guess=start.copy()
    options=dict(winSize=(9,9),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,.0001),minEigThreshold=1e-6)
    targets=[start[:,0].copy()];validity=[np.ones(len(start),bool)]
    for frame in frames[1:]:
        end,status,_=cv2.calcOpticalFlowPyrLK(frames[0],frame,start,guess.copy(),flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**options)
        back,reverse,_=cv2.calcOpticalFlowPyrLK(frame,frames[0],end,None,**options)
        good=(status[:,0]==1)&(reverse[:,0]==1)&(np.linalg.norm(back[:,0]-start[:,0],axis=1)<.35)
        guess=end;targets.append(end[:,0].copy());validity.append(good)
    return dict(tracked_pixels=np.array(targets),valid=np.array(validity),initial_owner=owners,initial_local=local)


def platform_plane(row,tracks):
    camera=json.loads((ROOT/row['public_inputs']).read_text())['observed_camera_frames'][0]
    from experiments.closed_chain_eval.geometry_spatial_reader import nominal
    initial=nominal(row)[:3].astype(float);R=np.asarray(camera['R_world_to_camera'])
    start=tracks['tracked_pixels'][0];owner=tracks['initial_owner'];result=[];spread=[];counts=[]
    for target,valid in zip(tracks['tracked_pixels'],tracks['valid']):
        good=valid&(owner==0)
        if good.sum()<4:raise ValueError('Too few platform image correspondences')
        pixels=(target-start)[good];delta=np.median(pixels,axis=0)
        result.append(initial+R.T@np.r_[delta*camera['scale_m']/256,0.])
        spread.append(float(np.median(np.linalg.norm(pixels-delta,axis=1))));counts.append(int(good.sum()))
    return np.array(result),dict(platform_motion_mad_px=spread,platform_tracks=counts)


def reconstruct(row,frames):
    cv2.setNumThreads(1)
    public=json.loads((ROOT/row['public_inputs']).read_text());camera=public['observed_camera_frames'][0]
    # Geometry comes from static CAD; material and mechanical coefficients are unused.
    assembly=(ROOT/row['evaluator_directory']).parents[1]
    c=json.loads((assembly/'assembly.json').read_text())['config']
    with np.load(raymap(row)) as d:owner_map,local_map=d['owner'],d['local']
    pixels=[]
    for body in range(4):
        mask=np.uint8(owner_map==body)*255
        corners=cv2.goodFeaturesToTrack(frames[0],maxCorners=80,qualityLevel=.003,minDistance=2,mask=mask,blockSize=3)
        if corners is not None:pixels.extend(corners[:,0])
    pixels=np.asarray(pixels,dtype=np.float32)
    indices=np.rint(pixels).astype(int);owners=owner_map[indices[:,1],indices[:,0]]
    local=local_map[indices[:,1],indices[:,0]].astype(float)
    start=indices.astype(np.float32).reshape(-1,1,2)
    options=dict(winSize=(9,9),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,.0001),minEigThreshold=1e-6)
    initial=np.array([0.,0.,c['platform_height_m']]);p=initial.copy()
    # Anchor projection to the actual reset pixel to remove raster-centre offsets.
    offset=start[:,0]-project_surface(initial,local,owners,c,camera)
    positions=[p.copy()];rms=[0.];counts=[len(start)];all_targets=[start[:,0].copy()];validity=[np.ones(len(start),bool)]
    guess=start.copy()
    for frame in frames[1:]:
        end,status,_=cv2.calcOpticalFlowPyrLK(frames[0],frame,start,guess.copy(),flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**options)
        back,reverse,_=cv2.calcOpticalFlowPyrLK(frame,frames[0],end,None,**options)
        good=(status[:,0]==1)&(reverse[:,0]==1)&(np.linalg.norm(back[:,0]-start[:,0],axis=1)<.35)
        target=end[:,0];guess=end
        if good.sum()<8 or not np.any(good&(owners==0)) or not np.any(good&np.isin(owners,[2,3])):
            raise ValueError('Insufficient independently tracked depth-sensitive CAD features')
        def residual(value):
            try:return (project_surface(value,local[good],owners[good],c,camera)+offset[good]-target[good]).ravel()
            except ValueError:return np.full(2*good.sum(),1e3)
        fit=least_squares(residual,p,loss='soft_l1',f_scale=.2,diff_step=1e-5,max_nfev=80,xtol=1e-10,ftol=1e-10,gtol=1e-9)
        p=fit.x;positions.append(p.copy());rms.append(float(np.sqrt(np.mean(residual(p)**2))));counts.append(int(good.sum()))
        all_targets.append(target.copy());validity.append(good.copy())
    return dict(position=np.array(positions),image_fit_rms_px=np.array(rms),tracked_features=np.array(counts),
        tracked_pixels=np.array(all_targets),valid=np.array(validity),initial_owner=owners,initial_local=local)


def develop():
    for row in rows('calibration'):
        target=OPTICAL/'development'/(row['video_id']+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():continue
        result=reconstruct(row,images(row))
        # Labels are opened only after completing all image-only estimates.
        _,q,v,a=reference(row)
        errors=result['position']-q[:,:3]
        report=dict(id=row['video_id'],platform_rmse_mm=float(np.sqrt(np.mean(np.sum(errors[3:]**2,axis=1))))*1000,
            bias_mm=(errors[3:].mean(0)*1000).tolist(),axis_rmse_mm=(np.sqrt(np.mean(errors[3:]**2,axis=0))*1000).tolist(),
            mean_image_fit_px=float(result['image_fit_rms_px'].mean()),minimum_tracks=int(result['tracked_features'].min()),
            source_sha256=sha(__file__),future_truth_used_in_reconstruction=False)
        np.savez_compressed(target,**result);write(target.with_suffix('.json'),report);print(json.dumps(report),flush=True)


def photometric(row,frames,initial):
    public=json.loads((ROOT/row['public_inputs']).read_text());camera=public['observed_camera_frames'][0]
    assembly=(ROOT/row['evaluator_directory']).parents[1]
    c=json.loads((assembly/'assembly.json').read_text())['config']
    with np.load(raymap(row)) as d:owner_map,local_map=d['owner'],d['local']
    selected=np.zeros_like(owner_map,dtype=bool)
    for body in (0,1,3):
        mask=np.uint8(owner_map==body)
        if body:mask &= np.uint8(local_map[:,:,0]>.08)
        selected |= cv2.erode(mask,np.ones((3,3),np.uint8)).astype(bool)
    y,x=np.where(selected);owners=owner_map[y,x];local=local_map[y,x].astype(float)
    offsets=np.stack([x,y],axis=1)-project_surface(np.array([0.,0.,c['platform_height_m']]),local,owners,c,camera)
    # High-pass photometry reduces lighting bias; no temporal smoothing or
    # mechanical trajectory is used in this image registration.
    high=np.stack([im.astype(float)-cv2.GaussianBlur(im.astype(float),(0,0),3) for im in frames])/255
    reference_values=high[0,y,x]
    positions=[np.array([0.,0.,c['platform_height_m']])];errors=[0.]
    for k,frame in enumerate(high[1:],1):
        def residual(p):
            try:uv=project_surface(p,local,owners,c,camera)+offsets
            except ValueError:return np.full(len(local),1e3)
            return map_coordinates(frame,[uv[:,1],uv[:,0]],order=3,mode='nearest',prefilter=True)-reference_values
        fit=least_squares(residual,initial[k],loss='soft_l1',f_scale=.02,diff_step=1e-5,max_nfev=50,xtol=1e-10,ftol=1e-10,gtol=1e-9)
        positions.append(fit.x);errors.append(float(np.sqrt(np.mean(residual(fit.x)**2))))
    return dict(position=np.array(positions),photometric_rms=np.array(errors),registered_pixels=np.array(len(local)))


def refine_development():
    for row in rows('calibration'):
        target=OPTICAL/'photometric'/(row['video_id']+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():continue
        with np.load(OPTICAL/'development'/(row['video_id']+'.npz')) as d:initial=d['position']
        result=photometric(row,images(row),initial)
        _,q,_,_=reference(row);errors=result['position']-q[:,:3]
        report=dict(id=row['video_id'],platform_rmse_mm=float(np.sqrt(np.mean(np.sum(errors[3:]**2,axis=1))))*1000,
            bias_mm=(errors[3:].mean(0)*1000).tolist(),axis_rmse_mm=(np.sqrt(np.mean(errors[3:]**2,axis=0))*1000).tolist(),
            source_sha256=sha(__file__),future_truth_used_in_reconstruction=False)
        np.savez_compressed(target,**result);write(target.with_suffix('.json'),report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['develop','photometric']);args=p.parse_args()
    if args.command=='develop':develop()
    else:refine_development()
