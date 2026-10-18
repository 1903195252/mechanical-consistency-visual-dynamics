"""Calibrate image-plane motion and compare depth instruments on existing RGB.

This reuses the registered calibration videos. Optical platform translation is
measured independently of the depth readout; no command or force is fitted.
"""
import json
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from .common import ROOT,write,sha
from .delta_instrument import rows,reference,DIRECTORY,CALIBRATION,coordinate_errors
from .delta_optical import OPTICAL,project_surface


def positions(row,tracks):
    camera=json.loads((ROOT/row['public_inputs']).read_text())['observed_camera_frames'][0]
    c=json.loads(((ROOT/row['evaluator_directory']).parents[1]/'assembly.json').read_text())['config']
    own,local=tracks['initial_owner'],tracks['initial_local']
    initial=np.array([0.,0.,c['platform_height_m']]);start=tracks['tracked_pixels'][0]
    offset=start-project_surface(initial,local,own,c,camera)
    R=np.asarray(camera['R_world_to_camera']);result=[initial.copy()]
    plane=[initial.copy()]
    for target,valid in zip(tracks['tracked_pixels'][1:],tracks['valid'][1:]):
        platform=valid&(own==0)
        if platform.sum()<4:raise ValueError('Too few platform image correspondences')
        delta=np.median((target-start)[platform],axis=0)*camera['scale_m']/256
        p=initial+R.T@np.r_[delta,0.]
        plane.append(p.copy())
        # Distal upper-arm yokes have a translational image signature, whereas
        # circular rotor hubs and partly occluded thin beams are unreliable.
        good=valid&(own==3)&(local[:,0]>.1)&(local[:,2]>.04)
        if good.sum()<3:
            result.append(np.full(3,np.nan));continue
        direction=R[2]
        def residual(depth):
            value=p+direction*float(depth[0])
            try:return (project_surface(value,local[good],own[good],c,camera)+offset[good]-target[good]).ravel()
            except ValueError:return np.full(good.sum()*2,1e3)
        initial_depth=float((result[-1]-p)@direction) if np.isfinite(result[-1]).all() else 0.
        fit=least_squares(residual,[initial_depth],loss='soft_l1',f_scale=.15,
            diff_step=1e-5,max_nfev=60,xtol=1e-10,ftol=1e-10,gtol=1e-9)
        result.append(p+direction*fit.x[0])
    return np.array(plane),np.array(result)


def run():
    candidates={};reports=[]
    for row in rows('calibration'):
        model,q,v,a=reference(row)
        with np.load(OPTICAL/'development'/(row['video_id']+'.npz')) as d:plane,optical=positions(row,d)
        camera=json.loads((ROOT/row['public_inputs']).read_text())['observed_camera_frames'][0]
        direction=np.asarray(camera['R_world_to_camera'])[2]
        with np.load(CALIBRATION/'readouts/geometry_spatial_256'/(row['video_id']+'.npz')) as d:old=d['qpos'].astype(float)
        old[:,:3]+=model.qpos0[:3]-old[0,:3]
        with np.load(DIRECTORY/'development_readouts'/(row['video_id']+'.npz')) as d:relative=d['qpos']
        pool={'distal_optical':optical,
            'reset_depth':plane+np.outer((old[:,:3]-plane)@direction,direction),
            'relative_depth':plane+np.outer((relative[:,:3]-plane)@direction,direction)}
        case={}
        for kind,p in pool.items():
            for window in (1,):
                key=kind+'/'+str(window)
                pp=savgol_filter(p,window,3,axis=0) if window>1 else p
                estimate=q.copy();estimate[:,:3]=pp
                error=coordinate_errors(estimate,q,v,a) if np.isfinite(estimate).all() else [None]*3
                case[key]=error;candidates.setdefault(key,[]).append(error)
        target=OPTICAL/'fusion_development'/(row['video_id']+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,**pool)
        reports.append(dict(id=row['video_id'],candidates=case))
    means={k:np.mean(v,axis=0).tolist() if all(None not in row for row in v) else None for k,v in candidates.items()}
    objectives={k:float(np.max(np.array(v)/[.001,.01,.1])) if v is not None else None for k,v in means.items()}
    selected=min((k for k in objectives if objectives[k] is not None),key=objectives.get)
    write(OPTICAL/'fusion_selection.json',dict(selected=selected,mean_errors=means,objectives=objectives,cases=reports,
        source_sha256=sha(__file__),fit_mechanical_residuals=False,verification_opened=False))
    print(json.dumps(dict(selected=selected,mean_errors=means,objectives=objectives)),flush=True)


if __name__=='__main__':run()
