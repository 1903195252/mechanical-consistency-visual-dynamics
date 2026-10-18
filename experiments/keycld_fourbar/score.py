"""Independent geometry, force balance and energy metrics for KeyCLD outputs."""

import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import cumulative_trapezoid

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/keycld_fourbar'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'output/fourbar/.cache/matplotlib'))
sys.path.insert(0,str(ROOT/'experiments/fourbar'))
from mechanics import closure,closure_jacobian,closure_curvature,mass_bias_energy,friction,coupler_point


def rms(value):
    value=np.asarray(value);value=value[np.isfinite(value)]
    return float(np.sqrt(np.mean(value**2))) if value.size else None


def maximum(value):
    value=np.asarray(value);value=value[np.isfinite(value)]
    return float(np.max(np.abs(value))) if value.size else None


def metrics(xy,velocity,acceleration,times,actions,reference,c,observed=3):
    A=np.array([0,c['pivot_height_m']]);D=np.array([c['ground_length_m'],c['pivot_height_m']])
    B,C=xy[:,0],xy[:,1];vB,vC=velocity[:,0],velocity[:,1];aB,aC=acceleration[:,0],acceleration[:,1]
    r=np.stack([B-A,C-B,C-D],axis=1)
    v=np.stack([vB,vC-vB,vC],axis=1);a=np.stack([aB,aC-aB,aC],axis=1)
    radii=np.linalg.norm(r,axis=-1)
    valid=np.isfinite(r).all(axis=(1,2))&np.isfinite(v).all(axis=(1,2))&np.isfinite(a).all(axis=(1,2))&(radii>.001).all(axis=1)
    q=np.arctan2(r[:,:,1],r[:,:,0]);safe=np.maximum(radii**2,1e-30)
    cross=lambda left,right:left[:,:,0]*right[:,:,1]-left[:,:,1]*right[:,:,0]
    qdot=cross(r,v)/safe
    qddot=cross(r,a)/safe-2*np.sum(r*v,axis=-1)*cross(r,v)/safe**2
    quantities={name:np.full(len(times),np.nan) for name in ['closure_m','closure_velocity_m_s','closure_acceleration_m_s2',
                                                           'tangent_force_residual_Nm','energy_J','accounted_power_W']}
    for i in np.flatnonzero(valid):
        mass,coriolis,gravity,kinetic,potential=mass_bias_energy(q[i],qdot[i],c)
        f,_,_=friction(qdot[i],c)
        _,Jforce=coupler_point(q[i],c,c['force_application_along_coupler_m'])
        external=Jforce.T@actions[i,1:]
        applied=np.array([actions[i,0],0,0])+f+external
        J=closure_jacobian(q[i],c);tangent=np.cross(J[0],J[1]);norm=np.linalg.norm(tangent)
        if norm<1e-12:
            valid[i]=False;continue
        tangent/=norm
        quantities['closure_m'][i]=np.linalg.norm(closure(q[i],c))
        quantities['closure_velocity_m_s'][i]=np.linalg.norm(J@qdot[i])
        quantities['closure_acceleration_m_s2'][i]=np.linalg.norm(J@qddot[i]+closure_curvature(q[i],qdot[i],c))
        quantities['tangent_force_residual_Nm'][i]=tangent@(mass@qddot[i]+coriolis+gravity-applied)
        quantities['energy_J'][i]=kinetic+potential
        quantities['accounted_power_W'][i]=applied@qdot[i]
    future=np.arange(len(times))>=observed;usable=valid&future
    point_error=np.linalg.norm(xy-reference['keypoints_world_xz'],axis=-1)
    native_length_error=radii-np.asarray(c['lengths_m'])
    angular_error=np.arctan2(np.sin(q-reference['q']),np.cos(q-reference['q']))
    summary={'future_frames':int(future.sum()),'nondegenerate_future_frames':int(usable.sum()),
             'diagnostic_geometry_tolerance_m':.01,
             'geometry_consistent_future_frames':int(np.sum(usable&(np.max(np.abs(native_length_error),axis=1)<.01))),
             'minimum_identifiable_link_length_m':.001,
             'keypoint_position_rmse_m':rms(point_error[future]),
             'native_link_length_rmse_m':rms(native_length_error[future]),
             'native_link_length_max_error_m':maximum(native_length_error[future]),
             'angle_rmse_rad':rms(angular_error[usable]),
             'angular_closure_max_m':maximum(quantities['closure_m'][usable]),
             'closure_velocity_max_m_s':maximum(quantities['closure_velocity_m_s'][usable]),
             'closure_acceleration_max_m_s2':maximum(quantities['closure_acceleration_m_s2'][usable]),
             'independent_tangent_force_residual_rmse_Nm':rms(quantities['tangent_force_residual_Nm'][usable]),
             'independent_tangent_force_residual_max_Nm':maximum(quantities['tangent_force_residual_Nm'][usable]),
             'ideal_constraint_energy_work_gap_max_J':None}
    if usable.sum()==future.sum():
        energy=quantities['energy_J'][future];power=quantities['accounted_power_W'][future]
        gap=energy-energy[0]-cumulative_trapezoid(power,times[future],initial=0)
        summary['ideal_constraint_energy_work_gap_max_J']=maximum(gap)
    if not usable.any():
        summary['dynamics_status']='unavailable: collapsed/invalid keypoints prevent identifiable angles and derivatives'
    else:summary['dynamics_status']='computed from raw decoded keypoints and model derivatives; no IK projection or fitted reaction'
    quantities.update(point_error_m=point_error.mean(axis=1),native_length_error_m=np.max(np.abs(native_length_error),axis=1),valid=valid)
    return summary,quantities


def main():
    import cv2
    g=json.loads((OUT/'geometry.json').read_text());c=json.loads((ROOT/'output/fourbar/nominal/physical_metadata.json').read_text())
    inference=json.loads((OUT/'inference_report.json').read_text())
    with np.load(OUT/'prediction.npz') as data:prediction={name:data[name] for name in data.files}
    if prediction['x_recon'].dtype==np.uint8:
        prediction['x_recon']=prediction['x_recon'].astype(np.float32)/255
    with np.load(OUT/'evaluation_reference.npz') as data:reference={name:data[name] for name in data.files}
    with np.load(OUT/'model_input.npz') as data:
        actions=np.column_stack([np.interp(reference['time'],data['action_time'],data['actions'][:,i]) for i in range(3)])
    scale=g['half_extent_m'];center=np.array(g['world_xz_center_m'])
    xy=prediction['keypoints_normalized']*scale+center
    velocity=prediction['keypoint_velocity_normalized_s']*scale
    acceleration=prediction['keypoint_acceleration_normalized_s2']*scale
    result,series=metrics(xy,velocity,acceleration,reference['time'],actions,reference,c)
    oracle,_=metrics(reference['keypoints_world_xz'],reference['keypoint_velocity_m_s'],reference['keypoint_acceleration_m_s2'],
                     reference['time'],actions,reference,c)
    stationary=np.repeat(reference['keypoints_world_xz'][2:3],len(reference['time']),axis=0)
    static,_=metrics(stationary,np.zeros_like(stationary),np.zeros_like(stationary),reference['time'],actions,reference,c)
    assert oracle['native_link_length_max_error_m']<2e-5
    assert oracle['independent_tangent_force_residual_max_Nm']<.002,oracle
    assert static['native_link_length_max_error_m']<2e-5
    assert static['independent_tangent_force_residual_rmse_Nm']>.01,static
    image_mse=np.mean((prediction['x_recon']-reference['x'])**2,axis=(1,2,3))
    result['future_image_mse']=float(np.mean(image_mse[3:]))
    report={'scope':'Functional input/prediction/scoring test; random untrained KeyCLD, not a research accuracy result',
            'training_performed':False,'weights':inference['weights'],'protocol_track':'geometry-and-calibration-given',
            'model_metrics':result,'scorer_checks':{'reference_trajectory':oracle,'stationary_closed_geometry':static,
                                                'checks_passed':True},
            'scoring_notes':['Model observes only three frames. All metrics exclude those observed frames.',
                             'Keypoint order is fixed B,C by supplied constraints; no future-fitted alignment.',
                             'Native point-distance constraints are scored before angle conversion.',
                             'Constraint reaction is eliminated using the independent angular-Jacobian nullspace.',
                             'Energy uses ideal zero-work constraints; reference has small finite-compliance work.',
                             'Energy is a 30 Hz trapezoidal diagnostic. Its reference gap includes sampling error and is not a standalone physics pass/fail metric.',
                             'Reference derivative control interpolates the 1 kHz ground-truth derivatives to video times.',
                             'Null physics metrics mean unidentifiable keypoints, never a perfect score.']}
    report['image_serialization']=inference.get('serialized_image_format','raw floating-point RGB')
    report['dynamics_score_interpretation']='Independent residuals are raw diagnostics; trajectories failing native geometry are not valid assembled-mechanism accuracy results.'
    (OUT/'scores.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (OUT/'metrics_by_frame.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['time_s','is_future','keypoint_error_m','native_length_error_m','identifiable',
                                                  'angular_closure_m','independent_force_residual_Nm','image_mse'])
        for i,t in enumerate(reference['time']):
            finite=lambda value:float(value) if np.isfinite(value) else ''
            writer.writerow([t,i>=3,finite(series['point_error_m'][i]),finite(series['native_length_error_m'][i]),bool(series['valid'][i]),
                             finite(series['closure_m'][i]),finite(series['tangent_force_residual_Nm'][i]),image_mse[i]])
    # Ordinary diagnostic media, with weights provenance visible in the video.
    writer=cv2.VideoWriter(str(OUT/'prediction_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(512,280))
    if not writer.isOpened():raise RuntimeError('Cannot create diagnostic comparison video')
    for i in range(len(reference['time'])):
        left=(reference['x'][i]*255).clip(0,255).astype(np.uint8)
        right=(prediction['x_recon'][i]*255).clip(0,255).astype(np.uint8)
        right=cv2.cvtColor(cv2.resize(right,(256,256)),cv2.COLOR_RGB2BGR)
        for keypoint,color in zip(prediction['keypoints_normalized'][i],[(0,0,255),(255,0,255)]):
            if np.isfinite(keypoint).all():
                x,y=(keypoint[0]+1)/2*255,(1-keypoint[1])/2*255
                if 0<=x<256 and 0<=y<256:cv2.circle(right,(round(x),round(y)),3,color,-1)
        left=cv2.cvtColor(cv2.resize(left,(256,256),interpolation=cv2.INTER_NEAREST),cv2.COLOR_RGB2BGR)
        canvas=np.zeros((280,512,3),dtype=np.uint8);canvas[24:,:256]=left;canvas[24:,256:]=right
        cv2.putText(canvas,'Reference',(8,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        cv2.putText(canvas,'KeyCLD RANDOM / NO TRAINING',(260,16),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
        writer.write(canvas)
        if i==3:cv2.imwrite(str(OUT/'prediction_first_future_frame.png'),canvas)
    writer.release()
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True,constrained_layout=True)
    axes[0].plot(reference['time'][3:],series['point_error_m'][3:]);axes[0].set_ylabel('Keypoint error (m)')
    axes[1].plot(reference['time'][3:],series['native_length_error_m'][3:]);axes[1].set_ylabel('Link-length error (m)')
    axes[2].plot(reference['time'][3:],image_mse[3:]);axes[2].set_ylabel('Image MSE');axes[2].set_xlabel('Time (s)')
    fig.suptitle('KeyCLD random initialization: pipeline diagnostic only')
    fig.savefig(OUT/'diagnostics.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
