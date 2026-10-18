"""Independent SI equations for native planar point predictions, no IK repair."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from experiments.mechanism_corpus import fourbar,slider_crank,double_crank
from experiments.mechanism_corpus.catalog import ROOT,write_json
from .prepare import OUT,point_states,sha

MODULES={'S-A':fourbar,'S-B':slider_crank,'S-J':double_crank}
METRICS=['geometry_m','velocity_m_s','acceleration_m_s2','force_balance_Nm','work_energy_J','prediction_m']


def residuals(x,v,a,c):
    A=np.array([0,c['pivot_height_m']]);D=A+[c['ground_length_m'],0]
    pairs=[(x[:,0]-A,v[:,0],a[:,0],c['lengths_m'][0]),
           (x[:,1]-x[:,0],v[:,1]-v[:,0],a[:,1]-a[:,0],c['lengths_m'][1])]
    if c['category']!='S-B':
        pairs.append((x[:,1]-D,v[:,1],a[:,1],c['lengths_m'][2]))
    rq=[];rv=[];ra=[]
    for r,u,b,length in pairs:
        norm=np.linalg.norm(r,axis=1);safe=np.maximum(norm,1e-12);ru=np.sum(r*u,axis=1)
        rq.append(norm-length);rv.append(ru/safe)
        ra.append((np.sum(u*u+r*b,axis=1))/safe-ru**2/safe**3)
    if c['category']=='S-B':
        rq.append(x[:,1,1]-A[1]-c['guide_offset_m']);rv.append(v[:,1,1]);ra.append(a[:,1,1])
    for value,out in [(x,rq),(v,rv),(a,ra)]:
        mid=value[:,2]-(value[:,0]+value[:,1])/2
        out.extend(mid.T)
    return np.column_stack(rq),np.column_stack(rv),np.column_stack(ra)


def decode(x,v,a,c):
    A=np.array([0,c['pivot_height_m']]);D=A+[c['ground_length_m'],0]
    vectors=[(x[0]-A,v[0],a[0]),(x[1]-x[0],v[1]-v[0],a[1]-a[0])]
    if c['category']!='S-B':
        vectors.append((x[1]-D,v[1],a[1]))
    q=[];qd=[];qdd=[]
    cross=lambda p,r:p[0]*r[1]-p[1]*r[0]
    for r,u,b in vectors:
        square=float(r@r)
        if square<1e-6:
            return None
        omega=cross(r,u)/square
        q.append(np.arctan2(r[1],r[0]));qd.append(omega)
        qdd.append(cross(r,b)/square-2*(r@u)*omega/square)
    if c['category']=='S-B':
        q.append(x[1,0]);qd.append(v[1,0]);qdd.append(a[1,0])
    return np.asarray(q),np.asarray(qd),np.asarray(qdd)


def measure(t,x,v,a,reference,params,target,limits):
    c=params['config'];module=MODULES[c['category']];bodies=params['bodies']
    future=t>=.1-1e-7; anchor=int(np.argmin(abs(t-2/30)))
    result={'metrics':{k:None for k in METRICS},'coverage':0.,'future_samples':int(future.sum()),'failure_reason':None}
    if not all(np.isfinite(y).all() for y in (x,v,a)):
        result['failure_reason']='nonfinite_prediction';result['joint_pass']=False;return result
    rq,rv,ra=residuals(x,v,a,c)
    maximum_rms=lambda value:float(np.sqrt(np.mean(np.square(value,dtype=float),axis=0)).max())
    result['metrics'].update(geometry_m=float(abs(rq[future]).max()),velocity_m_s=maximum_rms(rv[future]),
                             acceleration_m_s2=maximum_rms(ra[future]),prediction_m=maximum_rms((x-reference)[future].reshape(-1,6)))
    legal=np.max(abs(rq),axis=1)<=limits['geometry_m']
    force=[];energy=[];power=[];available=[];equation=[]
    for i in range(len(t)):
        state=decode(x[i],v[i],a[i],c)
        if state is None:
            force.append(np.nan);energy.append(np.nan);power.append(np.nan);available.append(False);continue
        q,qd,qdd=state
        J=module.closure_jacobian(q,c)
        if np.linalg.cond(J[:,1:])>1e10:
            force.append(np.nan);energy.append(np.nan);power.append(np.nan);available.append(False);continue
        N=np.r_[1.,np.linalg.solve(J[:,1:],-J[:,0])]
        M,C,G,T,V=module.mass_bias_energy(q,qd,c,bodies)
        Qf=module.friction(qd,c);Qext=module.external(q,c)
        actuator=c['actuator'];tau=float(np.clip(actuator['speed_gain_Nm_s_rad']*(target-qd[0]),
                                               -actuator['torque_limit_Nm'],actuator['torque_limit_Nm']))
        terms=np.array([N@(M@qdd),N@C,N@G,-N@Qf,-N@Qext,-tau])
        force.append(float(terms.sum()));energy.append(T+V);power.append(tau*qd[0]+Qf@qd+Qext@qd)
        available.append(bool(legal[i]));equation.append(terms)
    force=np.asarray(force);energy=np.asarray(energy);power=np.asarray(power);available=np.asarray(available)
    valid=available[future]; result['coverage']=float(valid.mean())
    if np.isfinite(force[future]).all():
        result['raw_angle_equation_residual_rms_Nm']=float(np.sqrt(np.mean(force[future]**2)))
    else:
        result['raw_angle_equation_residual_rms_Nm']=None
    if available[future].all():
        result['metrics']['force_balance_Nm']=float(np.sqrt(np.mean(force[future]**2)))
    if available[anchor:].all() and np.isfinite(energy[anchor:]).all():
        work=cumulative_trapezoid(power[anchor:],t[anchor:],initial=0)
        imbalance=energy[anchor:]-energy[anchor]-work
        result['metrics']['work_energy_J']=float(abs(imbalance[t[anchor:]>=.1-1e-7]).max())
    if not available[future].all():
        result['failure_reason']='physical_geometry_or_coordinate_coverage_incomplete'
    result['passed']={k:v is not None and np.isfinite(v) and v<=limits[k] for k,v in result['metrics'].items()}
    result['joint_pass']=bool(all(result['passed'].values()))
    if equation:
        result['equation_term_rms_Nm']=dict(zip(['inertia','coriolis','gravity','minus_friction','minus_external','minus_motor'],
                                             np.sqrt(np.mean(np.asarray(equation)**2,axis=0)).tolist()))
    return result


def reference(record):
    p=ROOT/record['physical_parameters'];f=ROOT/record['physics_reference']
    if sha(p)!=record['physical_parameters_sha256'] or sha(f)!=record['physics_reference_sha256']:
        raise RuntimeError('Physical reference changed: '+record['case_id'])
    params=json.loads(p.read_text())
    with np.load(f) as z:
        data={k:z[k] for k in z.files}
    t=np.arange(8,713,dtype=float)/240; indices=np.arange(248,953)
    values=point_states(data['q'][indices],data['qdot'][indices],data['qddot'][indices],params['config'])
    return params,data,t,values


def control_case(task):
    record,limits=task;params,data,t,(x,v,a)=reference(record);target=params['command_speed_rad_s']
    controls={'oracle':(x,v,a)}
    anchor=int(np.argmin(abs(t-2/30)))
    controls['persistence']=(np.broadcast_to(x[anchor],x.shape).copy(),np.zeros_like(v),np.zeros_like(a))
    warped_time=1.+2/30+.8*(t-2/30)
    interp=lambda name:np.column_stack([np.interp(warped_time,data['time'],data[name][:,i]) for i in range(3)])
    controls['time_scaled_0.8']=point_states(interp('q'),.8*interp('qdot'),.64*interp('qddot'),params['config'])
    return {'case_id':record['case_id'],'assembly_id':record['assembly_id'],'category':record['category'],
            'condition':record['condition'],'controls':{k:measure(t,*values,x,params,target,limits) for k,values in controls.items()}}


def calibrate():
    manifest=json.loads((OUT/'dataset_manifest.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    records=[r for r in manifest['records'] if r['split']=='calibration'];limits=protocol['evaluation']['limits_SI']
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(control_case,[(r,limits) for r in records]))
    summaries={k:{'passed':sum(r['controls'][k]['joint_pass'] for r in rows),'total':len(rows),
                  'maxima':{m:max((r['controls'][k]['metrics'][m] for r in rows if r['controls'][k]['metrics'][m] is not None),default=None) for m in METRICS}}
               for k in ('oracle','persistence','time_scaled_0.8')}
    report={'protocol_sha256':sha(OUT/'protocol.json'),'dataset_manifest_sha256':sha(OUT/'dataset_manifest.json'),
            'passed':summaries['oracle']['passed']==len(rows),'model_result':False,
            'summaries':summaries,'rows':rows,'limits_SI':limits,'test_assemblies_used':False}
    write_json(OUT/'calibration.json',report);print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


def score_case(task):
    record,folder,permutation,limits=task;path=folder/'predictions/test'/(record['case_id']+'.npz')
    base={k:record[k] for k in ('case_id','assembly_id','category','condition')}
    if not path.exists():
        return dict(base,metrics={k:None for k in METRICS},passed={k:False for k in METRICS},joint_pass=False,coverage=0.,failure_reason='missing_prediction')
    params,data,t,(ref,_,_)=reference(record)
    with np.load(path) as z:
        assert np.allclose(z['time'],t,atol=2e-7,rtol=0)
        x=z['points'][:,permutation];v=z['velocity'][:,permutation];a=z['acceleration'][:,permutation]
    return dict(base,**measure(t,x,v,a,ref,params,params['command_speed_rad_s'],limits))


def macro(rows,key):
    groups=defaultdict(list)
    for r in rows:
        value=r['joint_pass'] if key=='joint' else r.get('passed',{}).get(key,False)
        groups[(r['category'],r['condition'])].append(float(value))
    return 100*float(np.mean([np.mean(values) for values in groups.values()]))


def score(folder):
    protocol=json.loads((OUT/'protocol.json').read_text());manifest=json.loads((OUT/'dataset_manifest.json').read_text())
    calibration=json.loads((OUT/'calibration.json').read_text())
    if not calibration['passed'] or calibration['protocol_sha256']!=sha(OUT/'protocol.json'):
        raise RuntimeError('Independent calibration did not pass; formal model scores are withheld')
    report=json.loads((folder/'inference_report.json').read_text())
    assert report['dataset_manifest_sha256']==sha(OUT/'dataset_manifest.json')
    perms=list(itertools.permutations(range(3)));errors=[[] for _ in perms]
    for record in manifest['records']:
        if record['split']!='validation':
            continue
        path=folder/'predictions/validation'/(record['case_id']+'.npz')
        with np.load(path) as z:
            kp=z['observed_keypoints']
        with np.load(OUT/record['state_labels']) as z:
            ref=z['x'][:3].astype(float)*.65+[.2,.35]
        for i,p in enumerate(perms):
            value=float(np.mean((kp[:,p]-ref)**2))
            errors[i].append(value if np.isfinite(value) else float('inf'))
    best=int(np.argmin([np.mean(v) for v in errors]));permutation=perms[best]
    records=[r for r in manifest['records'] if r['split']=='test']
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(score_case,[(r,folder,permutation,protocol['evaluation']['limits_SI']) for r in records]))
    components={k:macro(rows,k) for k in METRICS}
    result={'arm':report['arm'],'seed':report['seed'],'test_assemblies':36,'test_trajectories':108,
            'joint_pass_rate_100':macro(rows,'joint'),
            'component_scores':components,'complete_physical_trajectories':sum(r['coverage']==1 for r in rows),
            'reader_permutation_from_validation':list(permutation),
            'validation_keypoint_rmse_m':float(np.sqrt(np.mean(errors[best]))) if np.isfinite(np.mean(errors[best])) else None,
            'checkpoint_sha256':report['checkpoint_sha256'],'protocol_sha256':sha(OUT/'protocol.json'),
            'calibration_sha256':sha(OUT/'calibration.json'),'rows':rows}
    from experiments.closed_chain_eval.score_reports import planar_report
    result = planar_report(result, protocol['evaluation']['limits_SI'], result['calibration_sha256'])
    write_json(folder/'scores.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--calibrate',action='store_true');p.add_argument('--run',type=Path)
    a=p.parse_args()
    if a.calibrate:
        calibrate()
    elif a.run:
        score(a.run)
    else:
        p.error('Provide --calibrate or --run')
