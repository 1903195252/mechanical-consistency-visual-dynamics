"""Raw and added-readout SI evidence, with every frozen test case retained."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from experiments.mechanism_corpus.catalog import ROOT,write_json
from experiments.keycld_corpus.physics_score import measure,METRICS,macro
from .crank_readout import reconstruct
OUT=ROOT/'output/con_corpus';DATA=ROOT/'output/keycld_corpus';EXT=OUT/'dynamics_extension'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def aggregate(rows,field):
    groups=defaultdict(list)
    for r in rows:
        v=r.get(field)
        if v is not None and np.isfinite(v):groups[(r['category'],r['condition'])].append(v)
    return dict(valid=sum(map(len,groups.values())),total=len(rows),macro_mean=float(np.mean([np.mean(v) for v in groups.values()])) if groups else None)

def case(task):
    record,ref,folder,permutation,limits,destination,identity=task
    output=destination/(record['case_id']+'.json')
    if output.exists():
        row=json.loads(output.read_text());assert row['identity']==identity;return row
    base={k:record[k] for k in ['case_id','assembly_id','category','condition']}
    path=folder/'predictions/test'/(record['case_id']+'.npz')
    meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['prediction_sha256']
    reference_path=ROOT/ref['reference_npz'];params_path=ROOT/ref['parameters_json']
    assert sha(reference_path)==ref['reference_sha256'] and sha(params_path)==ref['parameters_sha256']
    params=json.loads(params_path.read_text());c=params['config']
    with np.load(reference_path) as z:t=z['time'];truth=z['points']
    with np.load(path) as z:
        assert np.allclose(z['time'],t,atol=2e-7,rtol=0)
        x,v,a=[z[k][:,permutation].copy() for k in ['points','velocity','acceleration']]
    raw=measure(t,x,v,a,truth,params,params['command_speed_rad_s'],limits)
    row=dict(base,identity=identity,native_prediction_sha256=meta['prediction_sha256'],raw=raw,
             geometry_conditioned=True,raw_prediction_changed=False,correction_rms_m=None,correction_max_point_m=None)
    try:
        values,state=reconstruct(x,v,a,c)
        conditioned=measure(t,*values,truth,params,params['command_speed_rad_s'],limits)
        row['correction_rms_m']=float(np.sqrt(np.mean((values[0]-x)**2)))
        row['correction_max_point_m']=float(np.linalg.norm(values[0]-x,axis=-1).max())
        row['crank_angular_velocity_rms_rad_s']=float(np.sqrt(np.mean(state['qdot'][:,0]**2)))
        # Raw states and geometry are the sole inputs; evaluator parameters enter only measure().
        output_npz=output.with_suffix('.npz');tmp=output_npz.with_suffix('.npz.partial')
        with tmp.open('wb') as handle:np.savez_compressed(handle,time=t,points=values[0],velocity=values[1],acceleration=values[2],**state)
        tmp.replace(output_npz);row['readout_prediction_sha256']=sha(output_npz)
    except (ValueError,np.linalg.LinAlgError) as exc:
        conditioned=dict(metrics={k:None for k in METRICS},passed={k:False for k in METRICS},coverage=0.,joint_pass=False,failure_reason=str(exc))
    row['conditioned']=conditioned;write_json(output,row);return row

def main():
    started=time.monotonic();protocol=json.loads((EXT/'protocol.json').read_text());calibration=json.loads((EXT/'readout_calibration.json').read_text())
    assert calibration['passed'] and calibration['protocol_sha256']==sha(EXT/'protocol.json')
    assert calibration['readout_source_sha256']==sha(Path(__file__).with_name('crank_readout.py'))
    manifest=json.loads((OUT/'dataset_manifest.json').read_text());records=[r for r in manifest['records'] if r['split']=='test']
    assert sha(OUT/'dataset_manifest.json')==protocol['dataset_manifest_sha256']
    pack=json.loads((OUT/'evaluator/manifest.json').read_text());refs={r['case_id']:r for r in pack['references']}
    for relative,digest in pack['source_hashes'].items():
        assert sha(ROOT/relative)==digest, 'Independent evaluator changed: '+relative
    limits=json.loads((OUT/'protocol.json').read_text())['evaluation']['limits_SI'];run_list=[];tasks=[]
    for model,root,arms in [('keycld',DATA,['video_euler','state_euler','state_rk4']),('con',OUT,['video_con_iae','state_con_iae']),
                            ('con_integrator',EXT,['state_euler_con_iae','state_rk4_con_iae'])]:
        for arm in arms:
            for seed in [42,43,44]:
                folder=root/'runs'/f'{arm}_seed{seed}'
                if not (folder/'inference_report.json').exists():raise RuntimeError('Missing completed inference: '+str(folder))
                tr=json.loads((folder/'training_report.json').read_text());assert sha(folder/'params.p')==tr['checkpoint_sha256']
                permutation=json.loads((folder/'scores.json').read_text())['reader_permutation_from_validation'] if model=='keycld' else [0,1,2]
                identity=dict(model=model,arm=arm,seed=seed,checkpoint_sha256=tr['checkpoint_sha256'],protocol_sha256=sha(EXT/'protocol.json'),
                              source_sha256=sha(Path(__file__)),readout_source_sha256=sha(Path(__file__).with_name('crank_readout.py')),
                              evaluator_manifest_sha256=sha(OUT/'evaluator/manifest.json'),permutation=permutation)
                destination=EXT/'physics_readout'/f'{model}_{arm}_seed{seed}';destination.mkdir(parents=True,exist_ok=True)
                run_list.append((destination,identity))
                tasks.extend((r,refs[r['case_id']],folder,permutation,limits,destination,identity) for r in records)
    write_json(EXT/'physics_status.json',dict(stage='scoring',completed=0,total=len(tasks)))
    with ProcessPoolExecutor(max_workers=24) as pool:
        futures=[pool.submit(case,t) for t in tasks]
        for i,f in enumerate(as_completed(futures)):
            f.result()
            if i%24==0:
                write_json(EXT/'physics_status.json',dict(stage='scoring',completed=i+1,total=len(tasks),elapsed_s=time.monotonic()-started))
    index=[]
    for destination,identity in run_list:
        rows=[json.loads((destination/(r['case_id']+'.json')).read_text()) for r in records]
        summaries={}
        for track in ['raw','conditioned']:
            values=[dict({k:r[k] for k in ['category','condition']},**r[track]) for r in rows]
            metric_values=[dict({k:r[k] for k in ['category','condition']},**r[track]['metrics']) for r in rows]
            summaries[track]=dict(metrics={k:aggregate(metric_values,k) for k in METRICS},
                                  component_pass_100={k:macro(values,k) for k in METRICS},joint_pass_100=macro(values,'joint'),
                                  complete_physical_trajectories=sum(r['coverage']==1 for r in values),total=len(rows))
        summaries['correction_rms_m']=aggregate(rows,'correction_rms_m');summaries['correction_max_point_m']=aggregate(rows,'correction_max_point_m')
        report=dict(identity=identity,stage='complete',test_cases=len(rows),summary=summaries,rows=rows,
                    interpretation='Conditioned trajectories are an explicit geometry-readout intervention. Raw force/energy nulls and scores are preserved.')
        write_json(destination/'report.json',report);index.append(dict(identity=identity,summary=summaries))
    write_json(EXT/'physics_report.json',dict(stage='complete',runs=index,test_cases_per_run=108,runs_completed=len(index),
                                            elapsed_s=time.monotonic()-started,calibration_sha256=sha(EXT/'readout_calibration.json')))
    write_json(EXT/'physics_status.json',dict(stage='complete',completed=len(tasks),total=len(tasks)))
    print(json.dumps(dict(stage='complete',runs=len(index),cases=len(tasks))),flush=True)
if __name__=='__main__':main()
