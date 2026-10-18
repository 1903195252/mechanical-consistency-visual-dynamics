"""Independent calibration of the added geometric map; no learned model input."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import tarfile
import numpy as np
from experiments.keycld_corpus.physics_score import reference,measure,control_case,METRICS
from experiments.keycld_corpus.prepare import point_states,sha
from experiments.mechanism_corpus.catalog import ROOT,write_json
from .crank_readout import reconstruct
OUT=ROOT/'output/con_corpus';EXT=OUT/'dynamics_extension'

def case(r):
    params,data,t,(x,v,a)=reference(r);c=params['config'];target=params['command_speed_rad_s']
    limits=json.loads((OUT/'protocol.json').read_text())['evaluation']['limits_SI']
    anchor=int(np.argmin(abs(t-2/30)));warped_time=1+2/30+.8*(t-2/30)
    interp=lambda name:np.column_stack([np.interp(warped_time,data['time'],data[name][:,i]) for i in range(3)])
    values={'oracle':(x,v,a),'persistence':(np.broadcast_to(x[anchor],x.shape).copy(),np.zeros_like(v),np.zeros_like(a)),
            'time_scaled_0.8':point_states(interp('q'),.8*interp('qdot'),.64*interp('qddot'),c)}
    controls={}
    for name,original in values.items():
        mapped,state=reconstruct(*original,c)
        result=measure(t,*mapped,x,params,target,limits)
        result['correction_rms_m']=float(np.sqrt(np.mean((mapped[0]-original[0])**2)))
        controls[name]=result
    return dict(case_id=r['case_id'],category=r['category'],condition=r['condition'],controls=controls)

def main():
    manifest=json.loads((OUT/'dataset_manifest.json').read_text());records=[r for r in manifest['records'] if r['split']=='calibration']
    with ProcessPoolExecutor(max_workers=8) as pool:rows=list(pool.map(case,records))
    summaries={k:dict(passed=sum(r['controls'][k]['joint_pass'] for r in rows),total=len(rows),
                     maxima={m:max(r['controls'][k]['metrics'][m] for r in rows) for m in METRICS},
                     correction_max_rms_m=max(r['controls'][k]['correction_rms_m'] for r in rows))
               for k in ['oracle','persistence','time_scaled_0.8']}
    passed=summaries['oracle']['passed']==len(rows) and all(summaries[k]['passed']==0 for k in ['persistence','time_scaled_0.8'])
    report=dict(passed=passed,model_result=False,test_used=False,protocol_sha256=sha(EXT/'protocol.json'),
                readout_source_sha256=sha(Path(__file__).with_name('crank_readout.py')),dataset_manifest_sha256=sha(OUT/'dataset_manifest.json'),
                summaries=summaries,rows=rows)
    write_json(EXT/'readout_calibration.json',report);print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)
    assert passed,'Additional readout calibration failed; do not publish downstream metrics'
if __name__=='__main__':main()
