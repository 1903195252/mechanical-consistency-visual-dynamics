"""Reuse calibrated, independent SI mechanics without importing the CON runtime."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import cv2
from experiments.keycld_corpus.physics_score import METRICS, macro, reference, measure
from experiments.mechanism_corpus.catalog import ROOT, write_json

OUT=ROOT/'output/con_corpus'
DATA=ROOT/'output/keycld_corpus'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def future_images(record):
    video=ROOT/record['source_video']
    assert sha(video)==record['source_video_sha256']
    public=json.loads(video.with_suffix('.inputs.json').read_text())
    camera=public['observed_camera_frames'][0]
    R=np.asarray(camera['R_world_to_camera']); offset=np.asarray(camera['t_world_to_camera_m'])
    K=np.asarray(camera['K_affine_pixels'])
    projection=K@np.array([[R[0,0],R[0,2],offset[0]],[R[1,0],R[1,2],offset[1]],[0,0,1.]])
    pixel=np.array([[1.3/63,0,.2-.65],[0,-1.3/63,.35+.65],[0,0,1.]])
    transform=projection@pixel; cap=cv2.VideoCapture(str(video)); images=[]; index=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:
                break
            if index>=33:
                frame=cv2.warpPerspective(frame,transform,(64,64),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
                                          borderMode=cv2.BORDER_CONSTANT,borderValue=(128,128,128))
                images.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            index+=1
    finally:
        cap.release()
    assert index==120 and len(images)==87
    return np.asarray(images,dtype=float)/255


def case(task):
    r,folder,limits=task
    base={k:r[k] for k in ('case_id','assembly_id','category','condition')}
    path=folder/'predictions/test'/(r['case_id']+'.npz')
    if not path.exists():
        return dict(base,metrics={k:None for k in METRICS},passed={k:False for k in METRICS},
                    joint_pass=False,coverage=0.,failure_reason='missing_prediction')
    meta=json.loads(path.with_suffix('.json').read_text())
    assert sha(path)==meta['prediction_sha256']
    params,truth,t,(points,_,_)=reference(r)
    with np.load(path) as z:
        assert np.allclose(z['time'],t,atol=2e-7,rtol=0)
        result=measure(t,z['points'],z['velocity'],z['acceleration'],points,params,
                       params['command_speed_rad_s'],limits)
        if 'rgb_prediction' in z and not meta['finite']:
            result['rgb_mse']=None
            result['rgb_metric_failure']='nonfinite_model_state'
        elif 'rgb_prediction' in z:
            # Pixel targets are loaded only in the evaluator, after frozen forecasts.
            with np.load(DATA/r['observations']) as obs:
                prefix=obs['x'][2].astype(float)/255
            target=future_images(r)
            predicted=z['rgb_prediction'].astype(float)/255
            assert predicted.shape==target.shape
            result['rgb_mse']=float(np.mean((predicted-target)**2))
            result['rgb_persistence_mse']=float(np.mean((prefix-target)**2))
    return dict(base,**result)


def score(folder):
    protocol=json.loads((OUT/'protocol.json').read_text())
    manifest=json.loads((OUT/'dataset_manifest.json').read_text())
    calibration=json.loads((OUT/'calibration.json').read_text())
    assert calibration['passed'] and calibration['dataset_manifest_sha256']==sha(OUT/'dataset_manifest.json')
    assert protocol['evaluation']['limits_SI']==calibration['limits_SI']
    inference=json.loads((folder/'inference_report.json').read_text())
    tr=json.loads((folder/'training_report.json').read_text())
    assert inference['checkpoint_sha256']==tr['checkpoint_sha256']
    records=[r for r in manifest['records'] if r['split']=='test']
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(case,[(r,folder,protocol['evaluation']['limits_SI']) for r in records]))
    components={k:macro(rows,k) for k in METRICS}
    result={'arm':inference['arm'],'seed':inference['seed'],'test_assemblies':36,'test_trajectories':108,
            'joint_pass_rate_100':macro(rows,'joint'),
            'component_scores':components,'complete_physical_trajectories':sum(r['coverage']==1 for r in rows),
            'checkpoint_sha256':tr['checkpoint_sha256'],'protocol_sha256':sha(OUT/'protocol.json'),
            'calibration_sha256':sha(OUT/'calibration.json'),
            'track':'privileged_state' if tr['privileged_state'] else 'privileged_affine_latent_readout',
            'public_video_reader_score':False,'rows':rows}
    from experiments.closed_chain_eval.score_reports import planar_report
    result = planar_report(result, protocol['evaluation']['limits_SI'], result['calibration_sha256'])
    write_json(folder/'scores.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    score(parser.parse_args().run)
