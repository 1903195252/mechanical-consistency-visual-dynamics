"""Apply fixed physical readouts and audit every registered spatial forecast."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path

for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'

import numpy as np
import torch

from .run import ROOT, OUT, REFERENCE, records, configure
from experiments.model_expansion import spatial_vjepa as base
from experiments.closed_chain_eval.geometry_spatial_reader import GeometryHead, nominal, DIRECTORY as HEADS
from experiments.closed_chain_eval.calibrate_vjepa_readout import DIRECTORY as RELATIVE
from experiments.lingbot_expansion.delta_instrument import decode as relative_decode


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)


def decode(rows,seed):
    family=rows[0]['category'];run=OUT/('seed_%d'%seed)
    if family=='H-A':
        saved=base.load_checkpoint(HEADS/'H-A_head.pt');m=saved['metadata']
        head=GeometryHead(m['nq']).cuda().eval();head.load_state_dict(saved['model'],strict=True)
    else:
        with np.load(RELATIVE/'H-B_relative_head.npz') as data:
            assert int(data['window'])==1
            matrix=data['matrix'].copy()
    for mode in ('reference','forecast'):
        source=REFERENCE if mode=='reference' else run/'forecast_features'
        destination=(OUT/'reference' if mode=='reference' else run)/'calibrated/predictions'/family
        destination.mkdir(parents=True,exist_ok=True)
        for row in rows:
            target=destination/(row['video_id']+'.npz')
            if target.exists():continue
            q0=nominal(row);h=np.load(source/(row['video_id']+'.npy')).astype(np.float32)
            if family=='H-A':
                context=torch.tensor((np.r_[base.camera(row),q0]-m['cmean'])/m['cstd'],device='cuda')
                with torch.inference_mode():
                    q=head(torch.from_numpy(h).cuda(),context.expand(len(h),-1)).cpu().numpy()*m['ystd']+m['ymean']+q0
                q[:,3:7]/=np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
            else:q=relative_decode(h,q0,matrix,1)
            np.savez_compressed(target,time=np.arange(120)/30,qpos=q,observed_frames=3)
    if family=='H-A':del head
    saved=base.load_checkpoint(run/(family+'_readout.pt'));m=saved['metadata']
    head=base.Readout(len(m['ymean'])).cuda().eval();head.load_state_dict(saved['readout'],strict=True)
    target=run/'reference_internal/predictions'/family;target.mkdir(parents=True,exist_ok=True)
    for row in rows:
        path=target/(row['video_id']+'.npz')
        if path.exists():continue
        h=torch.from_numpy(np.load(REFERENCE/(row['video_id']+'.npy')).astype(np.float32)).cuda()
        c=torch.tensor((base.camera(row)-m['cmean'])/m['cstd'],device='cuda').expand(len(h),-1)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            y=head(h,c).float().cpu().numpy()*m['ystd']+m['ymean']
        q=y[:,:int(m['nq'])];q[:,3:7]/=np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
        np.savez_compressed(path,time=np.arange(120)/30,qpos=q)
    del head
    gc.collect();torch.cuda.empty_cache()


def score(task):
    from experiments.model_expansion.evaluate_spatial import evaluate_case
    row,destination,mode=task
    path=Path(destination)/'cases'/(row['video_id']+'.json')
    if path.exists():return str(path)
    result=evaluate_case((row,mode,Path(destination)))
    write(path,result)
    return str(path)


def run(family,seed,workers):
    available=sorted(os.sched_getaffinity(0))
    if len(available)>=20:os.sched_setaffinity(0,available[8:20])
    torch.set_num_threads(2);configure(seed)
    rows=[r for r in records(family) if r['split']=='test']
    assert len(rows)==150 and len({r['assembly_id'] for r in rows})==50
    decode(rows,seed)
    paths=[(OUT/('seed_%d'%seed)/reader,'score') for reader in ('native','calibrated','reference_internal')]
    paths.extend([(OUT/'reference/calibrated','score'),(OUT/'reference/controls','calibrate')])
    tasks=[(r,str(p),mode) for p,mode in paths for r in rows if not (p/'cases'/(r['video_id']+'.json')).exists()]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(score,t) for t in tasks]
        for i,future in enumerate(as_completed(futures)):
            future.result()
            if (i+1)%15==0 or i+1==len(tasks):print(json.dumps(dict(stage='measure',family=family,seed=seed,done=i+1,total=len(tasks))),flush=True)
    for destination,mode in paths:
        measurements=[json.loads((destination/'cases'/(r['video_id']+'.json')).read_text()) for r in rows]
        write(destination/(family+('_calibrate.json' if mode=='calibrate' else '_score.json')),measurements)
    write(OUT/('seed_%d'%seed)/(family+'_measurement_complete.json'),dict(seed=seed,family=family,cases=150,readers=['internal feedback','independent calibrated'],reference_cases=150))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',required=True,choices=['H-A','H-B']);p.add_argument('--seed',required=True,type=int);p.add_argument('--workers',default=12,type=int)
    a=p.parse_args();run(a.family,a.seed,a.workers)
