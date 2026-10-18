"""Frozen V-JEPA+AC replay with native feedback and independent readout controls.

Forecasting opens three observed frames and public commands. The original head
supplies recurrent state feedback; external heads are applied after forecasting.
Reference-video features enter only the subsequent measurement stage.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from experiments.model_expansion.spatial_vjepa import (
    ROOT, DESTINATION, UPSTREAM, WEIGHTS, DIM, TOKENS, Readout, records,
    pooled, camera, inputs, clean_weights, load_checkpoint)
from .geometry_spatial_reader import GeometryHead, nominal, DIRECTORY as HEADS
from .refine_spatial_reader import DIRECTORY as REFERENCE_FEATURES
from .visual_calibration import OUT, write, sha
from .calibrate_vjepa_readout import DIRECTORY as RELATIVE
from experiments.lingbot_expansion.delta_instrument import decode as relative_decode

DIRECTORY=OUT/'vjepa_readout_replay'

def readers(family):
    return ('native','cad','relative') if family=='H-B' else ('native','cad')


def selected(limit=None):
    return [r for f in ('H-A','H-B') for r in
            [x for x in records(f) if x['split']=='test'][:limit]]


def prefix_cache(rows):
    missing=[r for r in rows if not (DIRECTORY/'prefix'/(r['video_id']+'.npy')).exists()]
    if not missing:return
    sys.path.insert(0,str(UPSTREAM))
    from src.hub.backbones import vjepa2_1_vit_base_384
    encoder, unused=vjepa2_1_vit_base_384(pretrained=False)
    saved=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    encoder.load_state_dict(clean_weights(saved['ema_encoder']),strict=True)
    del saved,unused
    gc.collect()
    encoder.cuda().eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    for row in missing:
        cap=cv2.VideoCapture(str(ROOT/row['video']));images=[]
        try:
            for _ in range(3):
                ok,frame=cap.read()
                if not ok:raise ValueError('Incomplete observed prefix')
                images.append(cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(384,384),interpolation=cv2.INTER_CUBIC))
        finally:cap.release()
        x=torch.from_numpy(np.stack(images)).cuda().permute(0,3,1,2).float()/255
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            z=F.layer_norm(encoder(((x-mean)/std).unsqueeze(2).repeat(1,1,2,1,1)),(DIM,))
        path=DIRECTORY/'prefix'/(row['video_id']+'.npy');path.parent.mkdir(parents=True,exist_ok=True)
        np.save(path,z.float().cpu().numpy().astype(np.float16))
    del encoder,x,z
    gc.collect();torch.cuda.empty_cache()


def forecast(rows):
    sys.path.insert(0,str(UPSTREAM))
    from src.models.ac_predictor import vit_ac_predictor
    prefix_cache(rows)
    for family in ('H-A','H-B'):
        todo=[r for r in rows if r['category']==family and not (DIRECTORY/'forecast/features'/(r['video_id']+'.npy')).exists()]
        if not todo:continue
        readout=load_checkpoint(DESTINATION/(family+'_readout.pt'));m=readout['metadata']
        saved=load_checkpoint(DESTINATION/(family+'_inference_weights.pt'))
        assert saved['epoch']==saved['registered_epochs']
        head=Readout(len(m['ymean'])).cuda().eval();head.load_state_dict(readout['readout'],strict=True)
        predictor=vit_ac_predictor(img_size=(384,384),patch_size=16,num_frames=16,tubelet_size=2,
            embed_dim=DIM,predictor_embed_dim=1024,depth=24,num_heads=16,action_embed_dim=len(m['amean']),
            is_frame_causal=True,use_rope=True,use_extrinsics=False).cuda().eval()
        predictor.load_state_dict(saved['predictor'],strict=True)
        del saved,readout;gc.collect()
        ym,ys,sm,ss=[torch.tensor(m[k],device='cuda') for k in ('ymean','ystd','smean','sstd')]
        qi,vi,nq=m['qi'],m['vi'],int(m['nq'])
        for index,row in enumerate(todo):
            start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            history=torch.tensor(np.load(DIRECTORY/'prefix'/(row['video_id']+'.npy'))[None],device='cuda')
            actions=torch.tensor((inputs(row)[0]-m['amean'])/m['astd'],device='cuda')[None]
            cams=torch.tensor((camera(row)-m['cmean'])/m['cstd'],device='cuda')[None]
            states=[];outputs=[];features=[]
            def decode(feature):
                h=pooled(feature.float())
                y=head(h,cams).float()*ys+ym
                y[:,3:7]=F.normalize(y[:,3:7],dim=-1)
                state=(torch.cat([y[:,qi],y[:,nq+vi]],dim=-1)-sm)/ss
                features.append(h[0].cpu().numpy().astype(np.float16))
                outputs.append(y[0].cpu().numpy());states.append(state)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for k in range(3):decode(history[:,k])
                for k in range(3,120):
                    n=min(7,history.shape[1])
                    z=F.layer_norm(predictor(history[:,-n:].flatten(1,2),actions[:,k-n:k],
                                            torch.stack(states[-n:],dim=1)),(DIM,))[:,-TOKENS:]
                    decode(z)
                    history=torch.cat([history[:,-6:],z[:,None]],dim=1)
            q=np.stack(outputs)[:,:nq];path=DIRECTORY/'forecast/native/predictions'/family/(row['video_id']+'.npz')
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,time=np.arange(120)/30,qpos=q)
            original=DESTINATION/'predictions'/family/path.name
            with np.load(original) as d:old=d['qpos']
            report=dict(video_id=row['video_id'],seconds=time.monotonic()-start,
                torch_version=torch.__version__,device=torch.cuda.get_device_name(),
                numpy_version=np.__version__,autocast='bfloat16',batch_size=1,
                peak_memory_MiB=torch.cuda.max_memory_allocated()/2**20,
                archived_qpos_max_abs_difference=float(np.max(np.abs(q-old))),
                archived_platform_RMS_difference_mm=float(1000*np.sqrt(np.mean(np.sum((q[3:,:3]-old[3:,:3])**2,axis=1)))))
            write(DIRECTORY/'replay_checks'/(row['video_id']+'.json'),report)
            path=DIRECTORY/'forecast/features'/(row['video_id']+'.npy');path.parent.mkdir(parents=True,exist_ok=True)
            np.save(path,np.stack(features))
            print(json.dumps(dict(stage='forecast',family=family,case=index+1,total=len(todo),**report)),flush=True)
        del predictor,head,history,states,outputs,features,z
        gc.collect();torch.cuda.empty_cache()


def decode_readouts(rows,modes=('reference','forecast')):
    for family in ('H-A','H-B'):
        saved=load_checkpoint(DESTINATION/(family+'_readout.pt'));m=saved['metadata']
        native=Readout(len(m['ymean'])).cuda().eval();native.load_state_dict(saved['readout'],strict=True)
        saved=load_checkpoint(HEADS/(family+'_head.pt'));c=saved['metadata']
        calibrated=GeometryHead(c['nq']).cuda().eval();calibrated.load_state_dict(saved['model'],strict=True)
        for row in [r for r in rows if r['category']==family]:
            q0=nominal(row)
            context=torch.tensor((np.r_[camera(row),q0]-c['cmean'])/c['cstd'],device='cuda')
            cams=torch.tensor((camera(row)-m['cmean'])/m['cstd'],device='cuda')
            for mode in modes:
                path=(REFERENCE_FEATURES/'features/384' if mode=='reference' else DIRECTORY/'forecast/features')/(row['video_id']+'.npy')
                h=torch.from_numpy(np.load(path).astype(np.float32)).cuda()
                with torch.inference_mode():
                    q=calibrated(h,context.expand(len(h),-1)).cpu().numpy()*c['ystd']+c['ymean']+q0
                q[:,3:7]/=np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
                target=DIRECTORY/mode/'cad/predictions'/family/(row['video_id']+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target,time=np.arange(120)/30,qpos=q)
                if family=='H-B':
                    with np.load(RELATIVE/'H-B_relative_head.npz') as d:
                        assert int(d['window'])==1
                        q=relative_decode(h.cpu().numpy(),q0,d['matrix'],1)
                    target=DIRECTORY/mode/'relative/predictions'/family/(row['video_id']+'.npz')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(target,time=np.arange(120)/30,qpos=q)
                if mode=='reference':
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        q=native(h,cams.expand(len(h),-1)).float().cpu().numpy()*m['ystd']+m['ymean']
                    q=q[:,:int(m['nq'])];q[:,3:7]/=np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
                    target=DIRECTORY/mode/'native/predictions'/family/(row['video_id']+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(target,time=np.arange(120)/30,qpos=q)
        del native,calibrated;gc.collect();torch.cuda.empty_cache()


def score_one(task):
    from experiments.model_expansion.evaluate_spatial import evaluate_case
    row,mode,reader=task
    result=evaluate_case((row,'score',DIRECTORY/mode/reader))
    write(DIRECTORY/mode/reader/'cases'/(row['video_id']+'.json'),result)
    return result


def run(stage,limit,family=None,mode=None):
    rows=[r for r in selected(limit) if family is None or r['category']==family]
    modes=(mode,) if mode else ('reference','forecast')
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    if stage in ('all','forecast'):forecast(rows)
    if stage=='forecast':return
    if stage in ('all','read'):decode_readouts(rows,modes)
    if stage in ('all','score'):
        tasks=[(r,m,h) for r in rows for m in modes for h in readers(r['category'])]
        pending=[t for t in tasks if not (DIRECTORY/t[1]/t[2]/'cases'/(t[0]['video_id']+'.json')).exists()]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for i,result in enumerate(pool.map(score_one,pending)):
                print(json.dumps(dict(stage='score',completed=i+1,total=len(pending),video_id=result['video_id'])),flush=True)
        for mode in modes:
            for family in ('H-A','H-B'):
                for head in readers(family):
                    group=[json.loads((DIRECTORY/mode/head/'cases'/(r['video_id']+'.json')).read_text()) for r in rows if r['category']==family]
                    write(DIRECTORY/mode/head/(family+'_score.json'),group)
    sources=[WEIGHTS]+[DESTINATION/(f+s) for f in ('H-A','H-B') for s in ('_readout.pt','_inference_weights.pt')]+[HEADS/(f+'_head.pt') for f in ('H-A','H-B')]+[RELATIVE/'H-B_relative_head.npz']
    write(DIRECTORY/'protocol.json',dict(cases=[r['video_id'] for r in rows],training_updates=0,
        predictor_feedback='Original physical readout only; independent readout never fed back',
        compared_readouts={'H-A':list(readers('H-A')),'H-B':list(readers('H-B'))},reference_features_resolution=384,
        independent_readout='CAD-coordinate head; Delta also has reset-relative regression fitted at 384 pixels',
        calibration_selection_sha256=sha(RELATIVE/'selection.json'),
        measurement='Identical CAD/IK and 30 Hz differentiation for reference and forecast',
        source_sha256=sha(__file__),checkpoint_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','forecast','read','score'],default='all')
    p.add_argument('--limit',type=int)
    p.add_argument('--family',choices=['H-A','H-B'])
    p.add_argument('--mode',choices=['reference','forecast'])
    a=p.parse_args();run(a.stage,a.limit,a.family,a.mode)
