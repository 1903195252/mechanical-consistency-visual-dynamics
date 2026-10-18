"""Expanded AC study: shared cached inputs, independent seeds, calibrated readout.

Training reuses the full published AC implementation and registered objective.
Test forecasting has access only to three prefix frames and public commands.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from experiments.model_expansion import spatial_vjepa as base

ROOT = base.ROOT
OUT = ROOT / 'output/vjepa_expansion'
FEATURES = OUT / 'features'
REFERENCE = OUT / 'reference/features'
ORIGINAL_LABELS = ROOT / 'output/model_expansion/spatial/training_labels'


def records(category=None):
    return [r for r in json.loads((OUT/'dataset_manifest.json').read_text())['videos']
            if (category is None or r['category']==category)
            and (r['split']=='train' or r['view']=='front')]


def deadline():
    from experiments.runtime import check_deadline
    check_deadline()


def configure(seed):
    base.DESTINATION = OUT / ('seed_%d' % seed)
    base.DESTINATION.mkdir(parents=True,exist_ok=True)
    labels = base.DESTINATION/'training_labels'
    if not labels.exists():labels.symlink_to(ORIGINAL_LABELS,target_is_directory=True)
    base.FEATURES=FEATURES
    base.SEED=seed
    base.records=records
    base.deadline=deadline


def cache(category, reference=False):
    """Encode all training frames, or isolated test prefixes and reference pools."""
    sys.path.insert(0,str(base.UPSTREAM))
    from src.hub.backbones import vjepa2_1_vit_base_384
    encoder,unused=vjepa2_1_vit_base_384(pretrained=False)
    saved=torch.load(base.WEIGHTS,map_location='cpu',weights_only=True)
    encoder.load_state_dict(base.clean_weights(saved['ema_encoder']),strict=True)
    del saved,unused;gc.collect()
    encoder.cuda().eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    FEATURES.mkdir(parents=True,exist_ok=True);REFERENCE.mkdir(parents=True,exist_ok=True)
    rows=[r for r in records(category) if (r['split']=='test')==reference]
    start=time.monotonic()
    for i,row in enumerate(rows):
        deadline()
        prefix=FEATURES/(row['video_id']+'.npy')
        target=(REFERENCE if reference else FEATURES)/(row['video_id']+'.npy')
        shape=(120,12288) if reference else (120,base.TOKENS,base.DIM)
        if target.exists() and np.load(target,mmap_mode='r').shape==shape and (not reference or prefix.exists()):continue
        cap=cv2.VideoCapture(str(ROOT/row['video']));frames=[]
        try:
            for _ in range(120):
                ok,frame=cap.read()
                if not ok:raise ValueError('Incomplete video: '+row['video'])
                frames.append(cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(384,384),interpolation=cv2.INTER_CUBIC))
        finally:cap.release()
        temp=target.with_suffix('.tmp.npy')
        dest=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float16,shape=shape)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for first in range(0,120,48):
                x=torch.from_numpy(np.stack(frames[first:first+48])).cuda().permute(0,3,1,2).float()/255
                z=F.layer_norm(encoder(((x-mean)/std).unsqueeze(2).repeat(1,1,2,1,1)),(base.DIM,))
                if not torch.isfinite(z).all():raise FloatingPointError('Nonfinite encoder output')
                if reference and first==0:np.save(prefix,z[:3].float().cpu().numpy().astype(np.float16))
                values=base.pooled(z.float()) if reference else z
                dest[first:first+len(x)]=values.float().cpu().numpy().astype(np.float16)
        dest.flush();del dest;temp.replace(target)
        if (i+1)%5==0 or i+1==len(rows):
            print(json.dumps(dict(stage='reference_cache' if reference else 'train_cache',family=category,done=i+1,total=len(rows),seconds=time.monotonic()-start)),flush=True)
    del encoder;gc.collect();torch.cuda.empty_cache()


def infer(category, batch_size=12, limit=None):
    sys.path.insert(0,str(base.UPSTREAM))
    from src.models.ac_predictor import vit_ac_predictor
    saved=base.load_checkpoint(base.DESTINATION/(category+'_predictor.pt'))
    assert saved['epoch']==saved['registered_epochs']
    readout=base.load_checkpoint(base.DESTINATION/(category+'_readout.pt'));m=readout['metadata']
    predictor=vit_ac_predictor(img_size=(384,384),patch_size=16,num_frames=16,tubelet_size=2,
        embed_dim=base.DIM,predictor_embed_dim=1024,depth=24,num_heads=16,
        action_embed_dim=len(m['amean']),is_frame_causal=True,use_rope=True,use_extrinsics=False).cuda().eval()
    predictor.load_state_dict(saved['predictor'],strict=True)
    predictor.attn_mask=predictor.attn_mask.cuda()
    head=base.Readout(len(m['ymean'])).cuda().eval();head.load_state_dict(readout['readout'],strict=True)
    del saved,readout;gc.collect()
    ym,ys,sm,ss=[torch.tensor(m[k],device='cuda') for k in ('ymean','ystd','smean','sstd')]
    qi,vi,nq=m['qi'],m['vi'],int(m['nq'])
    native=base.DESTINATION/'native/predictions'/category
    pool_dir=base.DESTINATION/'forecast_features'
    native.mkdir(parents=True,exist_ok=True);pool_dir.mkdir(parents=True,exist_ok=True)
    rows=[r for r in records(category) if r['split']=='test']
    if limit:rows=rows[:limit]
    rows=[r for r in rows if not (pool_dir/(r['video_id']+'.npy')).exists()]
    start=time.monotonic()
    for first in range(0,len(rows),batch_size):
        deadline();chunk=rows[first:first+batch_size]
        history=torch.tensor(np.stack([np.load(FEATURES/(r['video_id']+'.npy')) for r in chunk]),device='cuda')
        assert history.shape[1]==3
        actions=torch.tensor((np.stack([base.inputs(r)[0] for r in chunk])-m['amean'])/m['astd'],device='cuda')
        cams=torch.tensor((np.stack([base.camera(r) for r in chunk])-m['cmean'])/m['cstd'],device='cuda')
        states=[];outputs=[];pools=[]
        def decode(feature):
            h=base.pooled(feature.float());y=head(h,cams).float()*ys+ym
            y[:,3:7]=F.normalize(y[:,3:7],dim=-1)
            states.append((torch.cat([y[:,qi],y[:,nq+vi]],dim=-1)-sm)/ss)
            outputs.append(y[:,:nq].cpu().numpy());pools.append(h.cpu().numpy().astype(np.float16))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for k in range(3):decode(history[:,k])
            for k in range(3,120):
                deadline();n=min(7,history.shape[1])
                z=F.layer_norm(predictor(history[:,-n:].flatten(1,2),actions[:,k-n:k],torch.stack(states[-n:],dim=1)),(base.DIM,))[:,-base.TOKENS:]
                decode(z);history=torch.cat([history[:,-6:],z[:,None]],dim=1)
        q=np.stack(outputs,axis=1);h=np.stack(pools,axis=1)
        for row,values,features in zip(chunk,q,h):
            np.savez_compressed(native/(row['video_id']+'.npz'),time=np.arange(120)/30,qpos=values)
            np.save(pool_dir/(row['video_id']+'.npy'),features)
        print(json.dumps(dict(stage='forecast',family=category,seed=base.SEED,done=first+len(chunk),total=len(rows),seconds=time.monotonic()-start,batch=batch_size)),flush=True)
    del predictor,head;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['cache','reference','train','infer'])
    p.add_argument('--family',choices=['H-A','H-B'],required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch',type=int,default=12);p.add_argument('--limit',type=int)
    args=p.parse_args()
    available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[:8])
    torch.set_num_threads(8);cv2.setNumThreads(1);configure(args.seed)
    if args.stage in ('cache','reference'):cache(args.family,args.stage=='reference')
    elif args.stage=='train':base.train(args.family)
    else:infer(args.family,args.batch,args.limit)


if __name__=='__main__':main()
