"""Replay saved Delta reference/generated RGB through one frozen instrument.

Forecast pixels and forecasting checkpoints are unchanged. This is offline
measurement: reset-relative visual depth plus calibrated image-plane motion,
followed by the existing unconstrained and explicit CAD-reconstruction audits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import sys
import numpy as np
import cv2
from .common import ROOT,write,sha
from .delta_instrument import DIRECTORY,FEATURES,decode
from .delta_optical import OPTICAL,raymap,track_rgb,platform_plane,images

DEST=DIRECTORY/'paired'


def test_rows():
    manifest=json.loads((ROOT/'output/model_expansion/spatial/dataset_manifest.json').read_text())
    return [r for r in manifest['videos'] if r['category']=='H-B' and r['view']=='front' and r['split']=='test']


def rgb_path(row,mode):
    return ROOT/'output/model_expansion/lingbot_spatial'/mode/'predictions'/(row['video_id']+'_rgb.npz')


def cache():
    import torch
    from torch.nn import functional as F
    from experiments.model_expansion.profile_vjepa import UPSTREAM,WEIGHTS,clean_weights
    from experiments.model_expansion.spatial_vjepa import pooled,DIM
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(UPSTREAM))
    from src.hub.backbones import vjepa2_1_vit_base_384
    encoder,predictor=vjepa2_1_vit_base_384(pretrained=False)
    state=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    encoder.load_state_dict(clean_weights(state['ema_encoder']),strict=True)
    del state,predictor;gc.collect();encoder.eval().requires_grad_(False).cuda()
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    for mode in ('zero_shot','adapted'):
        for row in test_rows():
            source=rgb_path(row,mode);target=DEST/mode/'features'/(row['video_id']+'.npy')
            meta=target.with_suffix('.json');binding=dict(rgb_sha256=sha(source),encoder_sha256=sha(WEIGHTS),
                input_resolution=256,encoder_resolution=384,batch=8,encoder_precision='BF16',pooled_precision='FP16',
                source_sha256=sha(__file__))
            if target.exists() and meta.exists() and json.loads(meta.read_text())==binding:continue
            with np.load(source) as data:rgb=data['rgb']
            assert rgb.shape==(120,256,256,3)
            features=[]
            with torch.inference_mode():
                for first in range(0,120,8):
                    frames=np.stack([cv2.resize(im,(384,384),interpolation=cv2.INTER_CUBIC) for im in rgb[first:first+8]])
                    x=torch.from_numpy(frames).cuda().permute(0,3,1,2).float()/255
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        h=F.layer_norm(encoder(((x-mean)/std).unsqueeze(2).repeat(1,1,2,1,1)),(DIM,))
                    features.append(pooled(h.to(torch.float16).float()).cpu().numpy().astype(np.float16))
            target.parent.mkdir(parents=True,exist_ok=True);np.save(target,np.concatenate(features));write(meta,binding)
            print(json.dumps(dict(stage='features',mode=mode,id=row['video_id'])),flush=True)


def read_case(task):
    from experiments.closed_chain_eval.geometry_spatial_reader import nominal
    row,mode=task;folder=DEST/mode;target=folder/'predictions/H-B'/(row['video_id']+'.npz')
    meta=target.with_suffix('.json')
    if meta.exists():return json.loads(meta.read_text())
    features=FEATURES/(row['video_id']+'.npy') if mode=='reference' else folder/'features'/(row['video_id']+'.npy')
    if mode=='reference':frames=images(row);source=ROOT/row['video']
    else:
        source=rgb_path(row,mode)
        with np.load(source) as d:frames=np.stack([cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in d['rgb']])
    with np.load(DIRECTORY/'relative_head.npz') as d:matrix,window=d['matrix'],int(d['window'])
    assert window==1,'No coordinate smoothing in this instrument'
    q=decode(np.load(features),nominal(row),matrix,window)
    result=dict(mode=mode,case_id=row['video_id'],source_sha256=sha(source),
        head_sha256=sha(DIRECTORY/'relative_head.npz'),selection_sha256=sha(OPTICAL/'fusion_selection.json'),
        coordinate_smoothing=False,uses_CAD_reset=True,forecast_changed=False)
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        tracks=track_rgb(row,frames);plane,quality=platform_plane(row,tracks)
        camera=json.loads((ROOT/row['public_inputs']).read_text())['observed_camera_frames'][0]
        direction=np.asarray(camera['R_world_to_camera'])[2]
        q[:,:3]=plane+np.outer((q[:,:3]-plane)@direction,direction)
        if not np.isfinite(q).all():raise ValueError('Nonfinite measured coordinates')
        np.savez_compressed(target,qpos=q,time=np.arange(120)/30,observed_frames=np.array(3))
        np.savez_compressed(target.with_name(target.stem+'_tracks.npz'),**tracks)
        result.update(finite=True,quality=quality)
    except ValueError as error:result.update(finite=False,error=str(error))
    write(meta,result);return result


def score_case(task):
    from experiments.model_expansion.evaluate_spatial import evaluate_case
    row,mode=task;folder=DEST/mode
    result=evaluate_case((row,'score',str(folder)))
    write(folder/'cases'/(row['video_id']+'.json'),result)
    return result


def run():
    verification=json.loads((OPTICAL/'verification_summary.json').read_text())
    if verification['selection_sha256']!=sha(OPTICAL/'fusion_selection.json'):raise ValueError('Selection changed after verification')
    records=test_rows();seen=set()
    for row in records:
        if row['assembly_id'] not in seen:raymap(row);seen.add(row['assembly_id'])
    tasks=[(r,m) for m in ('reference','zero_shot','adapted') for r in records]
    with ProcessPoolExecutor(max_workers=3) as pool:readings=list(pool.map(read_case,tasks))
    write(DEST/'reading_complete.json',dict(cases=readings,selection_sha256=sha(OPTICAL/'fusion_selection.json'),
        frozen_before_test_replay=True,forecast_pixels_changed=False))
    summaries={}
    for mode in ('reference','zero_shot','adapted'):
        with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(score_case,[(r,mode) for r in records]))
        write(DEST/mode/'H-B_score.json',results)
        summaries[mode]=dict(total=len(records),measured=sum(r.get('status')=='evaluated' for r in results),
            mechanics=sum(r.get('kinematic_intervention',{}).get('mechanical_coverage',False) for r in results))
        print(json.dumps(dict(mode=mode,**summaries[mode])),flush=True)
    write(DEST/'summary.json',summaries)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['cache','score']);args=p.parse_args()
    if args.command=='cache':cache()
    else:run()
