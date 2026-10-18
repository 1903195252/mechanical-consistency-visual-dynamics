"""Paired reference/generated RGB metrology and the existing continuous SI audit."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys

from .common import ROOT,OUT,FAMILIES,SEEDS,STEPS,TEST_CASES,sha,write,records


def destination(seed=None,step=None):
    return OUT/'reference' if seed is None else OUT/'runs'/('seed%d'%seed)/('step%04d'%step)


def read_rgb(seed=None,step=None,limit=0):
    import cv2
    import gc
    import numpy as np
    import torch
    from torch.nn import functional as F
    from experiments.model_expansion.profile_vjepa import UPSTREAM,WEIGHTS,clean_weights
    from experiments.model_expansion.spatial_vjepa import camera,pooled,DIM
    from experiments.model_expansion.lingbot_spatial import read_video
    from experiments.model_expansion.checkpoint_io import load_checkpoint
    from experiments.closed_chain_eval.geometry_spatial_reader import GeometryHead,nominal,DIRECTORY
    from .delta_instrument import DIRECTORY as DELTA,decode
    from .delta_optical import OPTICAL,track_rgb,platform_plane
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(UPSTREAM))
    from src.hub.backbones import vjepa2_1_vit_base_384
    encoder,predictor=vjepa2_1_vit_base_384(pretrained=False)
    weights=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    encoder.load_state_dict(clean_weights(weights['ema_encoder']),strict=True)
    del weights,predictor;gc.collect();encoder.eval().requires_grad_(False).cuda()
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    heads={};hashes={}
    measurement=json.loads((OUT/'measurement_protocol.json').read_text())
    for path,expected in measurement['files'].items():
        if sha(ROOT/path)!=expected['sha256']:raise ValueError('Frozen measurement input changed: '+path)
    for family in ('H-A',):
        path=DIRECTORY/(family+'_head.pt');state=load_checkpoint(path);metadata=state['metadata']
        head=GeometryHead(metadata['nq']).cuda().eval();head.load_state_dict(state['model'])
        heads[family]=(head,metadata);hashes[family]=sha(path)
    with np.load(DELTA/'relative_head.npz') as state:
        delta_matrix=state['matrix'];delta_window=int(state['window'])
    assert delta_window==1,'The registered instrument has no coordinate smoothing'
    hashes['H-B']=sha(DELTA/'relative_head.npz')
    folder=destination(seed,step);rows=records('test')[:limit or None]
    profile=dict(encoder_sha256=sha(WEIGHTS),reader_sha256=hashes,
        implementation_sha256=sha(__file__),torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        measurement_protocol_sha256=sha(OUT/'measurement_protocol.json'),
        encoder_precision='native BF16 autocast',head_precision='FP32',TF32=False,batch=8,
        input_resize='640->256 INTER_AREA for reference; 256->384 INTER_CUBIC for both paths',
        prefix='same three observed RGB frames, measured by the same instrument',fit_test_labels=False,
        Delta_readout='reset-relative visual depth + independently tracked image-plane motion',
        coordinate_smoothing=False,uses_public_CAD_reset=True)
    for row in rows:
        source=ROOT/row['video'] if seed is None else folder/'predictions'/(row['video_id']+'_rgb.npz')
        target=folder/'readout/predictions'/row['category']/(row['video_id']+'.npz')
        if not source.exists():continue  # Failure remains in the score denominator.
        if seed is None:
            cache=folder/'rgb'/(row['video_id']+'.npz');cache.parent.mkdir(parents=True,exist_ok=True)
            if cache.exists():
                with np.load(cache) as data:images=data['rgb']
            else:
                images=read_video(row);temporary=cache.with_suffix('.partial')
                with temporary.open('wb') as stream:np.savez_compressed(stream,rgb=images)
                temporary.replace(cache)
        else:
            with np.load(source) as data:images=data['rgb']
        assert images.shape==(120,256,256,3)
        binding=dict(source_sha256=sha(source),profile=profile)
        meta=target.with_suffix('.json')
        if meta.exists():
            prior=json.loads(meta.read_text())
            if prior.get('binding')==binding and (not prior['finite'] or target.exists()):continue
        q0=nominal(row)
        if row['category']=='H-A':
            head,m=heads[row['category']]
            context=torch.tensor((np.r_[camera(row),q0]-m['cmean'])/m['cstd'],device='cuda')
        decoded=[]
        with torch.inference_mode():
            for start in range(0,120,8):
                frames=np.stack([cv2.resize(frame,(384,384),interpolation=cv2.INTER_CUBIC) for frame in images[start:start+8]])
                x=torch.from_numpy(frames).cuda().permute(0,3,1,2).float()/255
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    h=F.layer_norm(encoder(((x-mean)/std).unsqueeze(2).repeat(1,1,2,1,1)),(DIM,))
                h=pooled(h.to(torch.float16).float()).to(torch.float16).float()
                decoded.append((head(h,context.expand(len(h),-1)) if row['category']=='H-A' else h).cpu().numpy())
        if row['category']=='H-A':q=np.concatenate(decoded)*m['ystd']+m['ymean']+q0
        else:
            q=decode(np.concatenate(decoded),q0,delta_matrix,delta_window)
            try:
                gray=np.stack([cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) for frame in images])
                tracks=track_rgb(row,gray);plane,quality=platform_plane(row,tracks)
                public=json.loads((ROOT/row['public_inputs']).read_text())
                direction=np.asarray(public['observed_camera_frames'][0]['R_world_to_camera'])[2]
                q[:,:3]=plane+np.outer((q[:,:3]-plane)@direction,direction)
            except ValueError as error:
                target.unlink(missing_ok=True)
                write(meta,dict(binding=binding,finite=False,error=str(error)))
                continue
        norm=np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
        if not np.isfinite(q).all() or np.any(norm<1e-8):raise ValueError('Invalid measurement instrument output')
        q[:,3:7]/=norm
        target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.partial')
        with temp.open('wb') as stream:np.savez_compressed(stream,qpos=q,time=np.arange(120)/30,observed_frames=np.array(3))
        temp.replace(target);write(meta,dict(binding=binding,finite=True))
    write(folder/'reading_complete.json',dict(profile=profile,complete=len(rows)==TEST_CASES,expected_cases=len(rows),
        read_cases=sum((folder/'readout/predictions'/r['category']/(r['video_id']+'.npz')).exists() for r in rows),
        all_registered_cases=len(rows)==TEST_CASES))


def score_one(task):
    import numpy as np
    from experiments.model_expansion.evaluate_spatial import evaluate_case,Audit
    row,folder,reference=task;folder=Path(folder)
    result=evaluate_case((row,'score',str(folder/'readout')))
    if reference:
        calibration=evaluate_case((row,'calibrate'))
        result['exact_state_control']=calibration['reference_exact']
        result['sampled_state_control']=calibration['reference_30Hz']
    audit=Audit(row)
    result['engineering_limits']=dict(geometry=.001,constraint_velocity=.01,constraint_acceleration=.1,
        prediction=.01,dynamics=.05*audit.limit,energy=.01*audit.limit*float(np.sum(np.ptp(audit.command_p,axis=0))))
    result['effort_unit']=audit.unit
    if not reference:
        rgb=folder/'predictions'/(row['video_id']+'_rgb.npz')
        if rgb.exists():
            with np.load(rgb) as data:generated=data['rgb'][3:].astype(np.float32)/255
            with np.load(OUT/'reference/rgb'/(row['video_id']+'.npz')) as data:
                truth=data['rgb'][3:].astype(np.float32)/255;hold=data['rgb'][2].astype(np.float32)/255
            mask=np.max(np.abs(truth-hold),axis=-1)>2/255
            result['visual']=dict(rgb_mse=float(np.mean((generated-truth)**2)),
                persistence_mse=float(np.mean((hold-truth)**2)),
                motion_mask_mse=float(np.mean((generated-truth)[mask]**2)) if mask.any() else None,
                persistence_motion_mask_mse=float(np.mean((hold-truth)[mask]**2)) if mask.any() else None)
        result['paired_reference']=str((OUT/'reference/cases'/(row['video_id']+'.json')).relative_to(ROOT))
    write(folder/'cases'/(row['video_id']+'.json'),result)
    return result


def score(seed=None,step=None,workers=4):
    from experiments.closed_chain_eval.rescore_continuous import spatial_values,make_card,UNITS
    from experiments.closed_chain_eval.continuous import aggregate_continuous
    folder=destination(seed,step);rows=records('test')
    reading=json.loads((folder/'reading_complete.json').read_text())
    if not reading['all_registered_cases'] or reading['expected_cases']!=TEST_CASES:
        raise ValueError('A partial instrument run is not a complete evaluation')
    if seed is None and reading['read_cases']!=TEST_CASES:
        raise ValueError('Every matched reference must be measured')
    # All reference controls must exist before generated mechanics are reported.
    if seed is not None:
        reference_reading=json.loads((OUT/'reference/reading_complete.json').read_text())
        if reference_reading['profile']!=reading['profile']:
            raise ValueError('Reference/generated videos were measured with different instruments or runtimes')
        for row in rows:
            if not (OUT/'reference/cases'/(row['video_id']+'.json')).exists():raise ValueError('Missing paired reference control')
            marker=folder/'predictions'/(row['video_id']+'.json')
            if not marker.exists():raise ValueError('Forecasting has not finished for every registered case')
            prediction=json.loads(marker.read_text())
            if prediction['finite']:
                path=folder/'readout/predictions'/row['category']/(row['video_id']+'.json')
                if not path.exists():raise ValueError('A finite forecast has not passed through the instrument')
                readout=json.loads(path.read_text())
                if readout['finite'] and not path.with_suffix('.npz').exists():
                    raise ValueError('Measured coordinates are missing')
    tasks=[(r,str(folder),seed is None) for r in rows]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(score_one,tasks))
    families={}
    for family in FAMILIES:
        subset=[r for r in results if r['category']==family]
        calibration=sha(OUT/'measurement_protocol.json')
        expected=[(family,r['condition'],r['video_id']) for r in subset]
        branches={}
        for branch in ['native','kinematic_intervention']+(['exact_state_control','sampled_state_control'] if seed is None else []):
            cards=[make_card(spatial_values(r.get(branch,{})),r['engineering_limits'],dict(UNITS,dynamics=r['effort_unit']),
                             calibration,(family,r['condition'],r['video_id'])) for r in subset]
            branches[branch]=aggregate_continuous(cards,expected,calibration_id=calibration)
        import numpy as np
        visual=[r['visual'] for r in subset if 'visual' in r]
        visual_means={key:float(np.mean([r[key] for r in visual if r.get(key) is not None]))
                      for key in ('rgb_mse','persistence_mse','motion_mask_mse','persistence_motion_mask_mse')
                      if any(r.get(key) is not None for r in visual)}
        if visual_means.get('persistence_mse',0)>0:
            visual_means['rgb_to_persistence']=visual_means['rgb_mse']/visual_means['persistence_mse']
        families[family]=dict(cases=len(subset),branches=branches,visual=visual_means,
            finite_videos=len(visual) if seed is not None else len(subset),
            mechanics_coverage={b:sum(bool(r.get(b,{}).get('mechanical_coverage')) for r in subset)
                                for b in ('native','kinematic_intervention')})
    report=dict(seed=seed,step=step,cases=len(results),families=families,scoring_version='continuous-si-1',
        protocol_sha256=sha(OUT/'protocol.json'),manifest_sha256=sha(OUT/'dataset_manifest.json'),
        implementation_sha256=sha(__file__),reference_controls=seed is None,RMSE_subtraction=False)
    report['measurement_profile_sha256']=hashlib.sha256(json.dumps(reading['profile'],sort_keys=True).encode()).hexdigest()
    write(folder/'scores.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['read','score'])
    p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--step',type=int,choices=STEPS)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if (a.seed is None)!=(a.step is None):p.error('Specify seed and step together, or neither for reference controls')
    if a.command=='read':read_rgb(a.seed,a.step,a.limit)
    else:score(a.seed,a.step,a.workers)
