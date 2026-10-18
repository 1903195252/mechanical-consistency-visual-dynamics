"""Full native LingBot objective with explicit seeds, checkpoints and resume."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time

from .common import ROOT,OUT,OLD_RUN,SEEDS,STEPS,sha,write,records,training_records,generation_seed


def check_window():
    from experiments.runtime import check_deadline, deadline
    check_deadline()
    return {"compute_stop_unix": deadline()}


def configure(seed=None):
    from experiments.model_expansion import lingbot_spatial as native
    shared=OUT/'shared';shared.mkdir(exist_ok=True)
    for name in ('action_norm.json','text_embeddings.pt','official_weight_manifest.json'):
        source=OLD_RUN/name;destination=shared/name
        if not destination.exists():shutil.copyfile(source,destination)
        if sha(source)!=sha(destination):raise ValueError('Frozen shared input changed: '+name)
    native.OUT=shared if seed is None else OUT/'runs'/('seed%d'%seed)
    native.OUT.mkdir(parents=True,exist_ok=True)
    if seed is not None:
        for name in ('action_norm.json','text_embeddings.pt','official_weight_manifest.json'):
            target=native.OUT/name
            if not target.exists():shutil.copyfile(shared/name,target)
    native.SEED=42 if seed is None else seed
    native.LATENTS=shared/'latents'
    native.records=lambda category=None:[r for r in training_records() if category is None or r['category']==category]
    native.check_deadline=check_window
    return native


def prepare_cache(shard=0,shards=1):
    check_window()
    import numpy as np
    import torch
    native=configure();native.setup_native()
    from modules.utils import load_vae
    native.wait_verified('vae');native.LATENTS.mkdir(exist_ok=True)
    vae=load_vae(str(native.WEIGHTS/'vae'),torch.bfloat16,'cuda').eval()
    mean=torch.tensor(vae.config.latents_mean,device='cuda').view(1,-1,1,1,1)
    std=torch.tensor(vae.config.latents_std,device='cuda').view(1,-1,1,1,1)
    rows=training_records()
    if not 0<=shard<shards:raise ValueError('Invalid cache shard')
    with torch.inference_mode():
        for row in rows[shard::shards]:
            check_window();path=native.LATENTS/(row['video_id']+'.pt')
            if path.exists():
                cached=torch.load(path,map_location='cpu',weights_only=True)
                if cached.shape==(48,31,16,16) and torch.isfinite(cached).all():continue
                raise ValueError('Invalid training cache: '+str(path))
            rgb=native.read_video(row);rgb=np.concatenate([rgb,rgb[-1:]],axis=0)
            x=torch.from_numpy(rgb).cuda().permute(3,0,1,2)[None].to(torch.bfloat16)/127.5-1
            z=vae.encode(x).latent_dist.mode()
            z=((z.float()-mean)/std).to(torch.bfloat16)
            assert z.shape==(1,48,31,16,16) and torch.isfinite(z).all()
            temporary=path.with_suffix('.partial');torch.save(z[0].cpu(),temporary);temporary.replace(path)
    write(OUT/'shared'/('cache_shard%02d.json'%shard),dict(training_videos=len(rows[shard::shards]),shard=shard,shards=shards,
        training_manifest_sha256=sha(OUT/'training_manifest.json')))
    if all((native.LATENTS/(r['video_id']+'.pt')).exists() for r in rows):
        write(OUT/'shared/cache_complete.json',dict(training_videos=len(rows),test_frames_encoded=0,
            training_manifest_sha256=sha(OUT/'training_manifest.json'),frames=121,latent_frames=31))


def train(seed,max_updates=STEPS[-1],preflight=False):
    check_window()
    if seed not in SEEDS:raise ValueError('Unregistered seed')
    if max_updates!=STEPS[-1] and not preflight:raise ValueError('Formal training is fixed at 50 updates')
    import numpy as np
    import torch
    from torch.utils.data._utils.collate import default_collate
    from .checkpoint import SequenceOrder,save,restore
    native=configure(seed)
    if preflight:
        native.OUT=OUT/'preflight'/('seed%d'%seed);native.OUT.mkdir(parents=True,exist_ok=True)
        for name in ('action_norm.json','text_embeddings.pt','official_weight_manifest.json'):
            shutil.copyfile(OUT/'shared'/name,native.OUT/name)
    cache=json.loads((OUT/'shared/cache_complete.json').read_text())
    assert cache['training_manifest_sha256']==sha(OUT/'training_manifest.json')
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    cfg=native.config();cfg.num_steps=STEPS[-1];cfg.learning_rate=1e-5
    assert cfg.batch_size==1 and cfg.gradient_accumulation_steps==8 and cfg.warmup_steps==10
    os.environ.setdefault('MASTER_ADDR','127.0.0.1');os.environ.setdefault('MASTER_PORT',str(29700+seed))
    torch.cuda.set_device(0);torch.distributed.init_process_group('nccl',rank=0,world_size=1)
    trainer=native.native_trainer_class()(cfg)
    trainer.optimizer=torch.optim.AdamW(trainer.transformer.parameters(),lr=cfg.learning_rate,
        betas=(cfg.beta1,cfg.beta2),eps=1e-8,weight_decay=cfg.weight_decay,foreach=False)
    from utils import warmup_constant_lambda
    trainer.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(trainer.optimizer,
        lr_lambda=lambda step:warmup_constant_lambda(step,warmup_steps=cfg.warmup_steps))
    dataset=native.SpatialLatentDataset(cfg);order=SequenceOrder(len(dataset),seed)
    identity=dict(seed=seed,protocol_sha256=sha(OUT/'protocol.json'),training_manifest_sha256=sha(OUT/'training_manifest.json'),
                  adapter_sha256=sha(__file__),checkpoint_sha256=sha(Path(__file__).with_name('checkpoint.py')),
                  native_adapter_sha256=sha(native.__file__),preflight=preflight)
    resume=native.OUT/'resume.pt'
    trainer.step=restore(resume,trainer.transformer,trainer.optimizer,trainer.lr_scheduler,order,identity,distributed=True) if resume.exists() else 0
    def export_current():
        trainer.save_checkpoint()
        weight=native.OUT/'checkpoints'/('checkpoint_step_%d'%trainer.step)/'transformer/diffusion_pytorch_model.safetensors'
        write(native.OUT/('checkpoint_%04d.json'%trainer.step),dict(step=trainer.step,seed=seed,
            checkpoint=str(weight.relative_to(ROOT)),checkpoint_sha256=sha(weight),identity=identity))
    # Recover a stopped export at the exact saved update before taking another step.
    if not preflight and trainer.step in STEPS[1:] and not (native.OUT/('checkpoint_%04d.json'%trainer.step)).exists():
        with (OUT/'checkpoint_io.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);export_current()
    stopping=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))
    window=check_window();started=time.time();initial=trainer.step;update_seconds=[];checkpoint_seconds=[]
    while trainer.step<max_updates and not stopping[0] and time.time()<window['compute_stop_unix']:
        start=time.time();step=trainer.step;latent_total=0.;action_total=0.
        for micro in range(8):
            losses=trainer._train_step(default_collate([dataset[order.next()]]),step*8+micro)
            if not torch.isfinite(losses['latent_loss']) or not torch.isfinite(losses['action_loss']):
                raise FloatingPointError('Nonfinite full native training objective')
            latent_total+=float(losses['latent_loss']);action_total+=float(losses['action_loss'])
        assert losses['should_log'];trainer.step+=1
        item=dict(update=trainer.step,exposures=trainer.step*8,seconds=time.time()-start,
                  latent_loss=latent_total,action_loss=action_total,
                  gpu_peak_GB=torch.cuda.max_memory_allocated()/1e9)
        update_seconds.append(item['seconds'])
        with (native.OUT/'training.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n')
        write(native.OUT/'status.json',dict(stage='training',**item))
        if trainer.step%25==0 or trainer.step in STEPS or stopping[0]:
            # Serializing large state exports prevents two seeds from peaking
            # in host memory and storage bandwidth at the same instant.
            with (OUT/'checkpoint_io.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                saved_at=time.time()
                save(resume,trainer.transformer,trainer.optimizer,trainer.lr_scheduler,order,trainer.step,identity,distributed=True)
                checkpoint_seconds.append(time.time()-saved_at)
                if trainer.step in STEPS and not preflight:export_current()
    if trainer.step%25:
        with (OUT/'checkpoint_io.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            saved_at=time.time()
            save(resume,trainer.transformer,trainer.optimizer,trainer.lr_scheduler,order,trainer.step,identity,distributed=True)
            checkpoint_seconds.append(time.time()-saved_at)
    report=dict(seed=seed,updates=trainer.step,complete=trainer.step==max_updates,sequence_exposures=trainer.step*8,
                resumed_at_update=initial,session_seconds=time.time()-started,identity=identity,
                update_seconds=update_seconds,full_checkpoint_save_seconds=checkpoint_seconds)
    if preflight and report['complete']:
        from .checkpoint import fingerprint
        before=fingerprint(trainer.transformer,trainer.optimizer,trainer.lr_scheduler,order)
        # A load into unchanged memory cannot detect a no-op restore. Perturb
        # one parameter and its Adam moment before checking all tensor bytes.
        with torch.no_grad():
            parameter=next(trainer.transformer.parameters())
            local=parameter.to_local() if hasattr(parameter,'to_local') else parameter
            local.reshape(-1)[0].add_(1.)
            for state in trainer.optimizer.state.values():
                if 'exp_avg' in state:
                    value=state['exp_avg'];value=value.to_local() if hasattr(value,'to_local') else value
                    value.reshape(-1)[0].add_(1.);break
        restored_at=time.time()
        restored=restore(resume,trainer.transformer,trainer.optimizer,trainer.lr_scheduler,order,identity,distributed=True)
        report['full_checkpoint_restore_seconds']=time.time()-restored_at
        assert restored==trainer.step
        after=fingerprint(trainer.transformer,trainer.optimizer,trainer.lr_scheduler,order)
        if before!=after:raise ValueError('Full-model checkpoint restore differs from the saved live state')
        report.update(full_optimizer_checkpoint_restore=True,checkpoint_bytes=resume.stat().st_size,
            all_model_optimizer_tensor_bytes_restored=True,state_sha256=after,
            gpu_peak_GB=torch.cuda.max_memory_allocated()/1e9,not_experimental_evidence=True)
        write(native.OUT/'benchmark.json',report)
    else:write(native.OUT/('training_complete.json' if report['complete'] else 'training_interrupted.json'),report)
    torch.distributed.destroy_process_group()


def infer(seed,step,shard=0,shards=1):
    check_window()
    if seed not in SEEDS or step not in STEPS or not 0<=shard<shards:raise ValueError('Unregistered inference job')
    import gc
    import numpy as np
    import torch
    from safetensors.torch import load_file
    native=configure(seed);native.setup_native()
    from wan_va_server import VA_Server
    from diffusers.video_processor import VideoProcessor
    cfg=native.config();server=VA_Server(cfg)
    weight_identity=sha(OLD_RUN/'official_weight_manifest.json')
    if step:
        saved=json.loads((native.OUT/('checkpoint_%04d.json'%step)).read_text())
        checkpoint=ROOT/saved['checkpoint'];weight_identity=sha(checkpoint)
        if saved['checkpoint_sha256']!=weight_identity:raise ValueError('Checkpoint changed')
        state=load_file(str(checkpoint));server.transformer.load_state_dict(state,strict=True);del state
    server.video_processor=VideoProcessor(vae_scale_factor=1)
    emb=torch.load(native.OUT/'text_embeddings.pt',weights_only=True)
    server.encode_prompt=lambda prompt,**kw:(emb[prompt][None].cuda(),emb['empty'][None].cuda())
    del server.text_encoder;gc.collect();torch.cuda.empty_cache()
    norms=json.loads((native.OUT/'action_norm.json').read_text())
    rows=sorted(records('test'),key=lambda r:r['video_id'])[shard::shards]
    destination=native.OUT/('step%04d'%step);pred=destination/'predictions';pred.mkdir(parents=True,exist_ok=True)
    identity=dict(seed=seed,step=step,checkpoint_sha256=weight_identity,manifest_sha256=sha(OUT/'dataset_manifest.json'),
                  protocol_sha256=sha(OUT/'protocol.json'),adapter_sha256=sha(__file__))
    started=time.time();finite=0
    with torch.inference_mode():
        for row in rows:
            check_window();target=pred/(row['video_id']+'_rgb.npz');meta=pred/(row['video_id']+'.json')
            if meta.exists():
                previous=json.loads(meta.read_text())
                if previous['identity']!=identity:raise ValueError('Prediction identity changed')
                if previous['finite'] and target.exists() and sha(target)==previous['rgb_sha256']:
                    finite+=1;continue
                if not previous['finite']:continue
            tick=time.time();torch.manual_seed(generation_seed(seed,row['video_id']))
            prefix=native.read_video(row,count=3)
            cfg.used_action_channel_ids=list(range(14,20 if row['category']=='H-A' else 17));cfg.norm_stat=norms[row['category']]
            server._reset(row['category'])
            action,mask=native.command_array(row,start_seconds=2/30);norm=norms[row['category']]
            action=np.clip((action-np.asarray(norm['q01']))/(np.asarray(norm['q99'])-norm['q01']+1e-6)*2-1,-1.5,1.5)*mask
            commands=torch.from_numpy(action.astype(np.float32)).cuda().reshape(32,8,30).permute(2,0,1)[None,...,None].to(torch.bfloat16)
            latent=torch.cat([native.video_chunk(server,prefix[-1],commands,k) for k in range(8)],dim=2)
            server.transformer.clear_cache(server.cache_name);server.streaming_vae.clear_cache()
            success=bool(torch.isfinite(latent).all())
            video=server.decode_one_video(latent,'np')[0] if success else None
            success=success and bool(np.isfinite(video).all())
            record=dict(identity=identity,video_id=row['video_id'],finite=success,
                        generation_seed=generation_seed(seed,row['video_id']),seconds=time.time()-tick)
            if success:
                rgb=np.rint(np.clip(video[:118],0,1)*255).astype(np.uint8)
                assert rgb.shape==(118,256,256,3)
                temporary=target.with_suffix('.partial')
                with temporary.open('wb') as stream:np.savez_compressed(stream,rgb=np.concatenate([prefix[:2],rgb]),time=np.arange(120)/30,observed_frames=np.array(3))
                temporary.replace(target);record['rgb_sha256']=sha(target);finite+=1
            write(meta,record)
    write(destination/('inference_shard%02d.json'%shard),dict(identity=identity,cases=len(rows),finite=finite,
          shard=shard,shards=shards,seconds=time.time()-started,complete=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['cache','preflight','train','infer'])
    p.add_argument('--seed',type=int,choices=SEEDS,default=42);p.add_argument('--step',type=int,choices=STEPS,default=0)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args()
    if a.command=='cache':prepare_cache(a.shard,a.shards)
    elif a.command=='preflight':train(a.seed,max_updates=3,preflight=True)
    elif a.command=='train':train(a.seed)
    else:infer(a.seed,a.step,a.shard,a.shards)
