"""Native LingBot-VA spatial data adapter, full-model training and command replay.

The released 5B transformer and joint video/action flow-matching objective are
preserved. Spatial commands occupy left-joint channels, with unused channels
masked. SI coordinates come from the separately fitted, frozen visual observer.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import struct
import sys
import time
import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .prepare_spatial import ROOT, write_json, sha256
from .spatial_vjepa import records, inputs, camera, Readout, pooled, DIM
from .checkpoint_io import load_checkpoint

OUT = ROOT / 'output/model_expansion/lingbot_spatial'
UPSTREAM = ROOT / 'output/model_expansion/upstream/lingbot_va'
WEIGHTS = ROOT / 'output/model_expansion/weights/lingbot-va-base'
LATENTS = OUT / 'latents'
SEED = 42
PROMPTS = {
    'H-A': 'A fixed camera observes a six-leg Stewart parallel platform moving under joint position commands. The rigid metal rods connect the base and moving platform.',
    'H-B': 'A fixed camera observes a three-arm Delta parallel robot moving under joint position commands. The rigid metal rods connect the base and moving platform.',
    'empty': '',
}


def check_deadline():
    from experiments.runtime import check_deadline as check
    check()


def training_event(stage, **details):
    event = {'stage': stage, 'unix': time.time(),
             'process_peak_rss_GB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9,
             **details}
    if torch.cuda.is_initialized():
        event['gpu_allocated_GB'] = torch.cuda.memory_allocated() / 1e9
        event['gpu_peak_GB'] = torch.cuda.max_memory_allocated() / 1e9
    memory = Path('/sys/fs/cgroup/memory.current')
    if memory.exists():
        event['container_memory_GB'] = int(memory.read_text()) / 1e9
    print(json.dumps(event), flush=True)


def save_streaming_checkpoint(trainer):
    """Write the native BF16 checkpoint without gathering a second full model.

    This adapter runs on one rank. Its local FSDP shards therefore contain the
    complete parameter tensors. The file is standard safetensors and every
    tensor is checked against the live model before the checkpoint is published.
    """
    if torch.distributed.get_world_size() != 1:
        raise ValueError('Streaming checkpoint requires the registered one-rank run')
    state = trainer.transformer.state_dict()
    tensors = {}
    for name, value in state.items():
        local = value.to_local() if hasattr(value, 'to_local') else value
        if tuple(local.shape) != tuple(value.shape):
            raise ValueError('Local FSDP tensor is not complete: ' + name)
        tensors[name] = local.detach()
    header, offset = {}, 0
    for name in sorted(tensors):
        value = tensors[name]
        size = value.numel() * 2
        header[name] = {'dtype': 'BF16', 'shape': list(value.shape), 'data_offsets': [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    directory = OUT / 'checkpoints' / ('checkpoint_step_%d' % trainer.step) / 'transformer'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'diffusion_pytorch_model.safetensors'
    temporary = target.with_suffix('.safetensors.tmp')
    digest = hashlib.sha256()
    with temporary.open('wb') as stream:
        first = struct.pack('<Q', len(encoded)) + encoded
        stream.write(first); digest.update(first)
        for name in sorted(tensors):
            value = tensors[name].to(dtype=torch.bfloat16, device='cpu').contiguous()
            block = value.view(torch.uint8).numpy().tobytes()
            stream.write(block); digest.update(block)
    from safetensors import safe_open
    with safe_open(str(temporary), framework='pt', device='cpu') as saved:
        if set(saved.keys()) != set(tensors):
            raise ValueError('Saved checkpoint tensor keys differ')
        for name, value in tensors.items():
            if not torch.equal(saved.get_tensor(name), value.to(dtype=torch.bfloat16, device='cpu')):
                raise ValueError('Saved checkpoint differs from the live model: ' + name)
    temporary.replace(target)
    cfg = dict(trainer.transformer.config)
    cfg.pop('_name_or_path', None)
    write_json(directory / 'config.json', cfg)
    write_json(directory.parent / 'verification.json', {
        'step': trainer.step, 'parameters': sum(v.numel() for v in tensors.values()),
        'tensors': len(tensors), 'dtype': 'BF16, identical to the released trainer checkpoint format',
        'sha256': digest.hexdigest(), 'every_tensor_equal_to_live_model_bf16': True,
        'bytes': target.stat().st_size,
    })


def setup_native():
    sys.path.insert(0, str(UPSTREAM / 'wan_va'))
    # Upstream imports FlashAttention even when its torch/flex backends are used.
    # Supply the exact SDPA-equivalent operation for that optional import; actual
    # inference explicitly selects torch and training explicitly selects flex.
    try:
        import flash_attn_interface
    except ImportError:
        shim = types.ModuleType('flash_attn_interface')
        def optional_flash(q, k, v, **kwargs):
            raise RuntimeError('FlashAttention backend is unavailable; use native torch/flex')
        shim.flash_attn_func = optional_flash
        sys.modules['flash_attn_interface'] = shim


def config():
    setup_native()
    from configs.va_demo_train_cfg import va_demo_train_cfg
    cfg = copy.deepcopy(va_demo_train_cfg)
    cfg.wan22_pretrained_model_name_or_path = str(WEIGHTS)
    cfg.save_root = str(OUT)
    cfg.obs_cam_keys = ['front']
    cfg.local_rank = cfg.rank = 0
    cfg.world_size = 1
    cfg.enable_offload = False
    cfg.enable_wandb = False
    cfg.load_worker = 0
    cfg.used_action_channel_ids = list(range(14, 20))
    cfg.inverse_used_action_channel_ids = [6] * 30
    for i, j in enumerate(cfg.used_action_channel_ids):
        cfg.inverse_used_action_channel_ids[j] = i
    norms = OUT / 'action_norm.json'
    if norms.exists():
        cfg.norm_stat = json.loads(norms.read_text())['H-A']
    return cfg


def read_video(row, count=120, offset=0):
    capture = cv2.VideoCapture(str(ROOT / row['video']))
    frames = []
    try:
        for i in range(offset + count):
            ok, frame = capture.read()
            if not ok:
                raise ValueError('Incomplete video: ' + row['video_id'])
            if i >= offset:
                frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (256, 256), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
    return np.stack(frames)


def command_array(row, frames=32, start_seconds=0):
    action, _ = inputs(row)
    n = 6 if row['category'] == 'H-A' else 3
    times = np.arange((frames - 1) * 8) / 60 + start_seconds
    q = np.stack([np.interp(times, np.arange(120) / 30, action[:, i]) for i in range(n)], -1)
    q = np.concatenate([np.zeros((8, n)), q])
    result = np.zeros((frames * 8, 30), np.float32)
    mask = np.zeros_like(result, dtype=bool)
    result[:, 14:14 + n] = q
    mask[:, 14:14 + n] = True
    return result, mask


def prepare_norm():
    norms = {}
    for category in ('H-A', 'H-B'):
        q = np.concatenate([command_array(r)[0][8:] for r in records(category) if r['split'] == 'train'])
        norms[category] = {'q01': np.quantile(q, .01, axis=0).tolist(), 'q99': np.quantile(q, .99, axis=0).tolist()}
    write_json(OUT / 'action_norm.json', norms)
    return norms


@torch.no_grad()
def prepare_latents():
    setup_native()
    from modules.utils import load_vae, load_text_encoder, load_tokenizer
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    LATENTS.mkdir(parents=True, exist_ok=True)
    prepare_norm()
    wait_verified('vae')
    vae = load_vae(str(WEIGHTS / 'vae'), torch.bfloat16, 'cuda').eval()
    mean = torch.tensor(vae.config.latents_mean, device='cuda').view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device='cuda').view(1, -1, 1, 1, 1)
    rows = [r for r in records() if r['split'] == 'train']
    start = time.time()
    for i, row in enumerate(rows):
        check_deadline()
        target = LATENTS / (row['video_id'] + '.pt')
        if target.exists():
            continue
        rgb = read_video(row)
        rgb = np.concatenate([rgb, rgb[-1:]], axis=0)
        x = torch.from_numpy(rgb).cuda().permute(3, 0, 1, 2)[None].to(torch.bfloat16) / 127.5 - 1
        z = vae.encode(x).latent_dist.mode()
        z = ((z.float() - mean) / std).to(torch.bfloat16)
        assert z.shape == (1, 48, 31, 16, 16), z.shape
        torch.save(z[0].cpu(), target)
        if (i + 1) % 5 == 0 or i == len(rows) - 1:
            print(json.dumps({'stage': 'vae_cache', 'videos': i + 1, 'total': len(rows), 'seconds': time.time() - start}), flush=True)
    write_json(OUT / 'cache_complete.json', {'training_videos': len(rows), 'vae_frames': 121, 'repeated_terminal_frame': True, 'test_frames_encoded': 0})
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    wait_verified('text_encoder')
    tokenizer = load_tokenizer(str(WEIGHTS / 'tokenizer'))
    text = load_text_encoder(str(WEIGHTS / 'text_encoder'), torch.bfloat16, 'cuda').eval()
    embeddings = {}
    for key, prompt in PROMPTS.items():
        token = tokenizer([prompt_clean(prompt)], padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        z = text(token.input_ids.cuda(), token.attention_mask.cuda()).last_hidden_state
        z[:, token.attention_mask[0].sum():] = 0
        embeddings[key] = z[0].cpu()
    torch.save(embeddings, OUT / 'text_embeddings.pt')


def wait_verified(component):
    manifest = json.loads((OUT / 'official_weight_manifest.json').read_text())
    required = [f for f in manifest['Data']['Files']
                if f['Type'] == 'blob' and f['Path'].startswith(component + '/')]
    if not required:
        raise ValueError('No registered weight files for ' + component)
    for record in required:
        check_deadline()
        path = WEIGHTS / record['Path']
        if not path.is_file():
            raise FileNotFoundError('Download the pinned base model first: ' + str(path))
        if path.stat().st_size != record['Size'] or sha256(path) != record['Sha256']:
            raise ValueError('Base weight does not match the registered checksum: ' + record['Path'])


class SpatialLatentDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.rows = [r for r in records() if r['split'] == 'train']
        self.norms = json.loads((OUT / 'action_norm.json').read_text())
        self.embeddings = torch.load(OUT / 'text_embeddings.pt', weights_only=True)
        self.cfg_prob = config.cfg_prob

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        z = torch.load(LATENTS / (row['video_id'] + '.pt'), weights_only=True).float()
        a, mask = command_array(row, frames=z.shape[1])
        norm = self.norms[row['category']]
        a = np.clip((a - np.asarray(norm['q01'])) / (np.asarray(norm['q99']) - norm['q01'] + 1e-6) * 2 - 1, -1.5, 1.5) * mask
        a = torch.from_numpy(a.astype(np.float32)).reshape(z.shape[1], 8, 30).permute(2, 0, 1)[..., None]
        mask = torch.from_numpy(mask).reshape(z.shape[1], 8, 30).permute(2, 0, 1)[..., None]
        emb = self.embeddings['empty' if torch.rand(1).item() < self.cfg_prob else row['category']]
        return dict(latents=z, actions=a, actions_mask=mask, text_emb=emb)


def native_trainer_class():
    setup_native()
    source = (UPSTREAM / 'wan_va/train.py').read_text()
    source = source.replace('from dataset import MultiLatentLeRobotDataset',
                            'from experiments.model_expansion.lingbot_spatial import SpatialLatentDataset as MultiLatentLeRobotDataset')
    module = types.ModuleType('spatial_native_lingbot_trainer')
    module.__file__ = str(UPSTREAM / 'wan_va/train.py')
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    original_load = module.load_transformer
    def logged_load(*args, **kwargs):
        training_event('loading_full_transformer')
        model = original_load(*args, **kwargs)
        training_event('full_transformer_loaded', parameters=sum(p.numel() for p in model.parameters()))
        return model
    module.load_transformer = logged_load
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
    def offload_shard(model):
        training_event('cpu_offload_sharding_started')
        kw = dict(mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False),
                  reshard_after_forward=True, offload_policy=CPUOffloadPolicy(pin_memory=True))
        for block in model.blocks:
            fully_shard(block.attn1, **kw)
            fully_shard(block.attn2, **kw)
            fully_shard(block.ffn, **kw)
            fully_shard(block, **kw)
        fully_shard(model, **kw)
        training_event('cpu_offload_sharding_finished')
        return model
    module.shard_model = offload_shard
    module.Trainer.save_checkpoint = save_streaming_checkpoint
    return module.Trainer




@torch.no_grad()
def video_chunk(server, initial, commands, chunk_id):
    """Native video denoising followed by known-command cache update.

    Commands replace the policy's generated actions; no future state is supplied.
    The released block-causal ordering is unchanged (video, then action).
    """
    from utils import data_seq_to_patch
    frame = chunk_id * server.job_config.frame_chunk_size
    if frame == 0:
        server.init_latent = server._encode_obs({'obs': [{'front': initial}]})
    latents = torch.randn(1, 48, 4, 16, 16, device='cuda', dtype=torch.bfloat16)
    server.scheduler.set_timesteps(server.job_config.num_inference_steps)
    steps = F.pad(server.scheduler.timesteps, (0, 1), value=0)
    for i, t in enumerate(steps):
        check_deadline()
        cond = server.init_latent[:, :, :1] if frame == 0 else None
        inp = server._prepare_latent_input(latents, None, t, t, cond, None, frame_st_id=frame)
        pred = server.transformer(server._repeat_input_for_cfg(inp['latent_res_lst']),
            update_cache=2 if i == len(steps) - 1 else 0, cache_name=server.cache_name, action_mode=False)
        if i != len(steps) - 1:
            pred = data_seq_to_patch(server.job_config.patch_size, pred, 4, 16, 16, batch_size=2)
            pred = pred[1:] + server.job_config.guidance_scale * (pred[:1] - pred[1:])
            latents = server.scheduler.step(pred, t, latents, return_dict=False)
        if cond is not None:
            latents[:, :, :1] = cond
    inp = server._prepare_latent_input(None, commands[:, :, frame:frame + 4], frame_st_id=frame)
    server.transformer(server._repeat_input_for_cfg(inp['action_res_lst']), update_cache=2,
                       cache_name=server.cache_name, action_mode=True)
    return latents
