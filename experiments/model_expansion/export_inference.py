"""Export an inference checkpoint matching the registered BF16 linear arithmetic."""
import argparse
import gc
import json
import sys

import numpy as np
import torch
from torch import nn

from .prepare_spatial import DESTINATION, write_json
from .profile_vjepa import UPSTREAM
from .spatial_vjepa import DIM, cache_path, inputs, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', required=True, choices=['H-A', 'H-B'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, str(UPSTREAM))
    from src.models.ac_predictor import vit_ac_predictor
    checkpoint = torch.load(DESTINATION / (args.category + '_dynamics.pt'),
                            map_location='cpu', weights_only=False)
    if checkpoint['epoch'] != checkpoint['registered_epochs']:
        raise ValueError('Inference export requires completed training')
    metadata = checkpoint['metadata']
    predictor = vit_ac_predictor(img_size=(384, 384), patch_size=16,
        num_frames=16, tubelet_size=2, embed_dim=DIM, predictor_embed_dim=1024,
        depth=24, num_heads=16, action_embed_dim=len(metadata['amean']),
        is_frame_causal=True, use_rope=True, use_extrinsics=False).cuda().eval()
    predictor.load_state_dict(checkpoint['predictor'], strict=True)
    linear = {name + '.' + parameter for name, module in predictor.named_modules()
              if isinstance(module, nn.Linear) for parameter in ('weight', 'bias')}
    packed = {key: value.to(torch.bfloat16) if key in linear else value
              for key, value in checkpoint['predictor'].items()}
    row = next(r for r in records(args.category) if r['split'] == 'test')
    feature = torch.tensor(np.array(np.load(cache_path(row))), device='cuda').flatten(0, 1).unsqueeze(0)
    action = inputs(row)[0][:3]
    action = torch.tensor((action - metadata['amean']) / metadata['astd'], device='cuda').unsqueeze(0)
    state = torch.zeros_like(action)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        reference = predictor(feature, action, state)
    predictor.load_state_dict(packed, strict=True)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        restored = predictor(feature, action, state)
    maximum = float((reference - restored).abs().max())
    if not torch.equal(reference, restored):
        raise ValueError('Inference storage changed BF16 forward output: %g' % maximum)
    result = {key: checkpoint[key] for key in ('metadata', 'epoch', 'registered_epochs')}
    result.update(predictor=packed,
                  storage='BF16 linear weights/biases; FP32 normalization; load into FP32 model under BF16 autocast')
    path = DESTINATION / (args.category + '_inference_weights.pt')
    torch.save(result, path)
    report = {'category': args.category, 'file': path.name,
              'bytes': path.stat().st_size, 'prefix_forward_bitwise_equal': True,
              'max_output_difference': maximum, 'validation_frames': 3,
              'full_precision_optimizer_checkpoint': args.category + '_dynamics.pt',
              'compute_precision': 'same registered BF16 autocast; LayerNorm FP32'}
    write_json(DESTINATION / (args.category + '_export.json'), report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
