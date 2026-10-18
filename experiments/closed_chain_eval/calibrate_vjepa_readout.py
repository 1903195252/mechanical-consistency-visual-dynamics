"""Calibrate reset-relative Delta coordinates at V-JEPA's 384-pixel input.

The existing ridge protocol is fitted on training videos and selected using
coordinate/derivative errors on calibration identities. Verification is disjoint.
No forecasting checkpoint, test trajectory, force, or energy enters this fit.
"""
import json
import numpy as np
from experiments.lingbot_expansion.delta_instrument import (
    rows, reference, relative_features, coordinate_errors, decode, ALPHAS)
from .refine_spatial_reader import DIRECTORY as FEATURES
from .visual_calibration import ROOT, OUT, write, sha

DIRECTORY = OUT/'vjepa_readout_replay/calibration'


def run():
    train, development, verification = [rows(role) for role in
                                       ('train', 'calibration', 'verification')]
    identities = [set(r['assembly_id'] for r in group)
                  for group in (train, development, verification)]
    assert all(not identities[i] & identities[j]
               for i in range(3) for j in range(i+1, 3))
    sources = {}
    def features(row):
        path = FEATURES/'features/384'/(row['video_id']+'.npy')
        sources[str(path.relative_to(ROOT))] = sha(path)
        return np.load(path)
    xs, ys = [], []
    for row in train:
        model, q, _, _ = reference(row)
        xs.append(relative_features(features(row)))
        ys.append(q-model.qpos0)
    x, y = np.concatenate(xs).astype(float), np.concatenate(ys)
    kernel = x @ x.T
    scale = np.trace(kernel)/len(kernel)
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    projected = eigenvectors.T @ y
    cached = [(r, reference(r), features(r)) for r in development]
    candidates, matrices = {}, {}
    for alpha in ALPHAS:
        matrix = x.T @ (eigenvectors @ (projected/(eigenvalues[:, None]+alpha*scale)))
        errors = [coordinate_errors(decode(h, ref[0].qpos0, matrix, 1), *ref[1:])
                  for _, ref, h in cached]
        mean = np.mean(errors, axis=0)
        candidates[str(alpha)] = dict(mean_errors=mean.tolist(), per_case_errors=errors,
                                     objective=float(np.max(mean/[.001, .01, .1])))
        matrices[str(alpha)] = matrix
    chosen = min(candidates, key=lambda k:candidates[k]['objective'])
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY/'H-B_relative_head.npz'
    np.savez_compressed(path, matrix=matrices[chosen], window=1)
    write(DIRECTORY/'selection.json', dict(alpha=chosen, candidates=candidates,
        train_ids=[r['video_id'] for r in train],
        development_ids=[r['video_id'] for r in development],
        encoder_input_resolution=384, coordinate_smoothing=False,
        test_labels_used=False, force_energy_closure_losses=False,
        checkpoint_sha256=sha(path), source_sha256=sha(__file__), feature_sources=sources))
    # Open verification only after the selected checkpoint has been frozen.
    verified = []
    for row in verification:
        model, q, v, a = reference(row)
        predicted = decode(features(row), model.qpos0, matrices[chosen], 1)
        verified.append(dict(video_id=row['video_id'],
            errors=coordinate_errors(predicted, q, v, a)))
        target = DIRECTORY/'verification/predictions/H-B'/(row['video_id']+'.npz')
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, time=np.arange(120)/30, qpos=predicted)
    report = dict(selected_alpha=chosen, cases=verified,
                  mean_errors=np.mean([r['errors'] for r in verified], axis=0).tolist(),
                  checkpoint_sha256=sha(path), features=sources)
    write(DIRECTORY/'verification.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    run()
