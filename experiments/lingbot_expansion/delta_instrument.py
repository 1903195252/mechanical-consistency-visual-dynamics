"""Develop a reset-relative RGB instrument on disjoint calibration identities.

The fit sees training image features and training coordinates only. Selection
uses coordinate/derivative error on the registered calibration split, without
commands, forces, energy, or closure losses. Independent verification is a
separate explicit command and cannot change the selected instrument.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from .common import ROOT, OUT, write, sha

DIRECTORY = OUT / 'delta_instrument/no_coordinate_smoothing'
CALIBRATION = ROOT / 'output/closed_chain_eval/visual_calibration'
FEATURES = CALIBRATION / 'optimization/spatial_reader/features/256'
ALPHAS = (.0001, .001, .01, .1, 1.)
# Coordinate filtering can erase genuine rapid model errors. The deployed
# candidate preserves framewise coordinates; differentiation stays explicit.
WINDOWS = (1,)


def rows(role):
    if role == 'train':
        data = json.loads((ROOT/'output/model_expansion/spatial/dataset_manifest.json').read_text())['videos']
        return [r for r in data if r['category']=='H-B' and r['split']=='train' and r['view']=='front']
    data = json.loads((CALIBRATION/'protocol.json').read_text())['spatial_records']
    return [r for r in data if r['category']=='H-B' and r['calibration_role']==role]


def reference(row):
    import mujoco
    folder = ROOT / row['evaluator_directory']
    model = mujoco.MjModel.from_xml_path(str(folder/'model.xml'))
    with np.load(folder/'physics.npz') as d:
        q, v, a = (d[k][np.arange(120)*8].copy() for k in ('qpos','qvel','qacc'))
    if not np.allclose(q[0], model.qpos0, atol=1e-7, rtol=0):
        raise ValueError('Registered reset calibration pose is unavailable')
    return model, q, v, a


def relative_features(features):
    x = np.asarray(features, dtype=np.float32)
    return x-x[0]


def decode(features, nominal, matrix, window):
    q = np.asarray(nominal, dtype=float)+relative_features(features) @ matrix
    if window > 1:
        q = savgol_filter(q, window, 3, axis=0)
    q[:,3:7] /= np.linalg.norm(q[:,3:7], axis=1, keepdims=True)
    return q


def coordinate_errors(q, truth, v, a):
    # Platform translation is the unprojected coordinate retained by Delta IK.
    # Fixed derivative estimator for candidate selection, independent of forces.
    dt = 1/30
    vel = savgol_filter(q[:,:3], 7, 3, deriv=1, delta=dt, axis=0)
    acc = savgol_filter(q[:,:3], 7, 3, deriv=2, delta=dt, axis=0)
    s = slice(8,-8)
    return [float(np.sqrt(np.mean(np.sum(d[s]**2, axis=-1)))) for d in
            (q[:,:3]-truth[:,:3], vel-v[:,:3], acc-a[:,:3])]


def develop():
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    train, development = rows('train'), rows('calibration')
    if set(r['assembly_id'] for r in train)&set(r['assembly_id'] for r in development):
        raise ValueError('Instrument partitions overlap')
    specification = dict(train_ids=[r['video_id'] for r in train],
        development_ids=[r['video_id'] for r in development],
        verification_ids=[r['video_id'] for r in rows('verification')],
        model='linear ridge on reset-relative frozen V-JEPA features',
        alphas=ALPHAS, smoothing_windows=WINDOWS, input_resolution=256,
        selection='max(mean platform position/1mm, velocity/0.01m/s, acceleration/0.1m/s2)',
        test_labels_used=False, force_or_energy_loss=False, closure_loss=False,
        uses_known_CAD_reset=True, source_sha256=sha(__file__))
    write(DIRECTORY/'development_specification.json',specification)
    xs, ys = [], []
    for row in train:
        model, q, _, _ = reference(row)
        xs.append(relative_features(np.load(FEATURES/(row['video_id']+'.npy'))))
        ys.append(q-model.qpos0)
    x = torch.tensor(np.concatenate(xs), dtype=torch.float64, device='cuda')
    y = torch.tensor(np.concatenate(ys), dtype=torch.float64, device='cuda')
    kernel = x @ x.T
    scale = float(kernel.diag().mean())
    eye = torch.eye(len(x), dtype=x.dtype, device=x.device)
    matrices, summaries = {}, {}
    cached = [(r, reference(r), np.load(FEATURES/(r['video_id']+'.npy'))) for r in development]
    for alpha in ALPHAS:
        matrix = (x.T @ torch.linalg.solve(kernel+alpha*scale*eye, y)).cpu().numpy()
        matrices[alpha] = matrix
        for window in WINDOWS:
            key = '%g/%d' % (alpha, window)
            errors = [coordinate_errors(decode(f, ref[0].qpos0, matrix, window), *ref[1:]) for _, ref, f in cached]
            mean = np.mean(errors, axis=0)
            summaries[key] = dict(errors=errors, mean_errors=mean.tolist(),
                objective=float(np.max(mean / [.001,.01,.1])))
        print(json.dumps(dict(alpha=alpha, candidates={k:v['objective'] for k,v in summaries.items() if k.startswith('%g/'%alpha)})),flush=True)
    selected = min(summaries, key=lambda k:summaries[k]['objective'])
    alpha, window = selected.split('/'); matrix=matrices[float(alpha)]
    np.savez_compressed(DIRECTORY/'relative_head.npz', matrix=matrix, window=int(window))
    write(DIRECTORY/'selection.json',dict(selected=selected,candidates=summaries,
        specification_sha256=sha(DIRECTORY/'development_specification.json'),
        checkpoint_sha256=sha(DIRECTORY/'relative_head.npz')))
    # Development readouts are retained; verification is never implicitly run.
    for row, ref, features in cached:
        path = DIRECTORY/'development_readouts'/(row['video_id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,qpos=decode(features,ref[0].qpos0,matrix,int(window)),time=np.arange(120)/30)
    print(json.dumps(dict(selected=selected,mean_errors=summaries[selected]['mean_errors'])),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['develop'])
    args=p.parse_args()
    if args.command=='develop':develop()
