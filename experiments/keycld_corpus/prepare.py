"""Freeze accepted assemblies and prepare calibrated images; never render new data."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from experiments.mechanism_corpus.catalog import ROOT, OUTPUT, connect, write_json

OUT = ROOT / 'output/keycld_corpus'
PROTOCOL = Path(__file__).with_name('protocol.json')
CENTER = np.array([.2, .35])
SCALE = .65


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def point_states(q, v, a, c):
    r, length = c['lengths_m'][:2]
    e = lambda t: np.stack([np.cos(t), np.sin(t)], axis=-1)
    n = lambda t: np.stack([-np.sin(t), np.cos(t)], axis=-1)
    b = r * e(q[:, 0]) + [0, c['pivot_height_m']]
    end = b + length * e(q[:, 1])
    vb = r * n(q[:, 0]) * v[:, :1]
    ve = vb + length * n(q[:, 1]) * v[:, 1:2]
    ab = r * (n(q[:, 0])*a[:, :1] - e(q[:, 0])*v[:, :1]**2)
    ae = ab + length*(n(q[:, 1])*a[:, 1:2] - e(q[:, 1])*v[:, 1:2]**2)
    if c['category'] == 'S-B':
        end = np.column_stack([q[:, 2], np.full(len(q), c['pivot_height_m'] + c['guide_offset_m'])])
        ve = np.column_stack([v[:, 2], np.zeros(len(q))])
        ae = np.column_stack([a[:, 2], np.zeros(len(q))])
    mid = lambda first, second: np.stack([first, second, (first+second)/2], axis=1)
    return mid(b, end), mid(vb, ve), mid(ab, ae)


def prepare_case(task):
    cv2.setNumThreads(1)
    entry, condition, number, split = task
    identity = entry['assembly_id']; folder = OUTPUT / Path(entry['manifest_path']).parent
    video = folder / 'videos' / (identity + '-V%02d.mp4' % number)
    public = json.loads(video.with_suffix('.inputs.json').read_text())
    c = json.loads((folder / 'assembly.json').read_text())['config']
    camera = public['observed_camera_frames'][0]
    R = np.asarray(camera['R_world_to_camera']); t = np.asarray(camera['t_world_to_camera_m'])
    K = np.asarray(camera['K_affine_pixels'])
    world_to_source = K @ np.array([[R[0,0], R[0,2], t[0]], [R[1,0], R[1,2], t[1]], [0,0,1.]])
    pixel_to_world = np.array([[2*SCALE/63,0,CENTER[0]-SCALE], [0,-2*SCALE/63,CENTER[1]+SCALE], [0,0,1.]])
    mapping = world_to_source @ pixel_to_world
    cap = cv2.VideoCapture(str(video)); frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if len(frames) >= 30:
            pass
        warped = cv2.warpPerspective(frame, mapping, (64,64), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(128,128,128))
        frames.append(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != 120:
        raise ValueError('Invalid accepted video: ' + str(video))
    frames = np.asarray(frames, dtype=np.uint8)[30:120]
    cfolder = folder / 'conditions' / condition
    with np.load(cfolder/'commands.npz') as data:
        command = data['speed_command_rad_s'][data['time'] >= 1.]
    assert np.ptp(command) < 1e-10
    servo = public['controller_calibration']
    assert servo['ramp_s'] < 1.
    A = (np.array([0,c['pivot_height_m']])-CENTER)/SCALE
    D = (np.array([c['ground_length_m'],c['pivot_height_m']])-CENTER)/SCALE
    geometry = np.r_[A, D, np.asarray(c['lengths_m'])/SCALE,
                     c.get('guide_offset_m',0.)/SCALE, float(c['category']=='S-B')].astype(np.float32)
    controls = np.array([command[0], servo['speed_gain_Nm_s_rad'], servo['torque_limit_Nm'],
                         *public['known_external_force_world_xz_N']], dtype=np.float32)
    key = identity + '__' + condition
    destination = OUT / 'observations' / split / (key+'.npz')
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 50 if split == 'train' else 90
    np.savez_compressed(destination, x=frames[:count], t=np.arange(count,dtype=np.float32)/30,
                        geometry=geometry, action=controls)
    inputs = OUT / 'inputs' / split / (key+'.npz'); inputs.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(inputs, x=frames[:3], t=np.arange(90,dtype=np.float32)/30,
                        geometry=geometry, action=controls)
    # Privileged state labels are separate files, opened only by state arms / evaluator.
    with np.load(cfolder/'physics.npz') as data:
        ix = np.arange(240, 960, 8)
        x,v,a = point_states(data['q'][ix],data['qdot'][ix],data['qddot'][ix],c)
    state = OUT / 'state_labels' / split / (key+'.npz'); state.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(state, x=((x-CENTER)/SCALE).astype(np.float32), v=(v/SCALE).astype(np.float32),
                        a=(a/SCALE).astype(np.float32))
    return {'case_id':key, 'assembly_id':identity, 'category':entry['category'], 'condition':condition,
            'split':split, 'source_video':str(video.relative_to(ROOT)), 'source_video_sha256':sha(video),
            'observations':str(destination.relative_to(OUT)), 'observations_sha256':sha(destination),
            'inputs':str(inputs.relative_to(OUT)), 'inputs_sha256':sha(inputs),
            'state_labels':str(state.relative_to(OUT)), 'state_labels_sha256':sha(state),
            'physical_parameters':str((cfolder/'physical_parameters.json').relative_to(ROOT)),
            'physical_parameters_sha256':sha(cfolder/'physical_parameters.json'),
            'physics_reference':str((cfolder/'physics.npz').relative_to(ROOT)),
            'physics_reference_sha256':sha(cfolder/'physics.npz')}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'dataset_manifest.json').exists():
        raise RuntimeError('Registered dataset already exists; do not overwrite its selection')
    protocol = json.loads(PROTOCOL.read_text()); shutil.copy2(PROTOCOL,OUT/'protocol.json')
    db = connect(); entries = [dict(r) for r in db.execute('SELECT * FROM assemblies ORDER BY assembly_id')]; db.close()
    census = []
    tasks = []
    for e in entries:
        supported = e['category'] in protocol['dataset']['keycld_supported_categories']
        census.append(dict(e, keycld_supported=supported,
                           reason=protocol['unsupported'].get(e['category'])))
        if not supported:
            continue
        if e['split'] in ('test','validation') and e['status'] != 'verified':
            raise RuntimeError('Held-out assembly unavailable: ' + e['assembly_id'])
        if e['status'] != 'verified':
            continue
        split = ('calibration' if e['ordinal'] <= 102 else 'validation') if e['split']=='validation' else e['split']
        for condition, number in zip(protocol['dataset']['conditions'], (1,6,11)):
            tasks.append((e, condition, number, split))
    write_json(OUT/'assembly_census.json', {'assembly_count':len(census), 'records':census})
    start = time.monotonic(); records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        for record in pool.map(prepare_case,tasks):
            records.append(record)
            if len(records)%30 == 0:
                write_json(OUT/'status.json', {'stage':'preparing_existing_videos', 'completed':len(records),
                                              'total':len(tasks), 'elapsed_s':time.monotonic()-start})
                print(len(records), '/', len(tasks), flush=True)
    import collections
    manifest = {'protocol_sha256':sha(OUT/'protocol.json'), 'preparation_unix_s':time.time(),
                'records':records, 'counts':dict(collections.Counter(r['split'] for r in records)),
                'assemblies_by_split':{s:len({r['assembly_id'] for r in records if r['split']==s}) for s in {r['split'] for r in records}},
                'source_corpus_assemblies':700, 'supported_assemblies':360,
                'views_counted_as_independent_samples':False, 'data_generation_resumed':False}
    groups = {s:{r['assembly_id'] for r in records if r['split']==s} for s in manifest['counts']}
    assert all(not groups[a] & groups[b] for a in groups for b in groups if a!=b)
    assert manifest['counts']['test']==108
    write_json(OUT/'dataset_manifest.json', manifest)
    write_json(OUT/'status.json', {'stage':'data_prepared', 'counts':manifest['counts'],
                                  'assemblies_by_split':manifest['assemblies_by_split']})
    print(json.dumps({k:v for k,v in manifest.items() if k!='records'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
