"""Render V-JEPA and LingBot spatial audits without rerunning experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.text import Text
from prepare_artifacts import ROOT, PAPER, GREEN
from prepare_rendered_figures import export

DATA = ROOT / 'output/vjepa_expansion/seed_42/calibrated'
PURPLE = '#8064A2'
GRAY = '#677780'
FAMILIES = ('H-A', 'H-B')
NAMES = ('Stewart', 'Delta')
COLORS = ('#176B78', '#D56E4C')
FIGURE_SIZE = (7.05, 2.55)




def collect():
    from prepare_vjepa_expansion import collect as expanded
    d=expanded()
    summary=json.loads((ROOT/'output/vjepa_expansion/summary.json').read_text())
    manifest=json.loads((ROOT/'output/vjepa_expansion/dataset_manifest.json').read_text())['videos']
    rows={f:d['groups'][f+'_calibrated'][0]['rows'] for f in FAMILIES}
    return summary,d['protocol'],manifest,rows




def read_frame(row, frame=0):
    cap = cv2.VideoCapture(str(ROOT / row['video']))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, rgb = cap.read()
    cap.release()
    assert ok, row['video']
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def skeleton(audit, q):
    """Actual tree endpoints and platform sockets, before geometric correction."""
    from experiments.mechanism_corpus import delta

    states = audit.tree.forward(q, np.zeros(audit.model.nv))
    platform = states[audit.names['platform']]
    sockets, tips, links = [], [], []
    if audit.kind == 'H-A':
        for i, (base, anchor) in enumerate(zip(audit.c['base_anchors_m'],
                                              audit.c['platform_anchors_m'])):
            tip = states[audit.names['rod_%d' % i]]['position']
            sockets.append(audit.tree.point(platform, anchor)[0])
            tips.append(tip)
            links.append([base, tip])
    else:
        for i, j, elbow, anchor in delta.anchor_pairs(audit.c):
            upper = states[audit.names['upper_%d' % i]]
            joint = audit.tree.point(upper, elbow)[0]
            tip = audit.tree.point(states[audit.names['rod_%d_%d' % (i, j)]],
                                   [0, 0, audit.c['forearm_length_m']])[0]
            sockets.append(audit.tree.point(platform, anchor)[0])
            tips.append(tip)
            links.extend([[upper['position'], joint], [joint, tip]])
    sockets, tips = np.asarray(sockets), np.asarray(tips)
    # Verify that the displayed gaps are the same residuals used by the SI audit.
    np.testing.assert_allclose(tips - sockets,
                               audit.closure(q)[0].reshape(-1, 3), atol=1e-7)
    # Platform edges connect sockets around their physical platform perimeter.
    local = (sockets - platform['position']) @ platform['rotation']
    order = np.argsort(np.arctan2(local[:, 1], local[:, 0]))
    outline = sockets[np.r_[order, order[0]]]
    return {'sockets': sockets, 'tips': tips, 'links': np.asarray(links),
            'outline': outline}, states




def rollout_example(row):
    prediction_path = DATA / 'predictions' / row['category'] / (row['video_id'] + '.npz')
    physics_path = ROOT / row['evaluator_directory'] / 'physics.npz'
    with np.load(prediction_path) as data:
        time, prediction = data['time'], data['qpos'][:, 2] * 1000.
        observed = int(data['observed_frames'])
    with np.load(physics_path) as data:
        indices = np.arange(120) * 8
        np.testing.assert_allclose(data['time'][indices], time, atol=1e-10)
        reference = data['qpos'][indices, 2] * 1000.
    assert observed == 3 and len(time) == 120
    ranges = {'reference': float(np.ptp(reference[observed:])),
              'native': float(np.ptp(prediction[observed:]))}
    assert ranges['reference'] > 0
    return {'video_id': row['video_id'], 'category': row['category'],
            'prediction_source': str(prediction_path.relative_to(ROOT)),
            'physics_source': str(physics_path.relative_to(ROOT)),
            'time_s': time.tolist(), 'reference_z_mm': reference.tolist(),
            'native_z_mm': prediction.tolist(), 'last_observed_time_s': float(time[2]),
            'future_first_frame': observed, 'future_frame_count': len(time) - observed,
            'future_peak_to_peak_mm': ranges,
            'future_range_ratio_percent': 100 * ranges['native'] / ranges['reference']}






def style_axis(ax):
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    ax.grid(axis='y', color='#E5E9EC', lw=.45, zorder=0)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#85949C')
        ax.spines[side].set_linewidth(.6)










def draw_lingbot_overlay(ax, rgb, record):
    ax.imshow(rgb,extent=(-.5,639.5,639.5,-.5),interpolation='none')
    camera=record['camera']
    rotation=np.asarray(camera['R_world_to_camera'])
    translation=np.asarray(camera['t_world_to_camera_m'])
    intrinsic=np.asarray(camera['K_affine_pixels'])
    def pixels(points):
        p=np.asarray(points)@rotation.T+translation
        return p[...,:2]@intrinsic[:2,:2].T+intrinsic[:2,2]
    for key,color,style in [('native','#D55E00','-'),('reconstructed','#009E73',(0,(2.5,1.6)))]:
        for segment in record[key]['links']+[record[key]['outline']]:
            uv=pixels(segment)
            line,=ax.plot(uv[:,0],uv[:,1],color=color,ls=style,lw=.85,zorder=3)
            line.set_path_effects([pe.Stroke(linewidth=1.15,foreground='white',alpha=.65),pe.Normal()])
    ax.set(xlim=(-.5,639.5),ylim=(639.5,-.5),xticks=[],yticks=[])
    for spine in ax.spines.values():spine.set_visible(False)






def training_views(manifest):
    views = ('front', 'side', 'top', 'isometric', 'tracking_closeup')
    titles = ('Front', 'Side', 'Top', 'Isometric', 'Tracking close-up')
    fig, axes = plt.subplots(2, 5, figsize=(7.05, 2.9))
    selected = []
    for i, (category, name) in enumerate((('H-A', 'Stewart'), ('H-B', 'Delta'))):
        first = next(r for r in manifest if r['category'] == category and r['split'] == 'train')
        candidates = {r['view']: r for r in manifest
                      if r['assembly_id'] == first['assembly_id']
                      and r['condition'] == first['condition']}
        assert set(candidates) == set(views)
        for j, view in enumerate(views):
            row = candidates[view]
            ax = axes[i, j]
            ax.imshow(read_frame(row))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(titles[j], fontsize=7.5, pad=4)
            if j == 0:
                ax.set_ylabel(f'{category} · {name}', fontsize=8, weight='bold', labelpad=5)
            selected.append({key: row[key] for key in
                             ('assembly_id', 'video_id', 'condition', 'view', 'split', 'video')})
    fig.subplots_adjust(left=.045, right=.998, bottom=.01, top=.91, wspace=.025, hspace=.035)
    export(fig, 'spatial_vjepa_overview')
    return selected


def traces(manifest, rows):
    lookup = {r['video_id']: r for r in manifest}
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 3.1))
    for i, c in enumerate(('H-A', 'H-B')):
        row = lookup[rows[c][0]['video_id']]
        with np.load(ROOT / row['evaluator_directory'] / 'physics.npz') as z:
            truth = z['qpos'][np.arange(120) * 8, :3]
        with np.load(DATA / 'predictions' / c / (row['video_id'] + '.npz')) as z:
            t, q = z['time'], z['qpos'][:, :3]
        for j, ax in enumerate(axes[i]):
            ax.plot(t, truth[:, j] * 1000, color='#242B33', lw=1.2, label='Reference')
            ax.plot(t, q[:, j] * 1000, color=PURPLE, lw=1.2, label='V-JEPA 2.1 + AC')
            ax.axvline(2 / 30, color=GRAY, ls=':', lw=.8)
            ax.set_title(f'{c} · {NAMES[i]}: {"xyz"[j]}', loc='left')
            ax.set_ylabel('Platform (mm)'); ax.grid(alpha=.15)
            if i == 1: ax.set_xlabel('Time (s)')
    axes[0, 0].legend(fontsize=6, loc='best')
    fig.tight_layout(pad=.4)
    export(fig, 'spatial_vjepa_traces')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-only', action='store_true')
    parser.add_argument('--tables-only', action='store_true')
    args = parser.parse_args()
    from prepare_spatial_expansion import export_tables
    export_tables()
    if not args.tables_only:
        from render_spatial_expansion import figure as expanded_figure
        expanded_figure()
    if not args.main_only and not args.tables_only:
        summary, protocol, manifest, rows = collect()
        views = training_views(manifest)
        traces(manifest, rows)
        evidence = dict(summary=summary,protocol=protocol,training_view_examples=views,
            summary_source='output/vjepa_expansion/summary.json',
            protocol_source='output/vjepa_expansion/protocol.json',
            scope='V-JEPA + AC: 50 test assemblies per family, three conditions, three independently trained seeds; shared test cohort with LingBot')
        (PAPER/'sources/spatial_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
