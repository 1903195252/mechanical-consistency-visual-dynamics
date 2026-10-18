"""Paired 64/256-pixel evidence, with a common SI definition and explicit support.

This is an aggregation/export step: it never trains, forecasts, or rescales targets.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
EXT = ROOT / 'output/con_corpus/dynamics_extension'
REC = ROOT / 'output/planar_resolution/cloud/recovered_20261017'
REPORTS = REC / 'comparison/source_reports'
METRICS = ('geometry', 'constraint_velocity', 'constraint_acceleration', 'dynamics', 'energy', 'prediction')
FIELDS = ('geometry_m', 'velocity_m_s', 'acceleration_m_s2', 'force_balance_Nm', 'work_energy_J', 'prediction_m')
LIMITS = (.001, .01, .1, .25, .05, .01)
SOURCES = {}


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def balanced(rows, field):
    """Finite-case means, equally weighted by seed, family and condition."""
    groups = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None and np.isfinite(value):
            groups[(row['seed'], row['family'], row['condition'])].append(value)
    seeds = defaultdict(list)
    for (seed, _, _), values in groups.items():
        seeds[seed].append(float(np.mean(values)))
    return float(np.mean([np.mean(values) for values in seeds.values()])) if seeds else None


def canonical(row, seed):
    result = dict(case_id=row['case_id'], seed=seed, family=row['category'], condition=row['condition'],
                  complete=row['coverage'] == 1, joint=row['joint_pass'], score=row['continuous_score_100'])
    for key, field, tol in zip(METRICS, FIELDS, LIMITS):
        result[key] = row['metrics'][field]
        result[key + '_score'] = row['continuous_components_100'][field]
        result[key + '_pass'] = bool(row['passed'][field])
        result[key + '_ratio'] = row['metrics'][field] / tol if row['metrics'][field] is not None else None
    return result


def summarize(rows):
    seeds = [balanced([r for r in rows if r['seed'] == s], 'score') for s in (42, 43, 44)]
    components = {k: balanced(rows, k + '_score') for k in METRICS}
    result = dict(total=len(rows), coverage=sum(r['complete'] for r in rows),
                  joint=sum(r['joint'] for r in rows), score=balanced(rows, 'score'),
                  seed_scores=seeds, seed_sd=float(np.std(seeds, ddof=1)),
                  components=components, means={k: balanced(rows, k) for k in METRICS},
                  pass_rates={k: 100 * balanced(rows, k + '_pass') for k in METRICS}, rows=rows)
    np.testing.assert_allclose(result['score'], np.mean(list(components.values())), rtol=1e-12)
    assert len(rows) == 324 and len({(r['seed'], r['case_id']) for r in rows}) == 324
    return result


def localization_comparison():
    """The same unprojected Euclidean metric on future frames 3:90 at both sizes."""
    report64=read(ROOT/'output/closed_chain_eval/visual_calibration/paired_robust_planar_rgb/summary.json')
    result={}
    reference_ids=None
    for width in (64,256):
        result[str(width)]={}
        for model in ('reference','KeyCLD','CON'):
            means=[]; counts=[]; cases=[]
            for seed in ((42,) if model=='reference' else (42,43,44)):
                if width==64:
                    name='reference' if model=='reference' else f'{model.lower()}_seed{seed}'
                    rows=[r for r in report64['rows'] if r['model']==name]
                    field='point_euclidean_rmse_m'
                    assert all(r['geometry_and_point_frames']=='all future frames 3:90' for r in rows if r['finite'])
                else:
                    run_model='keycld' if model=='reference' else model.lower()
                    report=read(REPORTS/f'{run_model}_256_seed{seed}/external_rgb/report.json')
                    rows=[r for r in report['rows'] if r['input']==('reference' if model=='reference' else 'generated')]
                    field='future_point_euclidean_rmse_m'
                assert len(rows)==108
                ids=sorted(r['case_id'] for r in rows)
                if reference_ids is None:reference_ids=ids
                assert ids==reference_ids
                finite=[r for r in rows if r['finite']]
                means.append(1000*float(np.mean([r[field] for r in finite])))
                counts.append(len(finite));cases.extend(ids)
            result[str(width)][model]=dict(point_euclidean_rmse_mm=float(np.mean(means)),
                seed_means_mm=means,finite_by_seed=counts,unique_cases=len(set(cases)))
    for relative in ('experiments/closed_chain_eval/remeasure_planar_rgb.py','experiments/planar_resolution/measure_rgb.py'):
        SOURCES[relative]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    return dict(future_frames='3:90 in the cropped 90-frame sequence (87 frames; 1.100–3.967 s)',
        metric='Per-trajectory RMS of Euclidean B,C,P error, then finite-case means and equal seed weighting; no reconstruction.',
        reference_support='108 identical test trajectories at each resolution, counted once.',
        configurations='Readers are trained and calibrated separately for each resolution.',results=result)


def collect():
    SOURCES.clear()
    summary = read(EXT / 'summary.json')
    path = ROOT / 'output/closed_chain_eval/continuous/case_scores.csv'
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as stream:
        old = list(csv.DictReader(stream))
    corrections = read(REC / 'comparison/corrections_geek.json')
    runs = []
    for model, arm in (('KeyCLD', 'video_euler'), ('CON', 'video_con_iae')):
        for width in (64, 256):
            tracks = {}; visual = []; correction = []
            if width == 64:
                for track, native in (('native', 'raw'), ('reconstructed', 'conditioned')):
                    rows = []
                    for r in old:
                        if r['suite'] != 'planar-servo' or r['arm'] != arm or r['track'] != native:
                            continue
                        row = dict(case_id=r['case_id'], seed=int(r['seed']), family=r['family'], condition=r['condition'],
                                   complete=r['complete'] == 'True', joint=r['strict_joint_pass'] == 'True', score=float(r['continuous_score_100']))
                        for k in METRICS:
                            row[k] = float(r[k + '_SI']) if r[k + '_SI'] else None
                            row[k + '_score'] = float(r[k + '_score_100'])
                            row[k + '_ratio'] = float(r[k + '_ratio']) if r[k + '_ratio'] else None
                            row[k + '_pass'] = row[k + '_ratio'] is not None and row[k + '_ratio'] <= 1
                        rows.append(row)
                    tracks[track] = summarize(rows)
                for seed in (42, 43, 44):
                    report = read(EXT / 'video_diagnosis' / f'{model.lower()}_seed{seed}/report.json')
                    for r in report['rows']:
                        visual.append(dict(seed=seed, family=r['category'], condition=r['condition'], case_id=r['case_id'],
                            rgb_mse=r['image_mse']['prefix_forecast'], motion_mask_mse=r['motion_mask_mse']['prefix_forecast'],
                            persistence_rgb_mse=r['image_mse']['persistence'], persistence_motion_mask_mse=r['motion_mask_mse']['persistence']))
                    report = read(EXT / 'physics_readout' / f'{model.lower()}_{arm}_seed{seed}/report.json')
                    correction += [dict(seed=seed, family=r['category'], condition=r['condition'], case_id=r['case_id'],
                                        correction=r['correction_rms_m']) for r in report['rows']]
            else:
                track_rows = {k: [] for k in ('native', 'reconstructed')}
                for seed in (42, 43, 44):
                    run = f'{model.lower()}_256_seed{seed}'
                    report = read(REPORTS / run / 'scores.json')
                    training = read(REPORTS / run / 'training_report.json')
                    inference = read(REPORTS / run / 'inference_report.json')
                    assert report['scoring_version'] == 'continuous-si-1'
                    assert training['full_objective'] and training['scientific_training_result']
                    assert training['epochs'] == (40 if model == 'KeyCLD' else 50)
                    assert training.get('optimizer_updates', training.get('updates')) == (32520 if model == 'KeyCLD' else 2550)
                    assert report['checkpoint_sha256'] == training['checkpoint_sha256'] == inference['checkpoint_sha256']
                    assert inference['completed'] == 162 and inference['finite'] == 162
                    for track in track_rows:
                        track_rows[track] += [canonical(r, seed) for r in report[track]['rows']]
                    visual += [dict(r, seed=seed, family=r['category']) for r in report['visual']['rows']]
                    if seed == 44:
                        folder = REC / 'd6000/expanded/output/planar_resolution/runs' / run / 'measurements'
                        measured = [read(p) for p in sorted(folder.glob('*.json'))]
                    else:
                        measured = corrections[run]
                    assert len(measured) == 108
                    correction += [dict(seed=seed, family=r['category'], condition=r['condition'], case_id=r['case_id'],
                                        correction=r['correction_rms_m']) for r in measured]
                tracks = {k: summarize(rows) for k, rows in track_rows.items()}
            v = {k: balanced(visual, k) for k in ('rgb_mse', 'motion_mask_mse', 'persistence_rgb_mse', 'persistence_motion_mask_mse')}
            v['seed_rgb'] = [balanced([r for r in visual if r['seed'] == s], 'rgb_mse') for s in (42, 43, 44)]
            v['seed_motion'] = [balanced([r for r in visual if r['seed'] == s], 'motion_mask_mse') for s in (42, 43, 44)]
            v['rgb_sd'] = float(np.std(v['seed_rgb'], ddof=1))
            v['motion_sd'] = float(np.std(v['seed_motion'], ddof=1))
            v['rgb_hold_ratio'] = v['rgb_mse'] / v['persistence_rgb_mse']
            v['motion_hold_ratio'] = v['motion_mask_mse'] / v['persistence_motion_mask_mse']
            finite = int(sum(r['rgb_mse'] is not None and np.isfinite(r['rgb_mse']) for r in visual))
            runs.append(dict(model=model, width=width, arm=arm, finite=finite, visual=v, **tracks,
                             correction_mm=balanced(correction, 'correction') * 1000,
                             corrections_mm=[r['correction'] * 1000 for r in correction if r['correction'] is not None]))
    reader = {}
    for model in ('KeyCLD', 'CON'):
        for source in ('reference', 'generated'):
            for method in ('raw_cartesian_spline5', 'cartesian_spline5', 'phase_sg7d3'):
                rows = []
                for seed in (42, 43, 44):
                    report = read(REPORTS / f'{model.lower()}_256_seed{seed}/external_rgb/report.json')
                    block = report['summary'][source][method]
                    rows += [canonical(r, seed) for r in block['rows']]
                reader[f'{model}/{source}/{method}'] = summarize(rows)
    # References are repeated by the six model/seed evaluations, not new trajectories.
    for source in ('KeyCLD', 'CON'):
        for method in ('raw_cartesian_spline5', 'cartesian_spline5', 'phase_sg7d3'):
            block = reader[f'{source}/reference/{method}']
            for k in ('score', *METRICS):
                by_seed = [balanced([r for r in block['rows'] if r['seed'] == s], k) for s in (42, 43, 44)]
                # The two FP32 backends have an independently measured submicron
                # coordinate difference; derivative summaries retain that noise.
                np.testing.assert_allclose(by_seed, [by_seed[0]] * 3, rtol=5e-5, atol=1e-6)
    localization=localization_comparison()
    return dict(version='paired-resolution-1', scoring_version='continuous-si-1', runs=runs,
                reader=reader, reader_localization=localization,
                state_controls='Shared resolution-independent frozen checkpoints',
                aggregation='Equal seeds, families and conditions; finite-case SI means; missing scores are zero. SD is across three seed means.',
                sources=dict(SOURCES))


def row(*cells):
    return ' & '.join(map(str, cells)) + r' \\'


def export():
    data = collect()
    summary = json.loads((EXT / 'summary.json').read_text())
    from prepare_unified_scoring import planar_estimates
    state = planar_estimates()
    native = []; corrected = []; seed_rows = []; components = []
    for model in ('KeyCLD', 'CON'):
        if native:
            native += [r'\midrule', r'\multicolumn{10}{l}{\textit{CON}}\\']
            corrected += [r'\midrule', r'\multicolumn{6}{l}{\textit{CON}}\\']
        for run in [r for r in data['runs'] if r['model'] == model]:
            n, c, v = run['native'], run['reconstructed'], run['visual']
            native.append(row(f"Video {run['width']} px", f"{run['finite']}/324", f"{v['rgb_mse']*1e4:.2f}",
                f"{n['coverage']}/324", '/'.join(f"{n['pass_rates'][k]:.1f}" for k in METRICS[:3]), 'N/A', 'N/A',
                f"{n['means']['prediction']*1000:.1f}", '0', f"${n['score']:.2f}\pm{n['seed_sd']:.2f}$"))
            corrected.append(row(f"Video {run['width']} px", f"{c['coverage']}/324", f"{run['correction_mm']:.2f}",
                f"{c['means']['dynamics']:.3f}", f"{c['means']['energy']:.3f}", f"{c['means']['prediction']*1000:.2f}"))
            components.append(row(model, run['width'], *(f"{n['components'][k]:.2f}" for k in METRICS),
                                  f"{n['score']:.2f}", f"{c['score']:.2f}"))
            for i, seed in enumerate((42, 43, 44)):
                seed_rows.append(row(model, run['width'], seed, f"{v['seed_rgb'][i]*1e4:.4f}", f"{v['seed_motion'][i]*1e3:.4f}",
                                     f"{n['seed_scores'][i]:.2f}", f"{c['seed_scores'][i]:.2f}"))
        arms = [('state_euler', 'State--Euler'), ('state_rk4', 'State--RK4')] if model == 'KeyCLD' else [
            ('state_con_iae', 'State--Dopri5'), ('state_euler_con_iae', 'State--Euler'), ('state_rk4_con_iae', 'State--RK4')]
        for arm, name in arms:
            group = 'con_integrator' if model == 'CON' and arm != 'state_con_iae' else model.lower()
            s = summary['physics'][group + '/' + arm]; n = s['raw']; c = s['conditioned']; means = n['metrics']
            comp = n['component_scores_100']; est = state[arm]
            sd = np.std(list(est['by_seed'].values()), ddof=1)
            native.append(row(name, '324/324', '--', f"{n['complete']}/324", '/'.join(f'{comp[k]:.1f}' for k in FIELDS[:3]),
                *(f'{comp[k]:.2f}' if means[k]['valid'] else 'N/A' for k in FIELDS[3:5]),
                f"{means['prediction_m']['mean']*1000:.1f}", '0', f"${est['mean']:.2f}\pm{sd:.2f}$"))
            correction = s['correction_rms_m']['mean'] * 1000
            corrected.append(row(name, f"{c['complete']}/324", r'$<0.001$' if correction < .001 else f'{correction:.2f}',
                f"{c['metrics']['force_balance_Nm']['mean']:.3f}", f"{c['metrics']['work_energy_J']['mean']:.3f}", f"{c['metrics']['prediction_m']['mean']*1000:.2f}"))
    merged = []
    for native_line, corrected_line in zip(native, corrected):
        if ' & ' not in native_line:
            merged.append(native_line.replace('{10}', '{15}'))
            continue
        n = native_line.rstrip('\\ ').split(' & ')
        c = corrected_line.rstrip('\\ ').split(' & ')
        assert n[0] == c[0]
        # All three count columns share the /324 denominator in the caption.
        n[1], n[3], c[1] = (v.split('/')[0] for v in (n[1], n[3], c[1]))
        n[4] = '/'.join(f'{float(v):g}' for v in n[4].split('/'))
        merged.append(row(*(n + c[1:])))
    for name, lines in [('native_comparison_main_rows', native), ('dynamics_readout_main_rows', corrected),
                         ('planar_audit_main_rows', merged),
                         ('resolution_seed_rows', seed_rows), ('resolution_component_rows', components)]:
        (PAPER / 'tables' / (name + '.tex')).write_text('\n'.join(lines) + '\n')
    lines = []
    for model, source, label, divisor in [('KeyCLD', 'reference', 'Correct RGB', 3),
                                         ('KeyCLD', 'generated', 'KeyCLD RGB', 1), ('CON', 'generated', 'CON RGB', 1)]:
        for method, name in [('raw_cartesian_spline5', 'Cartesian'), ('cartesian_spline5', 'Projected Cartesian'), ('phase_sg7d3', 'Projected phase')]:
            block = data['reader'][f'{model}/{source}/{method}']; m = block['means']
            lines.append(row(label, name, f"{block['score']:.2f}", f"{block['coverage']//divisor}/{block['total']//divisor}",
                f"{block['joint']//divisor}/{block['total']//divisor}", f"{m['prediction']*1000:.3f}",
                f"{m['dynamics']:.3f}" if m['dynamics'] is not None else 'N/A',
                f"{m['energy']:.3f}" if m['energy'] is not None else 'N/A'))
    (PAPER / 'tables/resolution_reader_rows.tex').write_text('\n'.join(lines) + '\n')
    (PAPER / 'sources/resolution_comparison.json').write_text(json.dumps(data, indent=2) + '\n')
    return data


if __name__ == '__main__':
    data = export()
    print(json.dumps([dict({k: r[k] for k in ('model', 'width', 'finite', 'correction_mm')},
                      native=r['native']['score'], reconstructed=r['reconstructed']['score'], visual=r['visual'])
                     for r in data['runs']], indent=2))
