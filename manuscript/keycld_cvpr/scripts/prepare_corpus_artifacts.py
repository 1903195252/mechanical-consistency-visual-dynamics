"""Audit completed multi-assembly records and render evidence-bound paper additions."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.ticker import ScalarFormatter

from prepare_artifacts import ROOT, PAPER, BLUE, ORANGE, GREEN, save
from experiments.keycld_corpus.prepare import point_states

DATA = ROOT / 'output/keycld_corpus'
ARMS = ('video_euler', 'state_euler', 'state_rk4')
SEEDS = (42, 43, 44)
FAMILIES = ('S-A', 'S-B', 'S-J')
CONDITIONS = ('low_speed', 'rated', 'limit_or_load')
METRICS = ('geometry_m', 'velocity_m_s', 'acceleration_m_s2', 'force_balance_Nm', 'work_energy_J', 'prediction_m')
LABELS = {'video_euler': 'Video', 'state_euler': 'State--Euler', 'state_rk4': 'State--RK4'}
NAMES = {'S-A': 'Crank-rocker', 'S-B': 'Crank-slider', 'S-J': 'Double-crank'}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    manifest = read(DATA/'dataset_manifest.json')
    protocol = read(DATA/'protocol.json')
    calibration = read(DATA/'calibration.json')
    assert manifest['counts'] == {'train':813,'calibration':54,'validation':54,'test':108}
    groups = {s:{r['assembly_id'] for r in manifest['records'] if r['split']==s} for s in manifest['counts']}
    assert all(not groups[a]&groups[b] for a in groups for b in groups if a!=b)
    assert len(set.union(*groups.values())) == 343
    assert calibration['passed'] and calibration['protocol_sha256'] == digest(DATA/'protocol.json')
    test = [r for r in manifest['records'] if r['split']=='test']
    expected = {r['case_id'] for r in test}
    identities = {c:sorted({r['assembly_id'] for r in test if r['category']==c}) for c in FAMILIES}
    assert all(len(v)==12 for v in identities.values())
    reports = {}; inference = {}; training = {}; probes = {}; checked = 0
    for arm in ARMS:
        reports[arm] = []; inference[arm] = []; training[arm] = []; probes[arm] = []
        for seed in SEEDS:
            folder = DATA/'runs'/('%s_seed%d'%(arm,seed))
            tr = read(folder/'training_report.json'); inf = read(folder/'inference_report.json'); sc = read(folder/'scores.json')
            assert tr['epochs']==40 and tr['optimizer_updates']==32520
            assert tr['checkpoint_sha256'] == inf['checkpoint_sha256'] == sc['checkpoint_sha256'] == digest(folder/'params.p')
            assert tr['dataset_manifest_sha256'] == inf['dataset_manifest_sha256'] == digest(DATA/'dataset_manifest.json')
            assert sc['calibration_sha256'] == digest(DATA/'calibration.json')
            assert {r['case_id'] for r in sc['rows']} == expected and len(sc['rows'])==108
            assert inf['completed']==162
            inf_rows = {r['case_id']:r for r in inf['records']}
            for r in manifest['records']:
                if r['split'] not in ('test','validation'):continue
                file = folder/'predictions'/r['split']/(r['case_id']+'.npz')
                assert digest(file)==inf_rows[r['case_id']]['prediction_sha256'], str(file)
                checked += 1
            test_timeout = sum(inf_rows[k].get('failure_reason') is not None for k in expected)
            test_finite = sum(inf_rows[k]['finite'] for k in expected)
            inference[arm].append({'seed':seed,'test_finite':test_finite,'test_timeouts':test_timeout,
                                   'validation_timeouts':sum(r.get('failure_reason') is not None for k,r in inf_rows.items() if k not in expected)})
            reports[arm].append(sc); training[arm].append(tr)
            if arm!='video_euler':
                probe=read(folder/'constitutive_probe.json')
                assert probe['checkpoint_sha256']==tr['checkpoint_sha256'] and probe['sampled_states']==648
                probes[arm].append(probe)
    for i in range(3):
        assert len({training[a][i]['initial_checkpoint_sha256'] for a in ARMS})==1
    arrays = {}
    for arm in ('state_euler','state_rk4'):
        data = np.zeros((3,3,12,3,3))
        for si,report in enumerate(reports[arm]):
            lookup = {(r['assembly_id'],r['condition']):r for r in report['rows']}
            for fi,family in enumerate(FAMILIES):
                for ai,identity in enumerate(identities[family]):
                    for ci,condition in enumerate(CONDITIONS):
                        row=lookup[(identity,condition)]
                        assert row['coverage']==1.
                        data[si,fi,ai,ci]=[row['metrics'][k] for k in ('prediction_m','force_balance_Nm','work_energy_J')]
        arrays[arm]=data
    rng=np.random.default_rng(20261012); draws=[]; rawdraws={a:[] for a in arrays}
    for _ in range(2000):
        seed_draw=rng.integers(0,3,3); assembly_draws=[rng.integers(0,12,12) for _ in FAMILIES]
        means={a:np.mean([data[seed_draw,fi][:,assembly_draws[fi]].mean(axis=(0,1,2)) for fi in range(3)],axis=0)
               for a,data in arrays.items()}
        draws.append(means['state_rk4']-means['state_euler'])
        for a in arrays:rawdraws[a].append(means[a])
    differences = {}
    for mi,key in enumerate(('prediction_m','force_balance_Nm','work_energy_J')):
        e=arrays['state_euler'][...,mi].mean();r=arrays['state_rk4'][...,mi].mean()
        differences[key]={'euler_mean':float(e),'rk4_mean':float(r),'rk4_minus_euler':float(r-e),
                          'paired_95pct':np.percentile(np.asarray(draws)[:,mi],[2.5,97.5]).tolist(),
                          'relative_reduction_percent':float(100*(1-r/e))}
    summary = {}
    for arm in ARMS:
        rows=[row for report in reports[arm] for row in report['rows']]
        summary[arm]={'seeds':3,'test_assemblies':36,'predictions':len(rows),
                      'complete_physical_trajectories':sum(r['coverage']==1 for r in rows),
                      'composite_mean':float(np.mean([r['composite_score_100'] for r in reports[arm]])),
                      'joint_pass':sum(r['joint_pass'] for r in rows),
                      'criterion_pass_percent':{k:100*sum(r.get('passed',{}).get(k,False) for r in rows)/len(rows) for k in METRICS},
                      'criterion_measured':{k:sum(r['metrics'][k] is not None for r in rows) for k in METRICS},
                      'validation_point_rmse_m':[r['validation_keypoint_rmse_m'] for r in reports[arm]],
                      'raw_metrics':{},'per_family':{}}
        if arm in arrays:
            for mi,key in enumerate(('prediction_m','force_balance_Nm','work_energy_J')):
                summary[arm]['raw_metrics'][key]={'mean':float(arrays[arm][...,mi].mean()),
                    'median':float(np.median(arrays[arm][...,mi])),
                    'mean_95pct':np.percentile(np.asarray(rawdraws[arm])[:,mi],[2.5,97.5]).tolist()}
        for c in FAMILIES:
            chosen=[r for r in rows if r['category']==c]
            summary[arm]['per_family'][c]={
                'pass_percent':{k:100*sum(r.get('passed',{}).get(k,False) for r in chosen)/len(chosen) for k in METRICS},
                'mean':{k:float(np.mean([r['metrics'][k] for r in chosen if r['metrics'][k] is not None]))
                        if any(r['metrics'][k] is not None for r in chosen) else None for k in METRICS}}
    evidence={'dataset_counts':manifest['counts'],'assembly_splits':manifest['assemblies_by_split'],
              'evaluated_assemblies':343,'mechanism_families':3,'joint_topologies':2,
              'checked_prediction_archives':checked,'equal_initial_checkpoints_within_seed':True,
              'summary':summary,'paired_integrator_changes':differences,
              'inference_coverage':inference,'calibration':{k:v for k,v in calibration.items() if k!='rows'},
              'bootstrap':'2000 resamples of seeds and assembly identities within family; same resample in both state arms, paired conditions retained',
              'kinetic_capacity':{k:v for k,v in read(DATA/'kinetic_capacity_audit.json').items() if k!='rows'},
              'probe_summaries':{a:[{k:v for k,v in p.items() if k!='rows'} for p in probes[a]] for a in probes},
              'threshold_clarification':'The frozen 0.25 Nm threshold is shared and absolute. S-A/S-B actuators are 5 Nm with gain 2; S-J is 12 Nm with gain 12. The registration describes the 5 Nm reference scale imprecisely as a universal actuator bound; no score or threshold is changed.',
              'source':'output/keycld_corpus; original measurements and predictions are not modified'}
    (PAPER/'sources/corpus_evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    return evidence,reports,manifest


def overview(manifest):
    from prepare_rendered_figures import overview as rendered_overview
    rendered_overview(manifest)


def paired_residuals(reports):
    from prepare_integrator_comparison import build
    build()


def tables(evidence,reports):
    from prepare_unified_scoring import planar_estimates
    lines=[]
    continuous = planar_estimates()
    for arm in ARMS:
        s=evidence['summary'][arm];p=s['criterion_pass_percent']
        values=[('N/A' if s['criterion_measured'][k]==0 else '%.2f'%p[k]) for k in METRICS]
        ci=continuous[arm]['ci95']
        lines.append('%s & %d/324 & %s & 0 & %.2f [%.2f, %.2f] \\\\' %
                     (LABELS[arm],s['complete_physical_trajectories'],' & '.join(values),continuous[arm]['mean'],*ci))
    (PAPER/'tables/corpus_score_rows.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for arm in ('state_euler','state_rk4'):
        s=evidence['summary'][arm]
        for c in FAMILIES:
            f=s['per_family'][c];m=f['mean'];p=f['pass_percent']
            lines.append('%s & %s & %.2f & %.3f & %.3f & %.2f & %.2f & 0 \\\\' %
                         (LABELS[arm],NAMES[c],m['prediction_m']*1000,m['force_balance_Nm'],m['work_energy_J'],p['prediction_m'],p['force_balance_Nm']))
    (PAPER/'tables/corpus_family_rows.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for arm in ('state_euler','state_rk4'):
        for p in evidence['probe_summaries'][arm]:
            lines.append('%s & %d & %d/648 & %.2f & %.2f & %.3f \\\\' %
                         (LABELS[arm],p['seed'],p['positive_action_gain_states'],p['effective_inertia_relative_error_median'],
                          p['effective_inertia_relative_error_mean'],p['friction_torque_error_rms_Nm']))
    (PAPER/'tables/corpus_probe_rows.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for arm in ARMS:
        for report,cov in zip(reports[arm],evidence['inference_coverage'][arm]):
            lines.append('%s & %d & %d/108 & %d & %d/108 & %.2f & 0 \\\\' %
                         (LABELS[arm],report['seed'],cov['test_finite'],cov['test_timeouts'],
                          report['complete_physical_trajectories'],continuous[arm]['by_seed'][str(report['seed'])]))
    (PAPER/'tables/corpus_seed_rows.tex').write_text('\n'.join(lines)+'\n')


def main():
    evidence,reports,manifest=audit();overview(manifest);paired_residuals(reports);tables(evidence,reports)
    from prepare_constitutive_figure import build as constitutive_figure
    constitutive_figure()
    print(json.dumps({'audited_prediction_archives':evidence['checked_prediction_archives'],
                      'paired_integrator_changes':evidence['paired_integrator_changes']},indent=2))
    return {'multi_assembly_evidence':'sources/corpus_evidence.json',
            'multi_assembly_sources':'output/keycld_corpus/{dataset_manifest,calibration}.json and runs/*/{training_report,inference_report,scores,constitutive_probe}.json',
            'corpus_overview_selection':'All 343 participating identities: recorded rated-condition isometric frame at t=0; nine enlarged held-out examples; full source hashes in sources/rendered_figure_sources.json',
            'constitutive_figure':'sources/constitutive_figure_data.json: fixed-torque dissipation, pooled usable-gain coverage, reference-only kinetic capacity and both models geometric-readout costs',
            'dataset_coverage':'sources/dataset_coverage.json: all participating assemblies, calibrated SI ranges and paired conditions',
            'corpus_paired_residuals':'sources/integrator_comparison_data.json: paired KeyCLD and CON Euler/RK4 training; native point errors and explicitly labelled geometric-readout CON force/energy; all available pairs retained'}


if __name__=='__main__':main()
