"""Verified 0/50-update LingBot evidence for every manuscript consumer."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/lingbot_expansion'
RECOVERED = BASE / 'recovered_20261017'
RUNS = RECOVERED / 'geek/runs'
REF = RECOVERED / 'pro/reference'
FAMILIES = ('H-A', 'H-B')
SEEDS = (42, 43, 44)
STEPS = (0, 50)
METRICS = ('geometry','constraint_velocity','constraint_acceleration','dynamics','energy','prediction')
FIELDS = ('geometry_max_mm','velocity_constraint_max_m_s','acceleration_constraint_max_m_s2',
          'effort_rmse','work_energy_max_J','point_rmse_mm')
SCALES = (.001, 1, 1, 1, 1, .001)


def mean(values):
    a = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(a)) if a else None


def stats(values):
    return dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)))


def texrow(*cells):
    return ' & '.join(map(str, cells)) + r' \\'


def components(row, branch):
    b = row.get(branch, {})
    result = {}
    for k, field, scale in zip(METRICS, FIELDS, SCALES):
        v = b.get(field)
        result[k] = 0. if v is None else 100/(1+(v*scale/row['engineering_limits'][k])**2)
    return result


@lru_cache(None)
def collect():
    sources = {}
    def read(path):
        data = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    protocol = read(BASE/'protocol.json')
    assert protocol['checkpoints'] == [0,50] and protocol['seeds'] == list(SEEDS)
    assert protocol['training']['updates_per_seed'] == 50
    measurement = read(BASE/'measurement_protocol.json')
    manifest = read(RECOVERED/'pro/dataset_manifest.json')['videos']
    tests = {r['video_id']:r for r in manifest if r['split']=='test' and r['view']=='front'}
    assert len(tests)==300
    refs = {p.stem:read(p) for p in sorted((REF/'cases').glob('*.json'))}
    assert set(refs)==set(tests)
    groups = {}
    for seed in SEEDS:
        trained=read(RUNS/f'seed{seed}/training_complete.json')
        checkpoint=read(RUNS/f'seed{seed}/checkpoint_0050.json')
        assert trained['complete'] and trained['updates']==50 and trained['sequence_exposures']==400
        assert trained['seed']==checkpoint['seed']==seed and checkpoint['step']==50
    measurement_profiles=set()
    for family in FAMILIES:
        selected = [r for r in tests.values() if r['category']==family]
        assert len(selected)==150 and len({r['assembly_id'] for r in selected})==50
    for step in STEPS:
        for seed in SEEDS:
            folder = RUNS/f'seed{seed}'/f'step{step:04d}'
            report = read(folder/'scores.json')
            assert report['scoring_version']=='continuous-si-1' and not report['RMSE_subtraction']
            measurement_profiles.add(report['measurement_profile_sha256'])
            cases = [read(p) for p in sorted((folder/'cases').glob('*.json'))]
            assert len(cases)==300 and {r['video_id'] for r in cases}==set(tests)
            assert report['seed']==seed and report['step']==step
            for family in FAMILIES:
                rows = [dict(r,seed=seed,step=step) for r in cases if r['category']==family]
                f = report['families'][family]
                assert f['finite_videos']==150
                entry = dict(seed=seed, rows=rows, visual=f['visual'], branches={}, pairs=[])
                for branch in ('native','kinematic_intervention'):
                    b=f['branches'][branch]
                    comp={k:mean([components(r,branch)[k] for r in rows]) for k in METRICS}
                    np.testing.assert_allclose(list(comp.values()),list(b['continuous']['components_100'].values()),atol=1e-10)
                    score=mean(list(comp.values()))
                    np.testing.assert_allclose(score,b['continuous']['score_100'])
                    entry['branches'][branch]=dict(score=score,components=comp,
                        finite=sum(r.get(branch,{}).get('finite',False) for r in rows),
                        measurable=sum(r.get(branch,{}).get('mechanical_coverage',False) for r in rows),
                        joint=sum(r.get(branch,{}).get('physical_pass',False) for r in rows),
                        **{k:mean([r.get(branch,{}).get(k) for r in rows]) for k in FIELDS})
                assert entry['branches']['native']['measurable']==0
                assert all(not r.get('native',{}).get('component_pass',{}).get('geometry',False) for r in rows)
                assert all(b['joint']==0 for b in entry['branches'].values())
                for metric in ('rgb_mse','motion_mask_mse','persistence_mse','persistence_motion_mask_mse'):
                    np.testing.assert_allclose(mean([r['visual'][metric] for r in rows]),f['visual'][metric])
                entry['correction_mm']=mean([r.get('coordinate_correction_rmse_mm') for r in rows])
                for r in rows:
                    b=r.get('kinematic_intervention',{})
                    if not b.get('mechanical_coverage'): continue
                    ref=refs[r['video_id']]['kinematic_intervention']
                    assert ref['mechanical_coverage']
                    entry['pairs'].append(dict(video_id=r['video_id'],assembly_id=r['assembly_id'],seed=seed,
                        category=family,condition=r['condition'],
                        effort_tolerance=b['effort_tolerance'],energy_tolerance_J=b['energy_tolerance_J'],
                        reference_effort=ref['effort_rmse'],model_effort=b['effort_rmse'],
                        reference_energy_J=ref['work_energy_max_J'],model_energy_J=b['work_energy_max_J']))
                groups.setdefault(f'{family}_{step}',[]).append(entry)
    assert len(measurement_profiles)==1
    return dict(protocol=protocol,measurement=measurement,groups=groups,reference=refs,tests=tests,sources=sources)


def aggregate(entries):
    result=dict(registered_cases=450, seeds=list(SEEDS),
        visual={k:stats([e['visual'][k] for e in entries]) for k in entries[0]['visual']},
        correction_mm=stats([e['correction_mm'] for e in entries]),branches={})
    for branch in ('native','kinematic_intervention'):
        bs=[e['branches'][branch] for e in entries]
        result['branches'][branch]=dict(score=stats([b['score'] for b in bs]),
            components={k:mean([b['components'][k] for b in bs]) for k in METRICS},
            **{k:sum(b[k] for b in bs) for k in ('finite','measurable','joint')},
            **{k:mean([b[k] for b in bs]) for k in FIELDS})
    pairs=[p for e in entries for p in e['pairs']]
    result['paired']=dict(count=len(pairs),pairs=pairs,
        means={k:mean([mean([p[k] for p in e['pairs']]) for e in entries]) for k in
               ('reference_effort','model_effort','reference_energy_J','model_energy_J')},
        generated_higher={k:sum(p['model_'+k]>p['reference_'+k] for p in pairs) for k in ('effort','energy_J')})
    return result


def overview_profile():
    d=collect();entries=[e for f in FAMILIES for e in d['groups'][f'{f}_50']]
    rows=[r for e in entries for r in e['rows']]
    records=[dict(case_id=p['video_id'],seed=str(p['seed']),family=p['category'],condition=p['condition'],
        dynamics_ratio=p['model_effort']/p['effort_tolerance'],energy_ratio=p['model_energy_J']/p['energy_tolerance_J'])
        for e in entries for p in e['pairs']]
    return dict(model='LingBot',arm='adapted',suite='spatial-position',native_track='native',
        reconstructed_track='kinematic_intervention',registered_cases=900,
        components_100={k:mean([e['branches']['native']['components'][k] for e in entries]) for k in METRICS},
        reconstructed_components_100={k:mean([e['branches']['kinematic_intervention']['components'][k] for e in entries]) for k in METRICS},
        measured_counts={k:sum(r.get('native',{}).get(field) is not None for r in rows) for k,field in zip(METRICS,FIELDS)},
        native_score_100=mean([e['branches']['native']['score'] for e in entries]),
        reconstructed_score_100=mean([e['branches']['kinematic_intervention']['score'] for e in entries]),
        reconstructed_mechanics=dict(measurable_cases=len(records),cases=records,distributions={k:dict(
            quantiles=np.quantile([r[k+'_ratio'] for r in records],[0,.25,.5,.75,1]).tolist(),
            above_tolerance=sum(r[k+'_ratio']>1 for r in records)) for k in ('dynamics','energy')}))


def export_tables():
    d=collect(); groups={k:aggregate(v) for k,v in d['groups'].items()}
    pm=lambda v: f"${v['mean']:.2f}\\!\\pm\\!{v['sd']:.2f}$"
    main=[]; seedrows=[]; refs=[]; control=[]; detail=[]
    for family,name in zip(FAMILIES,('Stewart','Delta')):
        for step in STEPS:
            g=groups[f'{family}_{step}'];a,b=(g['branches'][k] for k in ('native','kinematic_intervention'))
            main.append(texrow(name,step,f"{g['visual']['rgb_mse']['mean']*1e4:.2f}",
                f"{g['visual']['motion_mask_mse']['mean']*1e2:.2f}",pm(a['score']),pm(b['score']),f"{b['measurable']}/450"))
            refs.append(texrow(name,step,g['paired']['count'],*[f'{g["paired"]["means"][k]:.3f}' for k in
                         ('reference_effort','model_effort','reference_energy_J','model_energy_J')]))
            detail.append(texrow(name,step,f"{a['finite']}/450",f"{b['finite']}/450",f"{g['correction_mm']['mean']:.2f}",
                    f"{a['point_rmse_mm']:.2f}",f"{b['point_rmse_mm']:.2f}",*[f'{b["components"][k]:.2f}' for k in METRICS]))
            for e in d['groups'][f'{family}_{step}']:
                a,b=(e['branches'][k] for k in ('native','kinematic_intervention'))
                seedrows.append(texrow(name,step,e['seed'],f"{e['visual']['rgb_mse']*1e4:.2f}",
                    f"{e['visual']['motion_mask_mse']*1e2:.2f}",f"{a['score']:.2f}",f"{b['score']:.2f}",
                    f"{b['measurable']}/150",f"{b['effort_rmse']:.3f}",f"{b['work_energy_max_J']:.3f}"))
        rows=[r for r in d['reference'].values() if r['category']==family]
        for key,label in (('exact_state_control','Exact states'),('sampled_state_control','30 Hz differences'),('kinematic_intervention','Reference RGB + CAD')):
            bs=[r[key] for r in rows]
            control.append(texrow(name,label,*[f'{mean([b[k] for b in bs]):.4f}' for k in ('point_rmse_mm','effort_rmse','work_energy_max_J')],
                f"{sum(b['component_pass']['effort'] for b in bs)}/150",f"{sum(b['component_pass']['energy'] for b in bs)}/150"))
    for name,lines in [('spatial_main_rows',main),('spatial_expansion_seed_rows',seedrows),('spatial_reference_rows',refs),
                       ('spatial_expansion_control_rows',control),('spatial_expansion_detail_rows',detail)]:
        (PAPER/'tables'/f'{name}.tex').write_text('\n'.join(lines)+'\n')
    from prepare_vjepa_expansion import export as export_ac
    ac = export_ac()
    evidence=dict(schema='lingbot-0-50-three-seeds',groups=groups,protocol=d['protocol'],sources=d['sources'],
        aggregation='Equal seed means; sample SD across three seeds. Each family has 50 assemblies and three conditions. Available-case physical means are computed within seed, then averaged. Counts pool seeds; missing cases remain in the composite denominator.')
    (PAPER/'sources/spatial_expansion.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return evidence


if __name__=='__main__':
    e=export_tables()
    print({k:(v['branches']['native']['score'],v['branches']['kinematic_intervention']['score'],v['paired']['count']) for k,v in e['groups'].items()})
