"""Verify and summarize the complete registered three-seed spatial experiment."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/vjepa_expansion'
FAMILIES=('H-A','H-B')
SEEDS=(42,43,44)
METRICS=('geometry','constraint_velocity','constraint_acceleration','dynamics','energy','prediction')
FIELDS=('geometry_max_mm','velocity_constraint_max_m_s','acceleration_constraint_max_m_s2',
        'effort_rmse','work_energy_max_J','point_rmse_mm')
FACTORS=(.001,1,1,1,1,.001)


def mean(values):
    values=[v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def stats(values):
    return dict(mean=mean(values),sd=float(np.std(values,ddof=1)) if len(values)>1 else 0.)


def components(row,branch):
    original=row['native'];b=row.get(branch,{})
    limits=(.001,.01,.1,original['effort_tolerance'],original['energy_tolerance_J'],.01)
    return {k:0. if b.get(field) is None else float(100/(1+(b[field]*factor/limit)**2))
            for k,field,factor,limit in zip(METRICS,FIELDS,FACTORS,limits)}


def summarize_rows(rows):
    result=dict(registered=len(rows),branches={},correction_mm=mean([r.get('coordinate_correction_rmse_mm') for r in rows]))
    for branch in ('native','kinematic_intervention','readout_persistence'):
        bs=[r.get(branch,{}) for r in rows]
        comp={k:mean([components(r,branch)[k] for r in rows]) for k in METRICS}
        result['branches'][branch]=dict(score=mean(list(comp.values())),components=comp,
            finite=sum(bool(b.get('finite')) for b in bs),measurable=sum(bool(b.get('mechanical_coverage')) for b in bs),
            joint=sum(bool(b.get('physical_pass')) for b in bs),
            **{field:mean([b.get(field) for b in bs]) for field in FIELDS})
    return result


def aggregate(entries):
    groups=[summarize_rows(e['rows']) for e in entries]
    result=dict(registered_cases=sum(g['registered'] for g in groups),
        correction_mm=stats([g['correction_mm'] for g in groups]),branches={})
    for branch in ('native','kinematic_intervention','readout_persistence'):
        bs=[g['branches'][branch] for g in groups]
        result['branches'][branch]=dict(score=stats([b['score'] for b in bs]),
            components={k:mean([b['components'][k] for b in bs]) for k in METRICS},
            **{k:sum(b[k] for b in bs) for k in ('finite','measurable','joint')},
            **{k:mean([b[k] for b in bs]) for k in FIELDS})
    return result


def collect():
    sources={}
    def read(path):
        content=path.read_bytes();sources[str(path.relative_to(ROOT))]=hashlib.sha256(content).hexdigest()
        return json.loads(content)
    protocol=read(OUT/'protocol.json');manifest=read(OUT/'dataset_manifest.json')['videos']
    for relative in (
        'output/closed_chain_eval/visual_calibration/optimization/geometry_spatial_reader/specification.json',
        'output/closed_chain_eval/visual_calibration/vjepa_readout_replay/calibration/selection.json',
        'output/closed_chain_eval/visual_calibration/vjepa_readout_replay/calibration/verification.json'):
        read(ROOT/relative)
    assert protocol['seeds']==list(SEEDS) and protocol['sequence_exposures_per_family_seed']==3000
    tests={r['video_id']:r for r in manifest if r['split']=='test'}
    assert len(tests)==300
    for f in FAMILIES:
        selected=[r for r in tests.values() if r['category']==f]
        assert len(selected)==150 and len({r['assembly_id'] for r in selected})==50
        for assembly in {r['assembly_id'] for r in selected}:
            cases=[r for r in selected if r['assembly_id']==assembly]
            assert len(cases)==3 and {r['condition'] for r in cases}=={'low_speed','rated','limit_or_load'}
            assert all(r['view']=='front' for r in cases)
    groups={};reference={};controls={};training=[];validation=[];pairs={};point={}
    for f in FAMILIES:
        expected={k for k,r in tests.items() if r['category']==f}
        reference[f]=read(OUT/'reference/calibrated'/(f+'_score.json'))
        controls[f]=read(OUT/'reference/controls'/(f+'_calibrate.json'))
        assert {r['video_id'] for r in reference[f]}==expected
        assert {r['video_id'] for r in controls[f]}==expected
        refs={r['video_id']:r for r in reference[f]}
        pairs[f]=[];point[f]=[]
        for seed in SEEDS:
            folder=OUT/('seed_%d'%seed)
            done=read(folder/(f+'_measurement_complete.json'));assert done['cases']==150
            history=read(folder/(f+'_training.json'));last=history[-1]
            assert last['sequence_exposures']==3000
            assert last['epoch']==(15 if f=='H-A' else 50)
            assert last['updates']==(386 if f=='H-A' else 400)
            training.append(dict(family=f,seed=seed,**last))
            v=read(folder/(f+'_readout_validation.json'));assert len(v)==12
            validation.append(dict(family=f,seed=seed,point_mm=mean([r['platform_position_rmse_mm'] for r in v])))
            for reader in ('native','calibrated','reference_internal'):
                rows=read(folder/reader/(f+'_score.json'))
                assert len(rows)==150 and {r['video_id'] for r in rows}==expected
                assert all(r['status']=='evaluated' for r in rows)
                rows=[dict(r,seed=seed) for r in rows]
                groups.setdefault(f+'_'+reader,[]).append(dict(seed=seed,rows=rows))
                if reader!='calibrated':continue
                for r in rows:
                    a,b=refs[r['video_id']]['kinematic_intervention'],r['kinematic_intervention']
                    if a.get('mechanical_coverage') and b.get('mechanical_coverage'):
                        pairs[f].append(dict(video_id=r['video_id'],assembly_id=r['assembly_id'],seed=seed,
                            reference_effort=a['effort_rmse'],model_effort=b['effort_rmse'],
                            reference_energy_J=a['work_energy_max_J'],model_energy_J=b['work_energy_max_J'],
                            effort_tolerance=b['effort_tolerance'],energy_tolerance_J=b['energy_tolerance_J']))
                    if r['native'].get('finite') and r['readout_persistence'].get('finite'):
                        point[f].append(dict(seed=seed,assembly_id=r['assembly_id'],condition=r['condition'],
                            delta_mm=r['native']['point_rmse_mm']-r['readout_persistence']['point_rmse_mm']))
    return dict(protocol=protocol,tests=tests,groups=groups,reference=reference,controls=controls,
        training=training,validation=validation,pairs=pairs,point=point,sources=sources)


def export():
    data=collect();result=dict(schema='vjepa-spatial-three-seeds-3000-exposures',protocol=data['protocol'],
        groups={k:aggregate(v) for k,v in data['groups'].items()},
        reference={k:summarize_rows(v) for k,v in data['reference'].items()},
        training=data['training'],validation=data['validation'],paired={},point={},controls={},sources=data['sources'])
    for f in FAMILIES:
        result['controls'][f]={branch:dict(
            cases=len(data['controls'][f]),
            joint=sum(bool(r[branch]['physical_pass']) for r in data['controls'][f]),
            **{field:mean([r[branch].get(field) for r in data['controls'][f]]) for field in FIELDS})
            for branch in ('reference_exact','reference_30Hz')}
        pairs=data['pairs'][f]
        result['paired'][f]=dict(count=len(pairs),means={k:mean([mean([p[k] for p in pairs if p['seed']==s]) for s in SEEDS])
            for k in ('reference_effort','model_effort','reference_energy_J','model_energy_J')},
            higher={k:sum(p['model_'+k]>p['reference_'+k] for p in pairs) for k in ('effort','energy_J')})
        rows=data['point'][f];ids=sorted({r['assembly_id'] for r in rows});assert len(rows)==450
        array=np.array([[mean([r['delta_mm'] for r in rows if r['seed']==s and r['assembly_id']==i]) for i in ids] for s in SEEDS])
        rng=np.random.default_rng(20261017)
        draws=[array[np.ix_(rng.integers(3,size=3),rng.integers(50,size=50))].mean() for _ in range(10000)]
        result['point'][f]=dict(mean_mm=float(array.mean()),ci95_mm=np.quantile(draws,[.025,.975]).tolist())
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(groups=result['groups'],paired=result['paired'],point=result['point']),indent=2))
    return result


if __name__=='__main__':export()
