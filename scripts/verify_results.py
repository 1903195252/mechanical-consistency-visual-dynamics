#!/usr/bin/env python3
"""Recompute published scores and check the registered test denominators."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.vjepa_expansion.summarize import aggregate

METRICS=('geometry','constraint_velocity','constraint_acceleration','dynamics','energy','prediction')
FIELDS=('geometry_max_mm','velocity_constraint_max_m_s','acceleration_constraint_max_m_s2','effort_rmse','work_energy_max_J','point_rmse_mm')
FACTORS=(.001,1,1,1,1,.001)


def read(path):return json.loads((ROOT/path).read_text())
def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10),(a,b)
def mean(values):return sum(values)/len(values)

def compare(actual,expected):
    if isinstance(actual,dict):
        for key,value in actual.items():compare(value,expected[key])
    elif isinstance(actual,(int,float)):close(actual,expected)
    else:assert actual==expected,(actual,expected)


def cohort(path):
    rows=read(path)['videos'];tests=[r for r in rows if r['split']=='test']
    assert len(tests)==300 and len({r['video_id'] for r in tests})==300
    for f in ('H-A','H-B'):
        subset=[r for r in tests if r['category']==f]
        ids={r['assembly_id'] for r in subset}
        assert len(subset)==150 and len(ids)==50
        assert not ids & {r['assembly_id'] for r in rows if r['split']!='test'}
        for i in ids:
            assert {r['condition'] for r in subset if r['assembly_id']==i}=={'low_speed','rated','limit_or_load'}
    return {r['video_id']:r for r in tests}


def ling_score(row,branch):
    values=row.get(branch,{})
    return mean([0. if values.get(field) is None else 100/(1+(values[field]*factor/row['engineering_limits'][key])**2)
                 for key,field,factor in zip(METRICS,FIELDS,FACTORS)])


def main():
    planar=read('data/manifests/planar.json')['records']
    parts={split:{r['assembly_id'] for r in planar if r['split']==split} for split in ('train','validation','calibration','test')}
    assert {k:len(v) for k,v in parts.items()}==dict(train=271,validation=18,calibration=18,test=36)
    assert len(set.union(*parts.values()))==343
    resolution=read('manuscript/keycld_cvpr/sources/resolution_comparison.json')['runs']
    assert {(r['model'],r['width']) for r in resolution}=={(m,w) for m in ('KeyCLD','CON') for w in (64,256)}
    for run in resolution:
        for branch in ('native','reconstructed'):
            block=run[branch];rows=block['rows'];assert len(rows)==324
            for row in rows:close(row['score'],mean([row[k+'_score'] for k in METRICS]))
            close(block['score'],mean([r['score'] for r in rows]))
    with (ROOT/'results/continuous/case_scores.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==read('results/continuous/scores.json')['metric_rows']==12956
    for row in rows:close(float(row['continuous_score_100']),mean([float(row[k+'_score_100']) for k in METRICS]))
    expected=cohort('data/manifests/spatial_lingbot.json')
    assert set(expected)==set(cohort('data/manifests/spatial_vjepa.json'))
    vjepa=read('results/spatial/vjepa/summary.json')
    for family in ('H-A','H-B'):
        ids={k for k,r in expected.items() if r['category']==family}
        for mode in ('native','calibrated','reference_internal'):
            entries=[]
            for seed in (42,43,44):
                rows=read('results/spatial/vjepa/seed_%d/%s/%s_score.json'%(seed,mode,family))
                assert len(rows)==150 and {r['video_id'] for r in rows}==ids
                history=read('results/spatial/vjepa/seed_%d/%s_training.json'%(seed,family))
                assert history[-1]['sequence_exposures']==3000
                entries.append(dict(seed=seed,rows=rows))
            compare(aggregate(entries),vjepa['groups'][family+'_'+mode])
    ling=read('results/spatial/lingbot/summary.json')
    refs={r['video_id']:r for r in read('results/spatial/lingbot/reference.json')}
    assert set(refs)==set(expected)
    for step in (0,50):
        for seed in (42,43,44):
            rows=read('results/spatial/lingbot/seed%d/step%04d.json'%(seed,step))
            assert len(rows)==300 and {r['video_id'] for r in rows}==set(expected)
            for family in ('H-A','H-B'):
                subset=[r for r in rows if r['category']==family]
                summary=next(r for r in ling['groups'][family+'_'+str(step)] if r['seed']==seed)
                for branch,key in [('native','scomp_native'),('kinematic_intervention','scomp_reconstructed')]:
                    close(mean([ling_score(r,branch) for r in subset]),summary[key])
                assert summary['measurable']==sum(bool(r.get('kinematic_intervention',{}).get('mechanical_coverage')) for r in subset)
                assert summary['native_measurable']==0
                pairs=[(refs[r['video_id']]['kinematic_intervention'],r['kinematic_intervention']) for r in subset
                       if r.get('kinematic_intervention',{}).get('mechanical_coverage') and refs[r['video_id']].get('kinematic_intervention',{}).get('mechanical_coverage')]
                for metric in ('effort_rmse','work_energy_max_J'):
                    close(mean([a[metric] for a,b in pairs]),summary[metric]['reference'])
                    close(mean([b[metric] for a,b in pairs]),summary[metric]['generated'])
                    assert sum(b[metric]>a[metric] for a,b in pairs)==summary[metric]['generated_higher']
    print('Verified 12,956 continuous-score rows, four planar resolution settings, 900 AC forecasts and 1,800 LingBot videos against current aggregates.')


if __name__=='__main__':main()
