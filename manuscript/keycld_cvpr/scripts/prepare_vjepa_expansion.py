"""Current three-seed V-JEPA evidence shared by tables and figures."""
import json
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.vjepa_expansion.summarize import collect as measurements, aggregate, summarize_rows, METRICS, FIELDS, FAMILIES, SEEDS
PAPER=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/vjepa_expansion'
NAMES={'H-A':'Stewart','H-B':'Delta'}


@lru_cache(None)
def collect():
    return measurements()


def profile():
    d=collect();entries=[e for f in FAMILIES for e in d['groups'][f+'_calibrated']]
    rows=[dict(r,family=f) for f in FAMILIES for e in d['groups'][f+'_calibrated'] for r in e['rows']]
    g=aggregate(entries)
    case=[]
    for r in rows:
        b=r['kinematic_intervention']
        if not b.get('mechanical_coverage'):continue
        case.append(dict(case_id=r['video_id'],seed=str(r['seed']),family=r['family'],condition=r['condition'],
            dynamics_ratio=b['effort_rmse']/b['effort_tolerance'],energy_ratio=b['work_energy_max_J']/b['energy_tolerance_J']))
    return dict(model='V-JEPA',arm='AC',suite='spatial-position',native_track='native',
        reconstructed_track='kinematic_intervention',registered_cases=900,
        components_100=g['branches']['native']['components'],
        reconstructed_components_100=g['branches']['kinematic_intervention']['components'],
        native_score_100=g['branches']['native']['score']['mean'],
        reconstructed_score_100=g['branches']['kinematic_intervention']['score']['mean'],
        measured_counts={k:sum(r['native'].get(field) is not None for r in rows) for k,field in zip(METRICS,FIELDS)},
        reconstructed_mechanics=dict(measurable_cases=len(case),cases=case,distributions={k:dict(
            quantiles=np.quantile([r[k+'_ratio'] for r in case],[0,.25,.5,.75,1]).tolist(),
            above_tolerance=sum(r[k+'_ratio']>1 for r in case)) for k in ('dynamics','energy')}))


def texrow(*cells):return ' & '.join(map(str,cells))+r' \\'


def export():
    d=collect();groups={k:aggregate(v) for k,v in d['groups'].items()}
    pm=lambda v:f"${v['mean']:.2f}\\!\\pm\\!{v['sd']:.2f}$"
    number=lambda v,n=2:'N/A' if v is None else f'{v:.{n}f}'
    main=[];scores=[];seedrows=[];readouts=[];refs=[]
    for f,name in NAMES.items():
        g=groups[f+'_calibrated'];a,b=[g['branches'][k] for k in ('native','kinematic_intervention')]
        updates={r['updates'] for r in d['training'] if r['family']==f};assert len(updates)==1
        main.append(texrow('AC '+name,updates.pop(),'--','--',pm(a['score']),pm(b['score']),f"{b['measurable']}/450"))
        scores.append(texrow(name,pm(a['score']),pm(b['score']),f"{b['measurable']}/450",f"{b['joint']}/450"))
        for e in d['groups'][f+'_calibrated']:
            s=summarize_rows(e['rows']);raw,cad=[s['branches'][k] for k in ('native','kinematic_intervention')]
            seedrows.append(texrow(name,e['seed'],number(raw['score']),number(cad['score']),f"{cad['measurable']}/150",
                number(s['correction_mm']),number(raw['point_rmse_mm']),number(cad['point_rmse_mm']),number(cad['effort_rmse'],3),number(cad['work_energy_max_J'],3)))
        entries=[('Internal','Reference',groups[f+'_reference_internal']),
                 ('Calibrated','Reference',aggregate([dict(rows=d['reference'][f])])),
                 ('Internal','Forecast',groups[f+'_native']),('Calibrated','Forecast',g)]
        for reader,mode,s in entries:
            a,b=[s['branches'][k] for k in ('native','kinematic_intervention')]
            readouts.append(texrow(name,reader,mode,number(a['point_rmse_mm']),number(b['point_rmse_mm']),
                number(a['score']['mean']),number(b['score']['mean']),f"{b['measurable']}/{s['registered_cases']}",
                number(b['effort_rmse'],3),number(b['work_energy_max_J'],3)))
        pairs=d['pairs'][f]
        means=[np.mean([np.mean([p[k] for p in pairs if p['seed']==s]) for s in SEEDS])
               for k in ('reference_effort','model_effort','reference_energy_J','model_energy_J')]
        refs.append(texrow(name,len(pairs),*[number(v,3) for v in means]))
    for name,rows in [('spatial_ac_main_rows',main),('spatial_ac_rows',scores),('spatial_ac_seed_rows',seedrows),
                     ('vjepa_readout_rows',readouts),('spatial_ac_reference_rows',refs)]:
        (PAPER/'tables'/(name+'.tex')).write_text('\n'.join(rows)+'\n')
    result=dict(schema='vjepa-spatial-three-seeds-3000-exposures',groups=groups,protocol=d['protocol'],
        training=d['training'],validation=d['validation'],sources=d['sources'],
        reference={f:summarize_rows(v) for f,v in d['reference'].items()},
        aggregation='Equal means across seeds 42, 43, 44; each seed has 50 identities and three conditions per family. Missing terms remain in all 450-case scores. Native denotes unconstrained calibrated readout; CAD is the separate post-hoc geometric intervention.')
    (PAPER/'sources/vjepa_expansion.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':export()
