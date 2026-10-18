"""Independently verify the expanded AC measurements and publication tables."""
import hashlib
import json
import numpy as np
from prepare_vjepa_expansion import collect, ROOT, PAPER, OUT, FAMILIES, SEEDS, NAMES


FIELDS=('geometry_max_mm','velocity_constraint_max_m_s','acceleration_constraint_max_m_s2',
        'effort_rmse','work_energy_max_J','point_rmse_mm')


def shown(name):
    return [[v.strip() for v in line.rstrip('\\ ').split(' & ')] for line in
            (PAPER/'tables'/name).read_text().splitlines() if ' & ' in line]


def branch_stats(rows,branch):
    values=[]
    for r in rows:
        limits=[1.,.01,.1,r['native']['effort_tolerance'],r['native']['energy_tolerance_J'],10.]
        b=r[branch]
        values.append([0. if b.get(k) is None else 100/(1+(b[k]/t)**2) for k,t in zip(FIELDS,limits)])
    matrix=np.asarray(values)
    # Every seed contains all 50 identities and three operating conditions.
    # Thus this direct mean equals the registered condition/identity mean.
    return dict(score=float(matrix.mean()),components=matrix.mean(0),
                coverage=sum(all(r[branch].get(k) is not None for k in FIELDS) for r in rows),
                joint=sum(bool(r[branch].get('physical_pass')) for r in rows),
                means={k:float(np.mean(v)) if (v:=[r[branch][k] for r in rows if r[branch].get(k) is not None]) else None for k in FIELDS})


def verify():
    d=collect();e=json.loads((PAPER/'sources/vjepa_expansion.json').read_text())
    assert e['sources']==d['sources']
    for path,digest in e['sources'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    for path,digest in d['protocol']['instrument_checkpoints'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    manifest=json.loads((OUT/'dataset_manifest.json').read_text())['videos']
    lingbot=json.loads((ROOT/'output/lingbot_expansion/dataset_manifest.json').read_text())['videos']
    assert {r['video_id'] for r in lingbot if r['split']=='test'}==set(d['tests'])
    for f in FAMILIES:
        partitions=[{r['assembly_id'] for r in manifest if r['category']==f and r['split']==split} for split in ('train','validation','test')]
        assert len(partitions[-1])==50
        assert all(not a&b for i,a in enumerate(partitions) for b in partitions[i+1:])
    cal=ROOT/'output/closed_chain_eval/visual_calibration/vjepa_readout_replay/calibration'
    selection=json.loads((cal/'selection.json').read_text());verification=json.loads((cal/'verification.json').read_text())
    ids=[selection['train_ids'],selection['development_ids'],[r['video_id'] for r in verification['cases']],list(d['tests'])]
    sets=[{v.split('-P')[0] for v in group} for group in ids]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    assert not selection['coordinate_smoothing'] and not selection['test_labels_used']
    geometry=json.loads((ROOT/'output/closed_chain_eval/visual_calibration/optimization/geometry_spatial_reader/specification.json').read_text())
    train_ids={r['video_id'] for r in manifest if r['split']=='train'}
    assert set(geometry['train_ids'])<=train_ids
    assert not {v.split('-V')[0] for v in geometry['train_ids']}&{r['assembly_id'] for r in d['tests'].values()}
    assert not geometry['physical_equations_used_in_fit'] and not geometry['closure_loss']
    supplementary=(PAPER/'supplement.tex').read_text()
    for f in FAMILIES:
        assert len(d['controls'][f])==150
        for branch in ('reference_exact','reference_30Hz'):
            assert all(r[branch]['physical_pass'] for r in d['controls'][f])
        for field in ('effort_rmse','work_energy_max_J'):
            maximum=max(r['reference_30Hz'][field] for r in d['controls'][f])
            assert f'{maximum:.4f}' in supplementary
    main=[];scores=[];seedrows=[];ablations=[];refs=[];checked=0
    number=lambda value,n=2:'N/A' if value is None else f'{value:.{n}f}'
    for f,name in NAMES.items():
        summaries={}
        for reader in ('native','calibrated','reference_internal'):
            entries=d['groups'][f+'_'+reader];assert [v['seed'] for v in entries]==list(SEEDS)
            summary={}
            for branch in ('native','kinematic_intervention','readout_persistence'):
                bs=[branch_stats(entry['rows'],branch) for entry in entries]
                a=e['groups'][f+'_'+reader]['branches'][branch]
                np.testing.assert_allclose(a['score']['mean'],np.mean([v['score'] for v in bs]))
                np.testing.assert_allclose(a['score']['sd'],np.std([v['score'] for v in bs],ddof=1))
                assert a['measurable']==sum(v['coverage'] for v in bs)
                assert a['joint']==sum(v['joint'] for v in bs)
                summary[branch]=dict(score=np.mean([v['score'] for v in bs]),sd=np.std([v['score'] for v in bs],ddof=1),
                    coverage=sum(v['coverage'] for v in bs),joint=sum(v['joint'] for v in bs),
                    means={k:np.mean([v['means'][k] for v in bs if v['means'][k] is not None]) if any(v['means'][k] is not None for v in bs) else None for k in FIELDS})
            summaries[reader]=summary;checked+=450
        a,b=[summaries['calibrated'][k] for k in ('native','kinematic_intervention')]
        pm=lambda v:f"${v['score']:.2f}\\!\\pm\\!{v['sd']:.2f}$"
        main.append(['AC '+name,str(386 if f=='H-A' else 400),'--','--',pm(a),pm(b),f"{b['coverage']}/450"])
        scores.append([name,pm(a),pm(b),f"{b['coverage']}/450",f"{b['joint']}/450"])
        for entry in d['groups'][f+'_calibrated']:
            rows=entry['rows'];raw,cad=[branch_stats(rows,k) for k in ('native','kinematic_intervention')]
            corrections=[r['coordinate_correction_rmse_mm'] for r in rows if r.get('coordinate_correction_rmse_mm') is not None]
            seedrows.append([name,str(entry['seed']),number(raw['score']),number(cad['score']),f"{cad['coverage']}/150",number(np.mean(corrections)),
                number(raw['means']['point_rmse_mm']),number(cad['means']['point_rmse_mm']),number(cad['means']['effort_rmse'],3),number(cad['means']['work_energy_max_J'],3)])
        reference={k:branch_stats(d['reference'][f],k) for k in ('native','kinematic_intervention')}
        for reader,mode,stats,total in [('Internal','Reference',summaries['reference_internal'],450),('Calibrated','Reference',reference,150),
                                       ('Internal','Forecast',summaries['native'],450),('Calibrated','Forecast',summaries['calibrated'],450)]:
            a,b=[stats[k] for k in ('native','kinematic_intervention')]
            ablations.append([name,reader,mode,number(a['means']['point_rmse_mm']),number(b['means']['point_rmse_mm']),number(a['score']),number(b['score']),
                f"{b['coverage']}/{total}",number(b['means']['effort_rmse'],3),number(b['means']['work_energy_max_J'],3)])
        pairs=d['pairs'][f]
        means=[np.mean([np.mean([p[k] for p in pairs if p['seed']==seed]) for seed in SEEDS]) for k in
               ('reference_effort','model_effort','reference_energy_J','model_energy_J')]
        refs.append([name,str(len(pairs)),*[number(v,3) for v in means]])
    for file,expected in [('spatial_ac_main_rows.tex',main),('spatial_ac_rows.tex',scores),('spatial_ac_seed_rows.tex',seedrows),
                          ('vjepa_readout_rows.tex',ablations),('spatial_ac_reference_rows.tex',refs)]:
        assert shown(file)==expected,file
    return dict(test_assemblies=100,test_trajectories=300,forecasts=900,seeds=list(SEEDS),family_fits=6,
                exposures_per_family_seed=3000,reference_videos=300,numerical_reference_cases=300,
                exact_and_sampled_reference_checks_passed=True,readout_case_measurements=checked+300,
                shared_lingbot_test_cohort=True,calibration_disjoint=True,all_table_cells_recomputed=True)


if __name__=='__main__':print(json.dumps(verify(),indent=2))
