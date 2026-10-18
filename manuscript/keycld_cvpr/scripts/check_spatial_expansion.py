"""Independently check the current spatial tables and plotted case coordinates."""
import hashlib
import json
import numpy as np
from prepare_spatial_expansion import collect, aggregate, PAPER, ROOT, FAMILIES, STEPS, SEEDS


def cells(name):
    return [[v.strip() for v in line.rstrip('\\ ').split(' & ')] for line in
            (PAPER/'tables'/name).read_text().splitlines() if ' & ' in line]


def verify():
    d=collect();groups={k:aggregate(v) for k,v in d['groups'].items()}
    evidence=json.loads((PAPER/'sources/spatial_expansion.json').read_text())
    assert evidence['groups']==groups and evidence['sources']==d['sources']
    protocol=d['protocol'];assert protocol['training']['sequence_exposures_per_seed']==400
    assert protocol['training']['full_parameter_adaptation'] and protocol['inference']['future_frames']==117
    figure=json.loads((PAPER/'sources/spatial_expansion_figure.json').read_text())
    assert figure['schema']=='lingbot-0-50-three-seeds'
    assert not figure['text_layout']['overlaps']
    for path,digest in figure['sources'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    main=[];refs=[];detail=[];seedrows=[];controls=[]
    for f,name in zip(FAMILIES,('Stewart','Delta')):
        for step in STEPS:
            g=groups[f'{f}_{step}'];a,b=[g['branches'][key] for key in ('native','kinematic_intervention')]
            score=lambda b:f"${b['score']['mean']:.2f}\\!\\pm\\!{b['score']['sd']:.2f}$"
            main.append([name,str(step),f"{g['visual']['rgb_mse']['mean']*1e4:.2f}",f"{g['visual']['motion_mask_mse']['mean']*1e2:.2f}",score(a),score(b),f"{b['measurable']}/450"])
            refs.append([name,str(step),str(g['paired']['count'])]+[f'{g["paired"]["means"][k]:.3f}' for k in ('reference_effort','model_effort','reference_energy_J','model_energy_J')])
            detail.append([name,str(step),f"{a['finite']}/450",f"{b['finite']}/450",f"{g['correction_mm']['mean']:.2f}",f"{a['point_rmse_mm']:.2f}",f"{b['point_rmse_mm']:.2f}"]+[f'{v:.2f}' for v in b['components'].values()])
            for e in d['groups'][f'{f}_{step}']:
                a,b=[e['branches'][key] for key in ('native','kinematic_intervention')]
                seedrows.append([name,str(step),str(e['seed']),f"{e['visual']['rgb_mse']*1e4:.2f}",f"{e['visual']['motion_mask_mse']*1e2:.2f}",f"{a['score']:.2f}",f"{b['score']:.2f}",f"{b['measurable']}/150",f"{b['effort_rmse']:.3f}",f"{b['work_energy_max_J']:.3f}"])
                assert a['measurable']==a['joint']==b['joint']==0
            rows=[r for e in d['groups'][f'{f}_{step}'] for r in e['rows'] if r.get('native',{}).get('finite')]
            point=next(r for r in figure['point_comparison'] if r['family']==f and r['step']==step)
            values=[r['native']['point_rmse_mm']-r['readout_persistence']['point_rmse_mm'] for r in rows]
            np.testing.assert_allclose(point['values_mm'],values);assert min(values)>-10 and max(values)<27
        for key,label in (('exact_state_control','Exact states'),('sampled_state_control','30 Hz differences'),('kinematic_intervention','Reference RGB + CAD')):
            bs=[r[key] for r in d['reference'].values() if r['category']==f]
            assert all(b['mechanical_coverage'] for b in bs)
            controls.append([name,label]+[f'{np.mean([b[k] for b in bs]):.4f}' for k in ('point_rmse_mm','effort_rmse','work_energy_max_J')]+[f"{sum(b['component_pass'][k] for b in bs)}/150" for k in ('effort','energy')])
        for metric in ('rgb_mse','motion_mask_mse'):
            before=np.array([r['visual'][metric] for e in d['groups'][f'{f}_0'] for r in e['rows']]);after=np.array([r['visual'][metric] for e in d['groups'][f'{f}_50'] for r in e['rows']])
            plotted=next(r for r in figure['adaptation'] if r['family']==f and r['metric']==metric)
            np.testing.assert_allclose(plotted['case_ratios'],after/before)
            np.testing.assert_allclose(plotted['aggregate_percent_change'],100*(after.mean()/before.mean()-1))
            assert min(after/before)>.012 and max(after/before)<8
        for metric,tol in (('effort','effort_tolerance'),('energy_J','energy_tolerance_J')):
            plotted=next(r for r in figure['matched_reference_mechanics'] if r['family']==f and r['metric']==metric)
            pairs=groups[f'{f}_50']['paired']['pairs'];assert plotted['paired_cases']==len(pairs)==(450 if f=='H-A' else 424)
            for key,out in (('reference','reference_ratios'),('model','generated_ratios')):
                values=[p[key+'_'+metric]/p[tol] for p in pairs]
                np.testing.assert_allclose(plotted[out],values)
                assert plotted['limits'][0]<min(values) and max(values)<plotted['limits'][1]
            np.testing.assert_allclose(plotted['si_means'],[groups[f'{f}_50']['paired']['means'][key+'_'+metric] for key in ('reference','model')])
            diagnostic=next(r for r in figure['mechanics_diagnostics'] if r['family']==f and r['metric']==metric)
            ratios=np.array([p['model_'+metric]/p['reference_'+metric] for p in pairs])
            assert diagnostic['paired_cases']==len(pairs) and diagnostic['position']=='lower right'
            np.testing.assert_allclose(diagnostic['case_ratios'],ratios)
            np.testing.assert_allclose(diagnostic['mean'],ratios.mean())
            np.testing.assert_allclose(diagnostic['quartiles'],np.quantile(ratios,[.25,.5,.75]))
            np.testing.assert_allclose(diagnostic['range'],[ratios.min(),ratios.max()])
        assert groups[f'{f}_50']['paired']['generated_higher']['effort']==len(pairs)
    for file,expected in [('spatial_main_rows.tex',main),('spatial_reference_rows.tex',refs),('spatial_expansion_seed_rows.tex',seedrows),
                         ('spatial_expansion_detail_rows.tex',detail),('spatial_expansion_control_rows.tex',controls)]:
        assert cells(file)==expected,file
    for ex,f in zip(figure['examples'],FAMILIES):
        assert ex['video_id']==f+'-046-P10-V01' and ex['seed']==42 and ex['frame']==119
        np.testing.assert_allclose(ex['reconstructed']['tips'],ex['reconstructed']['sockets'],atol=1e-8)
        for path,digest in ex['sources'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    sources=(PAPER/'main.tex').read_text()+(PAPER/'supplement.tex').read_text()
    for stale in ('34 updates','272 sequence','25 updates','55.71/55.50','all 15 Stewart and 12 Delta','15/10 Stewart/Delta paths','piecewise cosine schedule','five identities per spatial family','all 60 generated RGB'):
        assert stale not in sources,stale
    assert len({r['assembly_id'] for r in d['tests'].values()})==100
    assert len(d['reference'])==300
    assert len(figure['zooms'])==10
    assert len(figure['mechanics_diagnostics'])==4
    for z in figure['zooms']:
        if z['panel']=='b':
            record=next(r for r in figure['adaptation'] if r['family']==z['family'] and r['metric']==z['metric'])
            values=record['case_ratios']
        elif z['panel']=='c':
            record=next(r for r in figure['point_comparison'] if r['family']==z['family'] and r['step']==z['step'])
            values=record['values_mm']
            np.testing.assert_allclose(record['mean_mm'],np.mean(values))
            assert record['full_range_cases']==len(values) and record['zoom_cases']==z['visible_cases']
        else:
            raise AssertionError(('Unexpected magnification panel',z['panel']))
        assert z['visible_cases']==sum(z['bounds'][0]<=v<=z['bounds'][1] for v in values)
        assert z['total_cases']==len(values)
    assert len(figure['density_contours'])==4
    assert all(r['cases'] in (450,424) and r['coordinate_space']=='log10 error/tolerance on both axes' for r in figure['density_contours'])
    from prepare_vjepa_expansion import collect as collect_ac
    ac=collect_ac()
    for f in FAMILIES:
        point=next(r for r in figure['point_comparison'] if r['family']==f and r['step']=='AC')
        cases=[r for e in ac['groups'][f+'_calibrated'] for r in e['rows']]
        assert len(cases)==450
        np.testing.assert_allclose(point['values_mm'],[r['native']['point_rmse_mm']-r['readout_persistence']['point_rmse_mm'] for r in cases])
    return dict(test_assemblies=100,test_trajectories=300,seeds=list(SEEDS),checkpoints=list(STEPS),
        generated_videos=1800,reference_videos=300,updates_per_seed=50,exposures_per_seed=400,
        main_table_cells=42,supplement_cells=242,adaptation_points=1800,matched_mechanics_points=1748,
        all_scores_recomputed_from_case_errors=True,failures_retained=True,source_hashes_verified=True,
        paired_reference_measurements_verified=True,paired_ratio_distributions_verified=True,
        geometry_overlays_verified=True,zoom_membership_verified=True,
        predictive_cohort_matched=True,old_lingbot_results_removed=True)


if __name__=='__main__':print(json.dumps(verify(),indent=2))
