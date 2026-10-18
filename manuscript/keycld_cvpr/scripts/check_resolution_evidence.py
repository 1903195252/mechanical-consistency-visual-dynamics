"""Recompute the new manuscript's SI scores from case residuals, independently."""
import hashlib
import json
from pathlib import Path
import numpy as np

PAPER=Path(__file__).resolve().parents[1]
ROOT=PAPER.parents[1]
FIELDS=('geometry_m','velocity_m_s','acceleration_m_s2','force_balance_Nm','work_energy_J','prediction_m')
TOLERANCES=(.001,.01,.1,.25,.05,.01)


def verify():
    evidence=json.loads((PAPER/'sources/resolution_comparison.json').read_text())
    assert evidence['scoring_version']=='continuous-si-1'
    for path,digest in evidence['sources'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    source=ROOT/'output/planar_resolution/cloud/recovered_20261017/comparison/source_reports'
    ids=None; comparisons=0; checkpoint_hashes=[]
    for model in ('KeyCLD','CON'):
        target=next(r for r in evidence['runs'] if r['model']==model and r['width']==256)
        for track in ('native','reconstructed'):
            seed_scores=[];cover=joint=0
            for seed in (42,43,44):
                folder=source/f'{model.lower()}_256_seed{seed}'
                scores=json.loads((folder/'scores.json').read_text())
                train=json.loads((folder/'training_report.json').read_text())
                pred=json.loads((folder/'inference_report.json').read_text())
                assert train['full_objective'] and train['scientific_training_result']
                assert train['epochs']==(40 if model=='KeyCLD' else 50)
                assert train.get('updates',train.get('optimizer_updates'))==(32520 if model=='KeyCLD' else 2550)
                assert train['checkpoint_sha256']==pred['checkpoint_sha256']==scores['checkpoint_sha256']
                assert pred['completed']==pred['finite']==162
                if track=='native':checkpoint_hashes.append(train['checkpoint_sha256'])
                rows=scores[track]['rows'];assert len(rows)==108
                current={(r['case_id'],r['category'],r['condition']) for r in rows}
                if ids is None:ids=current
                assert ids==current
                cells={}
                for r in rows:
                    components=[]
                    for field,tol in zip(FIELDS,TOLERANCES):
                        value=r['metrics'][field]
                        expected=100/(1+(value/tol)**2) if value is not None and np.isfinite(value) else 0.
                        np.testing.assert_allclose(r['continuous_components_100'][field],expected,rtol=1e-10,atol=1e-12)
                        components.append(expected);comparisons+=1
                    np.testing.assert_allclose(r['continuous_score_100'],np.mean(components),rtol=1e-10,atol=1e-12)
                    cells.setdefault((r['category'],r['condition']),[]).append(np.mean(components))
                    cover+=r['coverage']==1;joint+=r['joint_pass']
                assert len(cells)==9 and all(len(v)==12 for v in cells.values())
                score=float(np.mean([np.mean(v) for v in cells.values()]));seed_scores.append(score)
                np.testing.assert_allclose(score,scores[track]['composite_score_100'])
            np.testing.assert_allclose(target[track]['score'],np.mean(seed_scores))
            np.testing.assert_allclose(target[track]['seed_sd'],np.std(seed_scores,ddof=1))
            assert target[track]['coverage']==cover and target[track]['joint']==joint==0
    assert len(set(checkpoint_hashes))==6
    native=[l.rstrip('\\ ').split(' & ') for l in (PAPER/'tables/native_comparison_main_rows.tex').read_text().splitlines() if ' & ' in l]
    reconstructed=[l.rstrip('\\ ').split(' & ') for l in (PAPER/'tables/dynamics_readout_main_rows.tex').read_text().splitlines() if ' & ' in l]
    merged=[l.rstrip('\\ ').split(' & ') for l in (PAPER/'tables/planar_audit_main_rows.tex').read_text().splitlines() if ' & ' in l]
    assert len(merged)==len(native)==len(reconstructed)==9
    for full,native_row,reconstructed_row in zip(merged,native,reconstructed):
        assert len(full)==15 and native_row[0]==reconstructed_row[0]==full[0]
        n,c=full[:10],full[10:]
        n[1]+='/324';n[3]+='/324';c[0]+='/324'
        n[4]='/'.join(f'{float(v):.1f}' for v in n[4].split('/'))
        assert n==native_row and c==reconstructed_row[1:], 'Merged table must retain every original cell'
    for index,run in zip((0,1,4,5),evidence['runs']):
        r=native[index];n=run['native'];v=run['visual'];c=run['reconstructed']
        assert r[0]==f"Video {run['width']} px" and r[1]==f"{run['finite']}/324" and r[3]=='0/324'
        assert r[2]==f"{v['rgb_mse']*1e4:.2f}" and r[7]==f"{n['means']['prediction']*1000:.1f}"
        assert r[-1]==f"${n['score']:.2f}\\pm{n['seed_sd']:.2f}$"
        r=reconstructed[index]
        assert r[1]==f"{c['coverage']}/324" and r[2]==f"{run['correction_mm']:.2f}"
        assert r[3]==f"{c['means']['dynamics']:.3f}" and r[4]==f"{c['means']['energy']:.3f}"
        assert r[5]==f"{c['means']['prediction']*1000:.2f}"
    for model in ('KeyCLD','CON'):
        ref=evidence['reader'][model+'/reference/phase_sg7d3']
        gen=evidence['reader'][model+'/generated/phase_sg7d3']
        assert ref['coverage']==324 and ref['joint']==198 and gen['coverage']==324 and gen['joint']==0
        assert {r['case_id'] for r in ref['rows']}=={r['case_id'] for r in gen['rows']}
        assert len({r['case_id'] for r in ref['rows']})==108
    # Compare the same raw Euclidean readout statistic, not the projected SI
    # maximum-channel statistic or a different differentiation window.
    localization=evidence['reader_localization']['results']
    report64=json.loads((ROOT/'output/closed_chain_eval/visual_calibration/paired_robust_planar_rgb/summary.json').read_text())
    for width in (64,256):
        for model in ('reference','KeyCLD','CON'):
            values=[];counts=[]
            for seed in ((42,) if model=='reference' else (42,43,44)):
                if width==64:
                    name='reference' if model=='reference' else f'{model.lower()}_seed{seed}'
                    rows=[r for r in report64['rows'] if r['model']==name]
                    field='point_euclidean_rmse_m'
                    assert all(r['geometry_and_point_frames']=='all future frames 3:90' for r in rows if r['finite'])
                else:
                    name='keycld' if model=='reference' else model.lower()
                    report=json.loads((source/f'{name}_256_seed{seed}/external_rgb/report.json').read_text())
                    rows=[r for r in report['rows'] if r['input']==('reference' if model=='reference' else 'generated')]
                    field='future_point_euclidean_rmse_m'
                assert {(r['case_id'],r['category'],r['condition']) for r in rows}==ids
                finite=[r for r in rows if r['finite']]
                values.append(float(np.mean([r[field] for r in finite]))*1000);counts.append(len(finite))
            target=localization[str(width)][model]
            np.testing.assert_allclose(target['seed_means_mm'],values,rtol=1e-12)
            np.testing.assert_allclose(target['point_euclidean_rmse_mm'],np.mean(values),rtol=1e-12)
            assert target['finite_by_seed']==counts and target['unique_cases']==108
    source_text=(PAPER/'main.tex').read_text()+(PAPER/'supplement.tex').read_text()
    assert 'All completed planar model results use 64' not in source_text
    assert 'training and reader calibration at 256 and 512 pixels remain future work' not in source_text
    main=(PAPER/'main.tex').read_text()
    for number in (f"{localization['64']['reference']['point_euclidean_rmse_mm']:.2f}",
                   f"{localization['256']['reference']['point_euclidean_rmse_mm']:.3f}",
                   f"{localization['256']['KeyCLD']['point_euclidean_rmse_mm']:.1f}",
                   f"{localization['256']['CON']['point_euclidean_rmse_mm']:.1f}"):
        assert number in main,number
    result=dict(completed_256_runs=6,full_budget_checkpoints=6,forecast_files_recorded=972,
                scored_256_test_cases=648,case_component_values_recomputed=comparisons,
                fixed_engineering_tolerances=list(TOLERANCES),matched_case_identities=True,
                main_video_rows_verified=4,merged_table_rows=9,merged_table_original_cells_preserved=135,
                common_reader_reference_cases=108,
                reference_joint_passes=66,model_joint_passes=0,reference_duplicates_not_independent=True,
                raw_localization_comparison_verified=True,raw_localization_future_frames=87,
                reference_localization_mm=[localization[str(w)]['reference']['point_euclidean_rmse_mm'] for w in (64,256)])
    return evidence,result


if __name__=='__main__':print(json.dumps(verify()[1],indent=2))
