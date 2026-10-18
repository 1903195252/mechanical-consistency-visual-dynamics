"""Render verified post-hoc diagnostics without conflating raw and corrected tracks."""
import hashlib
import json
from pathlib import Path
import shutil
from prepare_unified_scoring import planar_estimates
PAPER=Path(__file__).resolve().parents[1];ROOT=PAPER.parents[1];EXT=ROOT/'output/con_corpus/dynamics_extension'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def collect():
    summary=json.loads((EXT/'summary.json').read_text());assert summary['stage']=='verified_complete'
    assert summary['prediction_files_verified']==972 and len(summary['paired_training'])==3
    assert len(summary['native_score_reproduction'])==15
    for r in summary['native_score_reproduction']:assert r['max_absolute_metric_difference']<1e-7
    for r in summary['paired_training']:assert r['identical_epoch_order'] and r['updates_each']==2550
    paths=[EXT/'summary.json',EXT/'protocol.json',EXT/'solver_protocol.json',EXT/'readout_calibration.json',
           EXT/'physical_case_metrics.csv',EXT/'collection_status.json',EXT/'source_snapshots.json',
           EXT/'figures/selection.json',EXT/'figures/video_comparison.png',EXT/'figures/video_comparison.gif',
           EXT/'figures/dynamics_example.pdf',EXT/'figures/dynamics_example_data.npz']
    paths+=sorted((EXT/'runs').glob('*/training_report.json'))+sorted((EXT/'runs').glob('*/inference_report.json'))
    paths+=sorted((EXT/'video_diagnosis').glob('*/report.json'))+sorted((EXT/'solver_diagnosis').glob('*/report.json'))+sorted((EXT/'physics_readout').glob('*/report.json'))
    assert json.loads((EXT/'collection_status.json').read_text())['stage']=='complete'
    for report in (EXT/'physics_readout').glob('*/report.json'):
        r=json.loads(report.read_text());assert len(r['rows'])==108
    return dict(summary=summary,sources=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in paths],
                interpretation='Post-hoc diagnostic supplement; geometric readout is an explicit intervention. Native SI missingness is retained.')
def main():
    evidence=collect();s=evidence['summary']
    continuous = planar_estimates()
    rows=[]
    for model,label in [('keycld','KeyCLD'),('con','CON')]:
        v=s['video'][model];image=v['image'];local=v['local']
        rows.append(' & '.join([label,f"{image['framewise_reconstruction']['mean']*1e4:.4f}",f"{image['prefix_forecast']['mean']*1e4:.4f}",
                                f"{image['prefix_forecast']['valid']}/324",f"{image['persistence']['mean']*1e4:.4f}",
                                f"{local['euler']['mean']*1000:.3f}",f"{local['rk4']['mean']*1000:.3f}",f"{local['latent_persistence']['mean']*1000:.3f}"])+r' \\')
    (PAPER/'tables/matched_video_rows.tex').write_text('\n'.join(rows)+'\n')
    labels={'keycld/video_euler':'KeyCLD Video','keycld/state_euler':'KeyCLD State--Euler','keycld/state_rk4':'KeyCLD State--RK4',
            'con/video_con_iae':'CON Video','con/state_con_iae':'CON State--Dopri5','con_integrator/state_euler_con_iae':'CON State--Euler','con_integrator/state_rk4_con_iae':'CON State--RK4'}
    rows=[]
    for name,label in labels.items():
        v=s['physics'][name];m=v['conditioned']['metrics']
        rows.append(' & '.join([label,f"{v['raw']['complete']}/324",f"{v['conditioned']['complete']}/324",f"{v['correction_rms_m']['mean']*1000:.4f}",
                                f"{m['force_balance_Nm']['mean']:.4f}",f"{m['work_energy_J']['mean']:.4f}",f"{m['prediction_m']['mean']*1000:.3f}"])+r' \\')
    (PAPER/'tables/geometry_readout_rows.tex').write_text('\n'.join(rows)+'\n')
    main_rows=[]
    for group in ['KeyCLD','CON']:
        if main_rows:
            main_rows.append(r'\midrule')
            main_rows.append(r'\multicolumn{6}{l}{\textit{'+group+r'}}\\')
        for name,label in labels.items():
            if not label.startswith(group+' '):continue
            v=s['physics'][name];m=v['conditioned']['metrics']
            correction_mm = v['correction_rms_m']['mean']*1000
            correction = r'$<0.001$' if correction_mm < 0.001 else f'{correction_mm:.2f}'
            main_rows.append(' & '.join([label[len(group)+1:],f"{v['conditioned']['complete']}/324",
                correction,f"{m['force_balance_Nm']['mean']:.3f}",
                f"{m['work_energy_J']['mean']:.3f}",f"{m['prediction_m']['mean']*1000:.2f}"])+r' \\')
    (PAPER/'tables/dynamics_readout_main_rows.tex').write_text('\n'.join(main_rows)+'\n')
    native_rows=[]
    metrics=('geometry_m','velocity_m_s','acceleration_m_s2','force_balance_Nm','work_energy_J','prediction_m')
    for group in ('KeyCLD','CON'):
        if native_rows:
            native_rows.append(r'\midrule')
            native_rows.append(r'\multicolumn{10}{l}{\textit{'+group+r'}}\\')
        for name,label in labels.items():
            if not label.startswith(group+' '):continue
            raw=s['physics'][name]['raw']; scores=raw['component_scores_100']; video='Video' in label
            model='keycld' if group=='KeyCLD' else 'con'
            finite=s['video'][model]['image']['prefix_forecast']['valid'] if video else raw['metrics']['prediction_m']['valid']
            rgb=f"{s['video'][model]['image']['prefix_forecast']['mean']*1e4:.2f}" if video else '--'
            cells=[label[len(group)+1:],f'{finite}/324',rgb,f"{raw['complete']}/324",
                   '/'.join(f'{scores[k]:.1f}' for k in metrics[:3])]
            cells += [f'{scores[k]:.2f}' if raw['metrics'][k]['valid'] else 'N/A' for k in metrics[3:]]
            assert raw['joint_passes']==0
            cells += ['0', f'{continuous[name.split("/")[-1]]["mean"]:.2f}']
            native_rows.append(' & '.join(cells)+r' \\')
    native_rows.extend([r'\midrule',r'Image persistence & 324/324 & 4.94 & -- & -- & -- & -- & -- & -- & -- \\'])
    (PAPER/'tables/native_comparison_main_rows.tex').write_text('\n'.join(native_rows)+'\n')
    rows=[]
    for kind in ['euler','rk4']:
        name='con_integrator/state_'+kind+'_con_iae';v=s['physics'][name]
        rows.append(' & '.join([kind.upper(),f"{v['raw']['metrics']['prediction_m']['mean']*1000:.3f}",f"{v['raw']['metrics']['geometry_m']['mean']*1000:.3f}",f"{v['raw']['complete']}/324"])+r' \\')
    (PAPER/'tables/con_integrator_rows.tex').write_text('\n'.join(rows)+'\n')
    rows=[]
    for model in ['keycld','con']:
        for name,v in s['solver'][model].items():
            rows.append(' & '.join([model.upper(),name.replace('_',' / '),f"{round(v['finite']['mean']*324)}/324",f"{v['rgb_mse']['mean']*1e4:.5f}",
                                    f"{v['native_difference_rmse_m']['mean']*1000:.4g} ({v['native_difference_rmse_m']['valid']})"])+r' \\')
    (PAPER/'tables/inference_solver_rows.tex').write_text('\n'.join(rows)+'\n')
    for source,target in [('video_comparison.png','matched_video_example.png'),('dynamics_example.pdf','conditioned_dynamics_example.pdf')]:
        shutil.copy2(EXT/'figures'/source,PAPER/'figures'/target)
    (PAPER/'sources/dynamics_extension_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return {'dynamics_extension':'sources/dynamics_extension_evidence.json: six new paired training runs, matched video diagnosis, inference sweeps, and explicit geometry-conditioned SI readout',
            'corpus_paired_residuals':'sources/integrator_comparison_data.json: paired KeyCLD and CON training, with explicit raw/readout tracks and common SI axes',
            'dynamics_readout_main_table':'tables/dynamics_readout_main_rows.tex: seven arms with readout coverage, coordinate correction, torque, energy and point errors; source is dynamics_extension_evidence.json',
            'dynamics_extension_results':'output/con_corpus/dynamics_extension/results.md: raw / additional readout separation, coverage, correction size, paired intervals and reproducibility'}
if __name__=='__main__':
    additions=main();p=PAPER/'sources/evidence_map.json';mapping=json.loads(p.read_text());mapping.update(additions);p.write_text(json.dumps(mapping,indent=2)+'\n');print(json.dumps(additions,indent=2))
