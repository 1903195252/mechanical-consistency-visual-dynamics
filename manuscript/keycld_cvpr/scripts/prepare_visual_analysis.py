"""Evidence for appearance, motion-region errors and short-window visual prediction."""
import hashlib
import json
import numpy as np
from prepare_artifacts import ROOT, PAPER, read


def build():
    folder=ROOT/'output/keycld_fourbar/diagnosis'
    rows=sorted(read(folder/'learned_rows.json'),key=lambda r:r['index'])
    report=read(folder/'report.json')
    assert [r['index'] for r in rows]==list(range(500,550))
    with np.load(folder/'visual_example.npz') as z:
        shape=z['reference_rgb'].shape
        assert shape==(50,64,64,3) and z['reconstructed_rgb'].shape==shape
    fractions=np.array([r['motion_mask_pixels']/(shape[1]*shape[2]) for r in rows])
    full=np.array([r['image_mse']['odeint'] for r in rows])
    motion=np.array([r['foreground_mse']['odeint'] for r in rows])
    contribution=fractions*motion/full
    assert np.all((fractions>0)&(fractions<1)) and np.all((contribution>=0)&(contribution<=1+1e-6))
    recon=report['summary']['image_mse']['framewise_oracle']['mean']
    recon_motion=report['summary']['foreground_mse']['framewise_oracle']['mean']
    errors=np.array([[r['local_latent_prediction'][arm]['four_step_rmse_m']*1000
                      for arm in ('euler_1','rk4_4','latent_persistence')] for r in rows])
    rng=np.random.default_rng(20261013);draws=rng.integers(0,50,(2000,50))
    local={}
    for j,arm in enumerate(('euler_1','rk4_4')):
        differences=errors[:,j]-errors[:,2]
        local[arm]={'mean_RMSE_mm':float(errors[:,j].mean()),
                    'mean_minus_persistence_mm':float(differences.mean()),
                    'paired_bootstrap_95pct_mm':np.percentile(differences[draws].mean(axis=1),[2.5,97.5]).tolist(),
                    'trajectories_better_than_persistence':int(np.sum(differences<0))}
    stats=report['summary']['visual_state']
    declared=stats['encoded_rmse_declared_m']['mean'];permuted=stats['encoded_rmse_validation_permutation_m']['mean']
    source_paths=[folder/'learned_rows.json',folder/'report.json',folder/'visual_example.npz',
                  ROOT/'output/keycld_fourbar/source/keycld/losses.py',
                  ROOT/'output/keycld_corpus/runtime/keycld/losses.py']
    assert source_paths[-1].read_bytes()==source_paths[-2].read_bytes()
    validation=[]
    for seed in (42,43,44):
        p=ROOT/'output/keycld_corpus/runs'/('video_euler_seed%d'%seed)/'scores.json'
        score=read(p);validation.append(score['validation_keypoint_rmse_m']*1000);source_paths.append(p)
    result={'sources':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths],
        'test_trajectories':50,'training_seed':42,'trajectory_ids':[r['index'] for r in rows],
        'image_size_px':[64,64],'reconstruction_MSE':recon,'forecast_MSE':float(full.mean()),
        'motion_reconstruction_MSE':recon_motion,'motion_forecast_MSE':float(motion.mean()),
        'forecast_reconstruction_MSE_ratio':float(full.mean()/recon),
        'motion_forecast_reconstruction_MSE_ratio':float(motion.mean()/recon_motion),
        'motion_pixel_fraction_mean':float(fractions.mean()),
        'motion_forecast_squared_error_share_mean':float(contribution.mean()),
        'spatial_aggregation':'Per-trajectory pixel fraction and squared-error share, then arithmetic mean over 50 trajectories.',
        'motion_mask':'Mean RGB temporal variance over all reference frames > 1e-4; evaluation only.',
        'paired_local_prediction':local,'latent_persistence_four_step_RMSE_mm':float(errors[:,2].mean()),
        'local_window_definition':'44 reset windows per trajectory; initial encoded positions and central-difference velocities from true images; four prediction steps.',
        'bootstrap':'Post-hoc paired resampling of 50 test trajectories, 2000 draws, seed 20261013; fixed checkpoint.',
        'global_permutation_reduction_percent':float(100*(1-permuted/declared)),
        'corpus_validation_RMSE_mm_by_seed':validation,
        'corpus_validation_scope':'Validation-set errors used to select the global permutation; test coverage is reported separately.',
        'reconstruction_panel':'Existing example 500, frames 2/9/49: reference, framewise reconstruction, prefix-conditioned forecast.'}
    table=[]
    for name,key in [('Framewise recon.','framewise_oracle'),('Video forecast','odeint'),('Last-frame persistence','persistence'),('Training-mean image','training_mean')]:
        table.append('%s & %.3g & %.3g & %.2f \\\\'%(name,report['summary']['image_mse'][key]['mean'],report['summary']['foreground_mse'][key]['mean'],report['summary']['vpt'][key]['mean']))
    (PAPER/'tables/visual_control_rows.tex').write_text('\n'.join(table)+'\n')
    (PAPER/'sources/visual_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':build()
