"""Freeze the paper-procedure checkpoint, then evaluate untouched trajectories."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import numpy as np
from keycld.models import predict
from keycld import util
from learned_common import load_model
from train_paper import load_observation,write_status


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True);args=parser.parse_args();root=args.root
    sys.path.insert(0,str(args.assets))
    from prepare import points
    from score import metrics
    start=time.monotonic();model=load_model(root)
    geometry=json.loads((root/'geometry.json').read_text());scale=geometry['half_extent_m'];center=np.asarray(geometry['world_xz_center_m'])
    threshold=json.loads((root/'vpt_thresholds.json').read_text())['independent_test_train_only']
    report=json.loads((root/'training_report.json').read_text())
    assert report['epochs_completed']==40 and report['optimizer_updates']==18000
    @jax.jit
    def forecast(parameters,images,t,action):
        heatmap=model.encoder(parameters,images[:3]);keypoints,_=util.map_to_keypoints(heatmap)
        box=[]
        def capture(f,x,t,action):
            history=odeint(f,x,t,action);box.append(history);return history
        predicted=predict(model,parameters,t,keypoints,action,solver=capture)
        rendered,_=model.renderer(parameters,predicted)
        history=box[0]
        full=jnp.concatenate([jnp.concatenate([keypoints[0].ravel(),history[0,6:]])[None],history])
        acceleration=jax.vmap(lambda y:model.euler_lagrange(parameters,y[:6],y[6:],action))(full)
        return predicted,rendered,full[:,6:].reshape(-1,3,2),acceleration.reshape(-1,3,2)
    conditions=[('random_initialization','initial_params.p',range(500,550)),
                ('trained_in_distribution','trained_params.p',range(500,550)),
                ('trained_heavy_payload','trained_params.p',range(600,620)),
                ('trained_friction','trained_params.p',range(700,720))]
    results={};example_outputs=[]
    for label,checkpoint,indices in conditions:
        with (root/checkpoint).open('rb') as handle:params=pickle.load(handle)
        rows=[];runs=[];predictions=[]
        write_status(root,'evaluating',condition=label,completed=0,total=len(indices))
        for i,index in enumerate(indices):
            run=load_observation(root,index)
            kp,images,v,a=jax.device_get(forecast(params,run['x'],run['t'],run['action']))
            finite=all(np.isfinite(array).all() for array in [kp,images,v,a])
            image_error=np.mean((images-run['x'])**2,axis=(1,2,3))
            row={'index':index,'finite':finite,'future_image_mse':float(np.mean(image_error[3:])) if finite else None,
                 'vpt_index':float(util.calculate_vpt(threshold,[run],[{'x_recon':images}])[0])}
            if finite:
                with np.load(root/'truth'/f'run_{index:04d}.npz') as data:truth={key:data[key] for key in data.files}
                c=json.loads((root/'truth'/f'run_{index:04d}.json').read_text())
                xy,xyv,xya=points(truth['q'],truth['qdot'],truth['qddot'],c)
                reference={'keypoints_world_xz':xy,'q':truth['q']}
                actions=np.tile([run['action'][0],*c['external_force_xz_N']],(50,1))
                diagnostic,_=metrics(kp[:,:2]*scale+center,v[:,:2]*scale,a[:,:2]*scale,truth['time'],actions,reference,c)
                diagnostic['midpoint_relation_rmse_m']=float(np.sqrt(np.mean((kp[:,2]-(kp[:,0]+kp[:,1])/2)**2))*scale)
                row['latent_coordinate_physics_diagnostic']=diagnostic
            rows.append(row);runs.append(run);predictions.append({'x_recon':images})
            if label=='trained_in_distribution' and i<4:
                example_outputs.append({'index':index,'reference_rgb':np.rint(run['x']*255).clip(0,255).astype(np.uint8),
                                        'prediction_rgb':np.rint(images*255).clip(0,255).astype(np.uint8),
                                        'time':run['t'],'keypoints_normalized':kp,
                                        'keypoint_velocity_normalized_s':v,'keypoint_acceleration_normalized_s2':a})
            if (i+1)%10==0:
                write_status(root,'evaluating',condition=label,completed=i+1,total=len(indices))
                print('Evaluated',label,i+1,'of',len(indices),flush=True)
        vpt=util.calculate_vpt(threshold,runs,predictions)
        results[label]={'run_count':len(rows),'vpt_mean_index':float(vpt[0]),'vpt_std_index':float(vpt[1]),
                        'finite_runs':sum(row['finite'] for row in rows),
                        'future_image_mse_mean':float(np.mean([row['future_image_mse'] for row in rows if row['finite']])) if any(row['finite'] for row in rows) else None,
                        'runs':rows}
        (root/'evaluation_progress.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    for i,example in enumerate(example_outputs):np.savez_compressed(root/f'test_example_{i}.npz',**example)
    final={'training_performed':True,'checkpoint_epoch':40,'optimizer_updates':18000,
           'evaluation_protocol':'paper 50-frame, first-three-frame, official odeint/VPT; additional untouched test split',
           'vpt_threshold_from_training_only':threshold,'vpt_note':'Official function returns indices 0..49 for 50 frames.',
           'physics_interpretation':'Latent coordinates are not guaranteed to be physical hinges. Extra residuals are diagnostics until semantic mapping is verified; paper VPT is the primary metric.',
           'conditions':results,'elapsed_seconds':time.monotonic()-start}
    (root/'evaluation_report.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    write_status(root,'complete',trained_epochs=40,evaluated_runs=sum(len(value['runs']) for value in results.values()),
                 primary_test_vpt_mean=results['trained_in_distribution']['vpt_mean_index'])
    print('Evaluation completed:',json.dumps({key:{k:v for k,v in item.items() if k!='runs'} for key,item in results.items()}),flush=True)


if __name__=='__main__':main()
