"""Same-corpus frozen video reconstruction, forecast, and reset-window diagnosis."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from experiments.runtime import deadline
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
import numpy as np
if not hasattr(jax,'tree_map'):jax.tree_map=jax.tree_util.tree_map
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'output/keycld_corpus'; OUT=ROOT/'output/con_corpus'; EXT=OUT/'dynamics_extension'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def quantize(x):return np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def rms(a,b):
    d=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    return float(np.sqrt(np.mean(d*d))) if np.isfinite(d).all() else None

def fixed(field,y,solver):
    def interval(y,_):
        if solver=='euler':y=y+field(y)/30
        else:
            def sub(_,y):
                h=1/120;k1=field(y);k2=field(y+h*k1/2);k3=field(y+h*k2/2);k4=field(y+h*k3)
                return y+h*(k1+2*k2+2*k3+k4)/6
            y=jax.lax.fori_loop(0,4,sub,y)
        return y,y
    _,history=jax.lax.scan(interval,y,None,length=4)
    return jnp.concatenate([y[None],history])

def main(model_name,seed):
    started=time.monotonic(); key=model_name=='keycld'
    manifest=json.loads((DATA/'dataset_manifest.json').read_text()); proto=json.loads((EXT/'protocol.json').read_text())
    assert sha(DATA/'dataset_manifest.json')==proto['dataset_manifest_sha256']
    arm='video_euler' if key else 'video_con_iae';folder=(DATA if key else OUT)/'runs'/f'{arm}_seed{seed}'
    destination=EXT/'video_diagnosis'/f'{model_name}_seed{seed}';destination.mkdir(parents=True,exist_ok=True)
    report=json.loads((folder/'training_report.json').read_text());assert sha(folder/'params.p')==report['checkpoint_sha256']
    identity={'checkpoint_sha256':report['checkpoint_sha256'],'protocol_sha256':sha(EXT/'protocol.json'),'source_sha256':sha(Path(__file__))}
    with (folder/'params.p').open('rb') as handle:params=pickle.load(handle)
    if key:
        from keycld import util
        from experiments.keycld_corpus.model import ControlledKeyCLD
        permutation=json.loads((folder/'scores.json').read_text())['reader_permutation_from_validation']
        @jax.jit
        def encode(images,g):
            net=ControlledKeyCLD(g);return util.map_to_keypoints(net.encoder(params,images))[0].reshape(-1,6)
        @jax.jit
        def render(z,g):return ControlledKeyCLD(g).renderer(params,z.reshape(-1,3,2))[0]
        @jax.jit
        def local(z,images,g,u):
            net=ControlledKeyCLD(g);v=(z[2:]-z[:-2])/(2/30)
            v=jax.vmap(lambda x,v:util.project_velocity(net.constraint_fn,x,v))(z[1:-1],v)
            initial=jnp.concatenate([z[1:-1],v],axis=1)[:84]
            field=lambda y:net.ode(params,y,0.,u)
            return {s:jax.vmap(lambda y:fixed(field,y,s))(initial)[:,:,:6] for s in ['euler','rk4']}
        def points(z):return np.asarray(z).reshape(*np.asarray(z).shape[:-1],3,2)[...,permutation,:]*.65+[.2,.35]
    else:
        from experiments.con_corpus.common import model
        network=model();mapping=np.asarray(json.loads((folder/'readout.json').read_text())['affine'])
        @jax.jit
        def encode(images,g):return network.apply({'params':params},images*2-1,method=network.encode)
        @jax.jit
        def render(z,g):return (network.apply({'params':params},z.astype(jnp.float32),method=network.decode)+1)/2
        @jax.jit
        def local(z,images,g,u):
            fun=lambda x:network.apply({'params':params},x,method=network.encode)
            def velocity(chunk):
                center,delta=chunk
                return jax.jvp(fun,(center,), (delta,))[1]
            centers=(images[1:85]*2-1).reshape(3,28,64,64,3)
            differences=((images[2:86]-images[:84])*2/(2/30)).reshape(3,28,64,64,3)
            v=jax.lax.map(velocity,(centers,differences)).reshape(84,8)
            initial=jnp.concatenate([z[1:85],v],axis=1)
            command=jnp.r_[u,g]/jnp.array([5,12,12,1,1]+[1]*9)
            field=lambda y:network.apply({'params':params},y,command,method=network.forward_dynamics)
            return {s:jax.vmap(lambda y:fixed(field,y,s))(initial)[:,:,:8] for s in ['euler','rk4']}
        def points(z):
            z=np.asarray(z);p=z@mapping[:8]+mapping[8]
            return p.reshape(*z.shape[:-1],3,2)*.65+[.2,.35]
    # Training-only null controls: no test-specific fit or tuning.
    mean_file=EXT/'training_means.npz'
    if not mean_file.exists():
        total=np.zeros((64,64,3));p_sum=np.zeros((3,2));count=0
        for r in manifest['records']:
            if r['split']!='train':continue
            with np.load(DATA/r['observations']) as z:total+=z['x'].astype(float).sum(axis=0)/255;n=len(z['x'])
            with np.load(DATA/r['state_labels']) as z:p_sum+=(z['x'][:n].astype(float)*.65+[.2,.35]).sum(axis=0)
            count+=n
        tmp=mean_file.with_suffix('.partial')
        with tmp.open('wb') as handle:np.savez_compressed(handle,image=total/count,points=p_sum/count,frames=count)
        tmp.replace(mean_file)
    with np.load(mean_file) as z:mean_image=z['image'];mean_points=z['points']
    rows=[];records=[r for r in manifest['records'] if r['split']=='test']
    for i,r in enumerate(records):
        path=destination/(r['case_id']+'.json')
        if path.exists():
            row=json.loads(path.read_text());assert row['identity']==identity;rows.append(row);continue
        if time.time()>=deadline()-90:
            write(destination/'status.json',dict(stage='paused_at_budget',completed=i,total=len(records)));return
        write(destination/'status.json',dict(stage='diagnosing',case_id=r['case_id'],completed=i,total=len(records),elapsed_s=time.monotonic()-started))
        assert sha(DATA/r['observations'])==r['observations_sha256']
        with np.load(DATA/r['observations']) as z:images=z['x'].astype(np.float32)/255;g=z['geometry'];u=z['action']
        with np.load(DATA/r['state_labels']) as z:truth=z['x'].astype(float)*.65+[.2,.35]
        z=np.concatenate([np.asarray(encode(chunk,g)) for chunk in np.split(images,3)])
        reconstruction=quantize(np.concatenate([np.asarray(render(chunk,g)) for chunk in np.split(z,3)]))
        pred_path=folder/'predictions/test'/(r['case_id']+'.npz');metadata=json.loads(pred_path.with_suffix('.json').read_text())
        assert sha(pred_path)==metadata['prediction_sha256']
        predicted=None
        if metadata['finite']:
            with np.load(pred_path) as p:
                if key:
                    native=((p['points'][16::8]-[.2,.35])/.65).reshape(87,6).astype(np.float32)
                    predicted=quantize(np.concatenate([np.asarray(render(chunk,g)) for chunk in np.split(native,3)]))
                else:predicted=p['rgb_prediction']
        target=images[3:].astype(float);mask=np.mean(np.var(images.astype(float),axis=0),axis=-1)>1e-4
        assert mask.any()
        options={'framewise_reconstruction':reconstruction[3:].astype(float)/255,
                 'prefix_forecast':None if predicted is None else predicted.astype(float)/255,
                 'persistence':np.broadcast_to(images[2],target.shape),
                 'train_mean_image':np.broadcast_to(mean_image,target.shape)}
        mse={};masked={};fraction={}
        for name,p in options.items():
            if p is None:mse[name]=masked[name]=fraction[name]=None;continue
            squared=(p-target)**2;mse[name]=float(squared.mean());masked[name]=float(squared[:,mask].mean())
            fraction[name]=float(squared[:,mask].sum()/max(squared.sum(),1e-30))
        forecasts=jax.device_get(local(jnp.asarray(z),jnp.asarray(images),g,u))
        latent_targets=np.stack([z[i:i+5] for i in range(1,85)])
        physical_targets=np.stack([truth[i:i+5] for i in range(1,85)])
        forecasts['latent_persistence']=np.repeat(latent_targets[:,:1],5,axis=1)
        local_scores={}
        for name,p in forecasts.items():
            local_scores[name]={}
            for n in [1,4]:
                local_scores[name][f'step{n}_latent_rmse']=rms(p[:,n],latent_targets[:,n])
                local_scores[name][f'step{n}_readout_self_consistency_rmse_m']=rms(points(p[:,n]),points(latent_targets[:,n]))
                local_scores[name][f'step{n}_true_point_rmse_m']=rms(points(p[:,n]),physical_targets[:,n])
        row={k:r[k] for k in ['case_id','assembly_id','category','condition']}
        row.update(identity=identity,prediction_sha256=metadata['prediction_sha256'],image_mse=mse,motion_mask_mse=masked,
                   motion_mask_error_fraction=fraction,motion_mask_fraction=float(mask.mean()),reset_windows=84,
                   local_prediction=local_scores,encoded_point_rmse_m=rms(points(z),truth),
                   train_mean_point_rmse_m=rms(mean_points[None],truth),forecast_finite=metadata['finite'])
        write(path,row);rows.append(row)
        if i%10==0:print(json.dumps(dict(model=model_name,seed=seed,completed=i+1,total=len(records))),flush=True)
        if i==0:
            np.savez_compressed(destination/'example.npz',target=quantize(images),reconstruction=reconstruction,
                                forecast=np.zeros((0,64,64,3),np.uint8) if predicted is None else predicted,
                                encoded_points=points(z),truth_points=truth)
    def aggregate(values):
        groups=defaultdict(list)
        for r,v in zip(rows,values):
            if v is not None and np.isfinite(v):groups[(r['category'],r['condition'])].append(v)
        return dict(valid=sum(map(len,groups.values())),total=len(rows),macro_mean=float(np.mean([np.mean(v) for v in groups.values()])) if groups else None)
    summary={section:{name:aggregate([r[section][name] for r in rows]) for name in rows[0][section]}
             for section in ['image_mse','motion_mask_mse','motion_mask_error_fraction']}
    for field in ['encoded_point_rmse_m','train_mean_point_rmse_m','motion_mask_fraction']:summary[field]=aggregate([r[field] for r in rows])
    summary['local_prediction']={method:{field:aggregate([r['local_prediction'][method][field] for r in rows])
                              for field in rows[0]['local_prediction'][method]} for method in rows[0]['local_prediction']}
    result=dict(model=model_name,seed=seed,identity=identity,stage='complete',test_cases=len(rows),rows=rows,summary=summary,
                elapsed_s=time.monotonic()-started,training_means_sha256=sha(mean_file),image_metrics='MSE in [0,1], reconstructions and decoded forecasts rounded to 8-bit',
                reconstruction_uses_future_images=True,reset_uses_future_images=True,prefix_forecast_uses_future_images=False,
                local_latent_units='model-specific; do not compare latent RMSE across models')
    write(destination/'report.json',result);write(destination/'status.json',dict(stage='complete',completed=len(rows)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['keycld','con'],required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();main(a.model,a.seed)
