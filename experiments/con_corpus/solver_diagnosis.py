"""Paired inference solver sweeps: frozen weights, identical native initial state."""
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
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'output/keycld_corpus';OUT=ROOT/'output/con_corpus';EXT=OUT/'dynamics_extension'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def rms(a,b):
    d=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    return float(np.sqrt(np.mean(d*d))) if np.isfinite(d).all() else None

def main(name,seed):
    started=time.monotonic();key=name=='keycld';arm='video_euler' if key else 'video_con_iae'
    folder=(DATA if key else OUT)/'runs'/f'{arm}_seed{seed}';dest=EXT/'solver_diagnosis'/f'{name}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((EXT/'solver_protocol.json').read_text());manifest=json.loads((DATA/'dataset_manifest.json').read_text())
    report=json.loads((folder/'training_report.json').read_text());assert sha(folder/'params.p')==report['checkpoint_sha256']
    identity=dict(model=name,seed=seed,checkpoint_sha256=report['checkpoint_sha256'],protocol_sha256=sha(EXT/'solver_protocol.json'),source_sha256=sha(Path(__file__)))
    with (folder/'params.p').open('rb') as h:params=pickle.load(h)
    if key:
        from experiments.keycld_corpus.model import ControlledKeyCLD
        def field(y,g,u):return ControlledKeyCLD(g).ode(params,y,0.,u)
        @jax.jit
        def decode(z,g):return ControlledKeyCLD(g).renderer(params,z.astype(jnp.float32).reshape(-1,3,2))[0]
        permutation=json.loads((folder/'scores.json').read_text())['reader_permutation_from_validation']
        def points(z):return np.asarray(z).reshape(-1,3,2)[:,permutation]*.65+[.2,.35]
        dim=6
    else:
        from experiments.con_corpus.common import model
        network=model();mapping=np.asarray(json.loads((folder/'readout.json').read_text())['affine']);dim=8
        def field(y,g,u):
            command=jnp.r_[u,g]/jnp.asarray([5,12,12,1,1]+[1]*9)
            return network.apply({'params':params},y,command,method=network.forward_dynamics)
        @jax.jit
        def decode(z,g):return (network.apply({'params':params},z.astype(jnp.float32),method=network.decode)+1)/2
        def points(z):return (np.asarray(z)@mapping[:8]+mapping[8]).reshape(-1,3,2)*.65+[.2,.35]
        permutation=[0,1,2]
    def factory(kind):
        @jax.jit
        def solve(y,g,u,substeps):
            def interval(y,_):
                def sub(_,y):
                    h=1/(30*substeps);k1=field(y,g,u)
                    if kind=='euler':return y+h*k1
                    k2=field(y+h*k1/2,g,u);k3=field(y+h*k2/2,g,u);k4=field(y+h*k3,g,u)
                    return y+h*(k1+2*k2+2*k3+k4)/6
                y=jax.lax.fori_loop(0,substeps,sub,y);return y,y
            _,history=jax.lax.scan(interval,y,None,length=88)
            return jnp.concatenate([y[None],history])
        return solve
    solvers={s:factory(s) for s in ['euler','rk4']}
    from experiments.keycld_corpus.physics_score import residuals
    pack=json.loads((OUT/'evaluator/manifest.json').read_text());refs={r['case_id']:r for r in pack['references']}
    rows=[]
    for i,r in enumerate(x for x in manifest['records'] if x['split']=='test'):
        output=dest/(r['case_id']+'.json')
        if output.exists():
            row=json.loads(output.read_text());assert row['identity']==identity;rows.append(row);continue
        if time.time()>=deadline()-90:write(dest/'status.json',dict(stage='paused_at_budget',completed=i));return
        write(dest/'status.json',dict(stage='integrating',completed=i,total=108,case_id=r['case_id']))
        # Prefix data and stored predicted initial state are the only forecast inputs.
        with np.load(DATA/r['inputs']) as z:g=z['geometry'];u=z['action']
        path=folder/'predictions/test'/(r['case_id']+'.npz');meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['prediction_sha256']
        with np.load(path) as z:
            if key:
                initial=np.r_[((z['points'][0]-[.2,.35])/.65).ravel(),(z['velocity'][0]/.65).ravel()].astype(np.float32)
            else:initial=z['latent_state'][0].astype(np.float64)
            native=z['points'][16::8,permutation]
        reinitialized = bool(key and not np.isfinite(initial).all())
        if reinitialized:
            # Original timeout placeholders contain no initial state. Reconstruct
            # solely from the same three observed frames, using the original initializer.
            from keycld import util
            with np.load(DATA/r['inputs']) as z:images=z['x'].astype(np.float32)/255
            net=ControlledKeyCLD(g);kp=util.map_to_keypoints(net.encoder(params,images))[0]
            x,v=util.finite_difference(kp.reshape(3,6),jnp.float32(1/30))
            vv=util.project_velocity(net.constraint_fn,x[0],v[0]);initial=np.asarray(jnp.r_[x[0],vv])
        forecasts={f'{kind}_{n}':np.asarray(solvers[kind](initial,g,u,n)) for kind in ['euler','rk4'] for n in [1,4,16]}
        # Future targets open only after all predicted trajectories are frozen.
        with np.load(DATA/r['observations']) as z:rgb=z['x'].astype(float)/255
        with np.load(DATA/r['state_labels']) as z:truth=z['x'][3:].astype(float)*.65+[.2,.35]
        mask=np.mean(np.var(rgb,axis=0),axis=-1)>1e-4;c=json.loads((ROOT/refs[r['case_id']]['parameters_json']).read_text())['config']
        values={}
        for method,history in forecasts.items():
            finite=bool(np.isfinite(history).all());value=dict(finite=finite,rgb_mse=None,motion_mse=None,point_rmse_m=None,native_difference_rmse_m=None,geometry_max_m=None,constraint_value_drift_max_m=None)
            if finite:
                xy=points(history[:,:dim]);dummy=np.zeros_like(xy);rq=residuals(xy,dummy,dummy,c)[0]
                decoded=np.concatenate([np.asarray(decode(chunk,g)) for chunk in np.split(history[2:,:dim],3)])
                rendered=np.rint(np.clip(decoded,0,1)*255)/255
                if np.isfinite(rendered).all():
                    err=(rendered-rgb[3:])**2;value.update(rgb_mse=float(err.mean()),motion_mse=float(err[:,mask].mean()))
                value.update(point_rmse_m=rms(xy[2:],truth),native_difference_rmse_m=rms(xy[2:],native),
                             geometry_max_m=float(np.abs(rq[2:]).max()),constraint_value_drift_max_m=float(np.abs(rq[2:]-rq[:1]).max()))
            values[method]=value
        archive=output.with_suffix('.npz')
        np.savez_compressed(archive,time=np.arange(1,90)/30,**forecasts)
        row={k:r[k] for k in ['case_id','assembly_id','category','condition']};row.update(initial_source='same_prefix_initializer_for_native_timeout' if reinitialized else 'stored_native_initial_state',trajectory_sha256=sha(archive),identity=identity,native_prediction_sha256=meta['prediction_sha256'],native_finite=meta['finite'],methods=values)
        write(output,row);rows.append(row)
        if i%10==0:print(json.dumps(dict(model=name,seed=seed,completed=i+1,total=108)),flush=True)
    summary={}
    for method in rows[0]['methods']:
        summary[method]={}
        for metric in rows[0]['methods'][method]:
            groups=defaultdict(list)
            for r in rows:
                value=r['methods'][method][metric]
                if value is not None and np.isfinite(value):groups[(r['category'],r['condition'])].append(float(value))
            summary[method][metric]=dict(valid=sum(map(len,groups.values())),total=len(rows),macro_mean=float(np.mean([np.mean(v) for v in groups.values()])) if groups else None)
    write(dest/'report.json',dict(identity=identity,stage='complete',test_cases=len(rows),rows=rows,summary=summary,elapsed_s=time.monotonic()-started))
    write(dest/'status.json',dict(stage='complete',completed=len(rows)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['keycld','con'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();main(a.model,a.seed)
