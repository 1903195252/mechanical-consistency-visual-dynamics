"""Frozen inference: video runs open prefix files only; no evaluator references."""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
from keycld import util
from .model import ControlledKeyCLD
from .train import write, sha


def run(root,arm,seed):
    root=Path(root); out=root/'runs'/('%s_seed%d'%(arm,seed))
    report=json.loads((out/'training_report.json').read_text())
    assert report['epochs']==40 and sha(out/'params.p')==report['checkpoint_sha256']
    manifest=json.loads((root/'dataset_manifest.json').read_text())
    assert sha(root/'dataset_manifest.json')==report['dataset_manifest_sha256']
    with (out/'params.p').open('rb') as handle:
        params=pickle.load(handle)
    times=np.arange(8,713,dtype=np.float32)/240
    @jax.jit
    def initial_state(parameters,images,g,command,oracle_x,oracle_v):
        model=ControlledKeyCLD(g)
        if arm=='video_euler':
            maps=model.encoder(parameters,images)
            kp,_=util.map_to_keypoints(maps)
            x,v=util.finite_difference(kp.reshape(3,6),jnp.float32(1/30))
            position=x[0]; velocity=v[0]
        else:
            kp=oracle_x
            position=oracle_x[1].ravel(); velocity=oracle_v[1].ravel()
        velocity=util.project_velocity(model.constraint_fn,position,velocity)
        return jnp.r_[position,velocity],kp
    @jax.jit
    def forecast(parameters,initial,g,command):
        model=ControlledKeyCLD(g)
        history=odeint(partial(model.ode,parameters),initial,jnp.asarray(times),command,mxstep=4096)
        a=jax.vmap(lambda y:model.euler_lagrange(parameters,y[:6],y[6:],command))(history)
        return history[:,:6].reshape(-1,3,2),history[:,6:].reshape(-1,3,2),a.reshape(-1,3,2)
    records=[r for r in manifest['records'] if r['split'] in ('test','validation')]
    started=time.monotonic(); completed=[]; compiled=None
    failed_path=out/'inference_failures.json'
    failed=json.loads(failed_path.read_text()) if failed_path.exists() else {}
    for i,record in enumerate(records):
        folder=out/'predictions'/record['split']; folder.mkdir(parents=True,exist_ok=True)
        path=folder/(record['case_id']+'.npz')
        metadata=path.with_suffix('.json')
        if path.exists() and metadata.exists():
            meta=json.loads(metadata.read_text())
            if meta.get('checkpoint_sha256')!=report['checkpoint_sha256'] or sha(path)!=meta['prediction_sha256']:
                raise RuntimeError('Existing prediction has incompatible provenance')
            completed.append(meta); continue
        with np.load(root/record['inputs']) as data:
            images=data['x'].astype(np.float32)/255; g=data['geometry']; commands=data['action']
        assert images.shape==(3,64,64,3)
        oracle_x=oracle_v=np.zeros((3,3,2),dtype=np.float32)
        if arm!='video_euler':
            with np.load(root/record['state_labels']) as data:
                oracle_x=data['x'][:3]; oracle_v=data['v'][:3]
        write(out/'inference_status.json',{'stage':'initializing','case_id':record['case_id'],
              'case_started_unix_s':time.time(),'completed':i,'total':len(records)})
        initial,kp=jax.device_get(initial_state(params,images,g,commands,oracle_x,oracle_v))
        if record['case_id'] in failed:
            x=v=a=np.full((len(times),3,2),np.nan)
        else:
            if compiled is None:
                write(out/'inference_status.json',{'stage':'compiling','case_id':record['case_id'],
                      'case_started_unix_s':time.time(),'completed':i,'total':len(records)})
                compiled=forecast.lower(params,initial,g,commands).compile()
            write(out/'inference_status.json',{'stage':'integrating','case_id':record['case_id'],
                  'case_started_unix_s':time.time(),'completed':i,'total':len(records)})
            x,v,a=jax.device_get(compiled(params,initial,g,commands))
        np.savez_compressed(path,time=times.astype(float),points=x.astype(float)*.65+[.2,.35],
                            velocity=v.astype(float)*.65,acceleration=a.astype(float)*.65,
                            observed_keypoints=kp.astype(float)*.65+[.2,.35])
        meta={'case_id':record['case_id'],'checkpoint_sha256':report['checkpoint_sha256'],
              'prediction_sha256':sha(path),'input_sha256':record['inputs_sha256'],
              'native_derivative_hz':240,'observed_frames':3,'future_measured_actions_used':False,
              'state_prefix_used':arm!='video_euler','finite':bool(all(np.isfinite(y).all() for y in (x,v,a))),
              'failure_reason':failed.get(record['case_id']),
              'devices':[str(d) for d in jax.devices()], 'inference_source_sha256':sha(Path(__file__))}
        write(metadata,meta); completed.append(meta)
        write(out/'inference_status.json',{'stage':'predicting','completed':i+1,'total':len(records),'elapsed_s':time.monotonic()-started})
    write(out/'inference_report.json',{'stage':'complete','arm':arm,'seed':seed,'completed':len(completed),
          'finite':sum(x['finite'] for x in completed),'checkpoint_sha256':report['checkpoint_sha256'],
          'dataset_manifest_sha256':report['dataset_manifest_sha256'],'elapsed_s':time.monotonic()-started,
          'records':completed})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True)
    p.add_argument('--seed',type=int,required=True); a=p.parse_args();run(a.root,a.arm,a.seed)
