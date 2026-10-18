"""Frozen, three-frame forecasts and a separately labeled affine readout probe."""
import argparse
import json
from pathlib import Path
import time

from .common import OUT, DATA, data, jax, jnp, model, state_model, setup, load_params, sha, write
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve
import numpy as np


def fit_reader(network, parameters, manifest, folder):
    path = folder/'readout.json'
    if path.exists():
        obj = json.loads(path.read_text())
        assert obj['checkpoint_sha256'] == sha(folder/'params.p')
        return np.asarray(obj['affine'])
    encode = jax.jit(lambda images: network.apply({'params': parameters}, images, method=network.encode))
    gram=np.zeros((9,9)); rhs=np.zeros((9,6)); n=0
    for r in manifest['records']:
        if r['split'] != 'train':
            continue
        item=data(r,state=True); z=np.asarray(encode(item['rendering_ts']))
        design=np.c_[z,np.ones(len(z))]; target=item['points'][:,:6]
        gram+=design.T@design; rhs+=design.T@target; n+=len(z)
    mapping=np.linalg.lstsq(gram,rhs,rcond=1e-10)[0]
    errors=[]; geometry=[]
    for r in manifest['records']:
        if r['split']!='validation':
            continue
        item=data(r,state=True); z=np.asarray(encode(item['rendering_ts'][:len(item['points'])]))
        prediction=np.c_[z,np.ones(len(z))]@mapping
        errors.append(float(np.sqrt(np.mean((prediction-item['points'][:,:6])**2))*.65))
    write(path, {'affine':mapping.tolist(), 'training_points':n,
          'validation_mean_point_rms_m':float(np.mean(errors)), 'validation_max_case_point_rms_m':max(errors),
          'checkpoint_sha256':sha(folder/'params.p'), 'fit_partitions':['train'],
          'test_labels_used':False, 'predictor_uses_reader':False,
          'track':'privileged downstream affine latent readout; independent of RGB video-reader ranking'})
    return mapping


OUT = OUT / "dynamics_extension"

def run(arm,seed):
    protocol,manifest=setup(); folder=OUT/'runs'/f'{arm}_seed{seed}'
    parameters,report=load_params(folder); privileged=arm.startswith('state_')
    network=state_model() if privileged else model()
    mapping=None if privileged else fit_reader(network,parameters,manifest,folder)
    times=jnp.arange(8,713,dtype=jnp.float64)/240
    @jax.jit
    def initial(images,points,velocity):
        if privileged:
            return jnp.r_[points[1],velocity[1]]
        encode=lambda x:network.apply({'params':parameters},x,method=network.encode)
        z,v=jax.jvp(encode,(images[1:2],),((images[2:3]-images[:1])/(2/30),))
        return jnp.r_[z[0],v[0]]
    @jax.jit
    def forecast(x,command):
        if privileged:
            fun=lambda t,y,u:network.apply({'params':parameters},y,u)
        else:
            fun=lambda t,y,u:network.apply({'params':parameters},y,u,method=network.forward_dynamics)
        sol=diffeqsolve(ODETerm(fun),Dopri5(),t0=times[0],t1=times[-1],dt0=.001,
                       y0=x.astype(jnp.float64),args=command,saveat=SaveAt(ts=times),
                       stepsize_controller=PIDController(rtol=1e-7,atol=1e-9),max_steps=32768,throw=False)
        states=sol.ys; derivative=jax.vmap(lambda y:fun(0.,y,command))(states)
        return states,derivative
    if not privileged:
        decode=jax.jit(lambda z:network.apply({'params':parameters},z.astype(jnp.float32),method=network.decode))
    started=time.monotonic(); records=[r for r in manifest['records'] if r['split'] in ('test','validation')]
    complete=[]
    for i,r in enumerate(records):
        path=folder/'predictions'/r['split']/(r['case_id']+'.npz'); path.parent.mkdir(parents=True,exist_ok=True)
        meta_path=path.with_suffix('.json')
        if path.exists() and meta_path.exists():
            meta=json.loads(meta_path.read_text())
            assert meta['prediction_sha256']==sha(path) and meta['checkpoint_sha256']==report['checkpoint_sha256']
            complete.append(meta); continue
        write(folder/'inference_status.json', {'stage':'predicting','arm':arm,'seed':seed,'completed':i,
                                             'total':len(records),'case_id':r['case_id']})
        item=data(r,state=privileged,prefix=True)
        zeros=np.zeros((3,8),np.float32)
        x=initial(jnp.asarray(item['rendering_ts']),jnp.asarray(item.get('points',zeros)),jnp.asarray(item.get('velocity',zeros)))
        states,derivative=map(np.asarray,forecast(x,jnp.asarray(item['tau'])))
        if privileged:
            position=states[:,:6]; velocity=states[:,8:14]; acceleration=derivative[:,8:14]
        else:
            position=np.c_[states[:,:8],np.ones(len(states))]@mapping
            velocity=states[:,8:]@mapping[:8]; acceleration=derivative[:,8:]@mapping[:8]
        values={'time':np.asarray(times),'points':position.reshape(-1,3,2)*.65+[.2,.35],
                'velocity':velocity.reshape(-1,3,2)*.65,'acceleration':acceleration.reshape(-1,3,2)*.65,
                'latent_state':states, 'latent_derivative':derivative}
        if not privileged:
            # Decode only predicted future states, with the exact 30 Hz frame grid.
            image_states=states[np.arange(16,705,8),:8]
            chunks=[np.asarray(decode(jnp.asarray(z))) for z in np.array_split(image_states,3)]
            rgb=np.concatenate(chunks)
            values['rgb_prediction']=np.rint((rgb+1)*127.5).clip(0,255).astype(np.uint8)
            values['rgb_time']=np.arange(3,90,dtype=float)/30
        tmp=path.with_name(path.name+'.partial')
        with tmp.open('wb') as handle:
            np.savez_compressed(handle,**values)
        tmp.replace(path)
        meta={'case_id':r['case_id'],'prediction_sha256':sha(path),'checkpoint_sha256':report['checkpoint_sha256'],
              'input_sha256':r['inputs_sha256'],'finite':bool(np.isfinite(states).all()),
              'observed_frames':3,'privileged_initial_state':privileged,'future_true_torques_used':False,
              'geometry_projection':False,'affine_readout_used':not privileged}
        write(meta_path,meta);complete.append(meta)
    write(folder/'inference_report.json',{'stage':'complete','arm':arm,'seed':seed,'completed':len(complete),
          'finite':sum(r['finite'] for r in complete),'elapsed_s':time.monotonic()-started,
          'checkpoint_sha256':report['checkpoint_sha256'],'test_cases':108,'validation_cases':54})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();run(args.arm,args.seed)
