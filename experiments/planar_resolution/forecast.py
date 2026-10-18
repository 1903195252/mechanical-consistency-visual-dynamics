"""Native 240 Hz forecasts from three real high-resolution frames only."""
import argparse
from functools import partial
import json
import pickle
import time
import numpy as np
from .common import ROOT, OUT, dataset, training_records, sha, write


def run(args):
    import jax
    import jax.numpy as jnp
    data=dataset(args.width)
    folder=OUT/"runs"/f"{args.model}_{args.width}_seed{args.seed}"
    report=json.loads((folder/"training_report.json").read_text())
    assert report["epochs"]==(40 if args.model=="keycld" else 50)
    assert sha(folder/"params.p")==report["checkpoint_sha256"]
    manifest_hash=report["identity"].get("dataset_manifest_sha256",report["identity"].get("dataset_sha256"))
    assert manifest_hash==sha(data.root/"dataset_manifest.json")
    with (folder/"params.p").open("rb") as handle: params=pickle.load(handle)
    # Keep the large native CON weights on device once and pass them as JIT
    # arguments. Capturing NumPy weights would embed GB-sized HLO constants.
    if args.model=="con":
        jax.config.update("jax_enable_x64",True)
        params=jax.device_put(params)
    times=np.arange(8,713,dtype=float)/240
    if args.model=="keycld":
        from .keycld import ResolutionKeyCLD
        from keycld import util
        from jax.experimental.ode import odeint
        @jax.jit
        def initial(images,g,command):
            model=ResolutionKeyCLD(g,args.width)
            kp,_=util.map_to_keypoints(model.encoder(params,images))
            x,v=util.finite_difference(kp.reshape(3,6),jnp.float32(1/30))
            v=util.project_velocity(model.constraint_fn,x[0],v[0])
            return jnp.r_[x[0],v],kp
        @jax.jit
        def solve(state,g,command):
            model=ResolutionKeyCLD(g,args.width)
            y=odeint(partial(model.ode,params),state,jnp.asarray(times,dtype=jnp.float32),command,mxstep=4096)
            a=jax.vmap(lambda s:model.euler_lagrange(params,s[:6],s[6:],command))(y)
            return y,a
        @jax.jit
        def render(z,g):
            return ResolutionKeyCLD(g,args.width).renderer(params,z.reshape(-1,3,2))[0]
        mapping=None
    else:
        from flax import linen as nn
        from diffrax import Dopri5,ODETerm,PIDController,SaveAt,diffeqsolve
        from src.models.autoencoders.vae import VAE
        from src.models.neural_odes.con_iae import ConIaeOde
        from src.models.dynamics_autoencoder import DynamicsAutoencoder
        network=DynamicsAutoencoder(autoencoder=VAE(img_shape=(args.width,args.width,3),latent_dim=8,norm_layer=nn.LayerNorm),
            dynamics=ConIaeOde(latent_dim=8,input_dim=14,num_layers=5,hidden_dim=30))
        encode=jax.jit(lambda p,images:network.apply({"params":p},images,method=network.encode))
        reader_path=folder/"readout.json"
        if reader_path.exists():
            saved=json.loads(reader_path.read_text())
            assert saved["checkpoint_sha256"]==report["checkpoint_sha256"]
            mapping=np.asarray(saved["affine"])
        else:
            gram=np.zeros((9,9));rhs=np.zeros((9,6));count=0
            for record in training_records(data):
                item=data.load(record,purpose="training",model="CON")
                z=np.concatenate([np.asarray(encode(params,x)) for x in np.array_split(item["rendering_ts"],25)])
                with np.load(ROOT/"output/keycld_corpus"/record["state_labels"]) as labels: target=labels["x"][:50].reshape(50,6)
                design=np.c_[z,np.ones(len(z))]
                gram+=design.T@design;rhs+=design.T@target;count+=len(z)
            mapping=np.linalg.lstsq(gram,rhs,rcond=1e-10)[0]
            write(reader_path,dict(affine=mapping.tolist(),training_points=count,fit_split="train",
                checkpoint_sha256=report["checkpoint_sha256"],predictor_uses_reader=False))
        @jax.jit
        def initial(p,images,g,command):
            encode_frames=lambda x:encode(p,x)
            z,v=jax.jvp(encode_frames,(images[1:2],),((images[2:3]-images[:1])/(2/30),))
            return jnp.r_[z[0],v[0]],encode(p,images)
        @jax.jit
        def solve(p,state,g,command):
            fun=lambda t,y,u:network.apply({"params":p},y,u,method=network.forward_dynamics)
            sol=diffeqsolve(ODETerm(fun),Dopri5(),t0=times[0],t1=times[-1],dt0=.001,
                y0=state.astype(jnp.float64),args=command,saveat=SaveAt(ts=jnp.asarray(times)),
                stepsize_controller=PIDController(rtol=1e-7,atol=1e-9),max_steps=32768,throw=False)
            return sol.ys,jax.vmap(lambda y:fun(0.,y,command))(sol.ys)
        @jax.jit
        def render(p,z,g):
            return (network.apply({"params":p},z.astype(jnp.float32),method=network.decode)+1)/2
    rows=[r for r in data.records if r["split"] in ("validation","test")]
    if args.case_id:
        rows=[r for r in rows if r["case_id"]==args.case_id]
        if not rows: raise ValueError("Unknown validation/test case")
    completed=[];started=time.monotonic()
    for record in rows:
        path=folder/"predictions"/record["split"]/(record["case_id"]+".npz")
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.with_suffix(".json").exists():
            meta=json.loads(path.with_suffix(".json").read_text())
            assert meta["checkpoint_sha256"]==report["checkpoint_sha256"] and sha(path)==meta["prediction_sha256"]
            completed.append(meta);continue
        assert sha(data.root/record["inputs"])==record["inputs_sha256"]
        item=data.load(record,purpose="forecast",model="KeyCLD" if args.model=="keycld" else "CON")
        images=item["x"] if args.model=="keycld" else item["rendering_ts"]
        geometry=item.get("geometry",np.zeros(9,np.float32))
        command=item["action"] if args.model=="keycld" else item["tau"]
        write(folder/"inference_status.json",dict(stage="predicting",case_id=record["case_id"],completed=len(completed),total=len(rows)))
        if args.model=="keycld":
            state,kp=initial(images,geometry,command)
            y,derivative=map(np.asarray,solve(state,geometry,command))
        else:
            state,kp=initial(params,images,geometry,command)
            y,derivative=map(np.asarray,solve(params,state,geometry,command))
        if args.model=="keycld":
            x,v,a=y[:,:6],y[:,6:],derivative
            latent=y[:,:6]
            observed=np.asarray(kp)
        else:
            x=np.c_[y[:,:8],np.ones(len(y))]@mapping
            v=y[:,8:]@mapping[:8];a=derivative[:,8:]@mapping[:8]
            latent=y[:,:8]
            observed=(np.c_[np.asarray(kp),np.ones(3)]@mapping).reshape(3,3,2)
        values=dict(time=times,points=x.reshape(-1,3,2)*.65+[.2,.35],
            velocity=v.reshape(-1,3,2)*.65,acceleration=a.reshape(-1,3,2)*.65,
            observed_keypoints=observed*.65+[.2,.35],rgb_time=np.arange(3,90)/30)
        finite=bool(all(np.isfinite(z).all() for z in (y,derivative)))
        if finite:
            pieces=[np.asarray(render(z.astype(np.float32),geometry) if args.model=="keycld"
                               else render(params,z.astype(np.float32),geometry))
                    for z in np.array_split(latent[16::8],29)]
            rgb=np.concatenate(pieces)
            finite=bool(np.isfinite(rgb).all())
            if finite: values["rgb_prediction"]=np.rint(rgb.clip(0,1)*255).astype(np.uint8)
        temporary=path.with_suffix(".partial")
        with temporary.open("wb") as handle: np.savez_compressed(handle,**values)
        temporary.replace(path)
        meta=dict(case_id=record["case_id"],input_sha256=record["inputs_sha256"],checkpoint_sha256=report["checkpoint_sha256"],
            prediction_sha256=sha(path),finite=finite,observed_frames=3,native_derivative_hz=240,
            future_measured_actions_used=False,source_sha256=sha(__file__),resolution=args.width)
        write(path.with_suffix(".json"),meta);completed.append(meta)
    if not args.case_id:
        write(folder/"inference_report.json",dict(model=args.model,width=args.width,seed=args.seed,
            completed=len(completed),finite=sum(r["finite"] for r in completed),records=completed,
            checkpoint_sha256=report["checkpoint_sha256"],elapsed_s=time.monotonic()-started))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=("keycld","con"),required=True)
    parser.add_argument("--width",type=int,choices=(256,512),required=True)
    parser.add_argument("--seed",type=int,choices=(42,43,44),required=True)
    parser.add_argument("--case-id")
    run(parser.parse_args())
