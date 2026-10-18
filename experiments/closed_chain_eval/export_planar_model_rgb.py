"""Recover existing forecasts for common visual measurement; no new rollout."""
import json
import os
import pickle
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from .visual_calibration import ROOT, OUT, sha, write


def run():
    import jax
    import jax.numpy as jnp
    if not hasattr(jax,"tree_map"):
        jax.tree_map=jax.tree_util.tree_map
    sys.path.insert(0,str(ROOT/"output/keycld_corpus/runtime"))
    from experiments.keycld_corpus.model import ControlledKeyCLD
    data=ROOT/"output/keycld_corpus"
    rows=[r for r in json.loads((data/"dataset_manifest.json").read_text())["records"] if r["split"]=="test"]
    report=[]
    for model in ("keycld","con"):
        for seed in (42,43,44):
            folder=ROOT/("output/keycld_corpus" if model=="keycld" else "output/con_corpus")/"runs"/(
                ("video_euler" if model=="keycld" else "video_con_iae")+"_seed"+str(seed))
            checkpoint_sha=sha(folder/"params.p")
            if model=="keycld":
                with (folder/"params.p").open("rb") as handle: params=pickle.load(handle)
                @jax.jit
                def render(z,g):
                    return ControlledKeyCLD(g).renderer(params,z.reshape(-1,3,2))[0]
            finite_count=0
            for row in rows:
                source=folder/"predictions/test"/(row["case_id"]+".npz")
                info=json.loads(source.with_suffix(".json").read_text())
                if sha(source)!=info["prediction_sha256"]:
                    raise ValueError("Frozen forecast changed")
                target=OUT/"model_rgb"/(model+"_seed"+str(seed))/(row["case_id"]+".npz")
                identity=dict(model=model,seed=seed,case_id=row["case_id"],finite=bool(info["finite"]),
                    prediction_sha256=info["prediction_sha256"],checkpoint_sha256=checkpoint_sha,source_sha256=sha(__file__),
                    uses_future_reference_for_rendering=False)
                if not info["finite"]:
                    write(target.with_suffix(".json"),identity)
                    continue
                finite_count+=1
                if target.exists():
                    if json.loads(target.with_suffix(".json").read_text())!=identity:
                        raise ValueError("Forecast RGB provenance changed")
                    continue
                with np.load(data/row["inputs"]) as prefix:
                    observed,g=prefix["x"],prefix["geometry"]
                with np.load(source) as prediction:
                    if model=="keycld":
                        state=((prediction["points"][16::8]-[.2,.35])/.65).reshape(87,6).astype(np.float32)
                        rgb=np.concatenate([np.asarray(render(jnp.asarray(chunk),g)) for chunk in np.split(state,3)])
                        rgb=np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)
                    else:
                        rgb=prediction["rgb_prediction"]
                target.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target,rgb=np.concatenate((observed,rgb)),time=np.arange(90)/30)
                write(target.with_suffix(".json"),identity)
            report.append(dict(model=model,seed=seed,finite=finite_count,total=len(rows)))
            print(json.dumps(report[-1]),flush=True)
            if model=="keycld":
                del params
                jax.clear_caches()
    write(OUT/"model_rgb/complete.json",dict(rows=report,source_sha256=sha(__file__),new_forecasting_performed=False))


if __name__=="__main__":
    run()
