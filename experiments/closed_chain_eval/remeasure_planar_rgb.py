"""Paired common-instrument measurement of saved planar forecasts and GT RGB.

This is a new RGB measurement track, separate from each model's original
state/latent readout. Reference images are evaluation controls, never used to
adjust the instrument or to generate model forecasts.
"""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

import numpy as np
import torch

from .planar_visual_reader import CoordinateReader, DIRECTORY as READER_DIRECTORY, DATA
from .visual_calibration import ROOT, OUT, sha, write
from .measurement_derivatives import planar_derivatives


DIRECTORY = OUT / "paired_planar_rgb"


def read_frames():
    torch.set_num_threads(4)
    net=CoordinateReader().cuda().eval()
    saved=torch.load(READER_DIRECTORY/"final.pt",map_location="cpu")
    net.load_state_dict(saved["model"])
    rows=[r for r in json.loads((DATA/"dataset_manifest.json").read_text())["records"] if r["split"]=="test"]
    settings=[("reference",None,None)]+[(model+"_seed"+str(seed),model,seed) for model in ("keycld","con") for seed in (42,43,44)]
    for name,model,seed in settings:
        count=0
        for row in rows:
            target=DIRECTORY/"readouts"/name/(row["case_id"]+".npz")
            if target.exists():count+=1;continue
            if model is None:
                path=DATA/row["observations"]
                with np.load(path) as data:images=data["x"]
            else:
                path=OUT/"model_rgb"/name/(row["case_id"]+".npz")
                meta=json.loads(path.with_suffix(".json").read_text())
                if not meta["finite"]:
                    write(target.with_suffix(".json"),dict(finite=False,reason="nonfinite_model_forecast",source=meta))
                    continue
                with np.load(path) as data:images=data["rgb"]
            predictions=[]
            with torch.no_grad():
                for first in range(0,len(images),45):
                    x=torch.from_numpy(images[first:first+45]).cuda().permute(0,3,1,2).float()/255
                    predictions.append(net(x)[0].cpu().numpy())
            points=np.concatenate(predictions)*.65+[.2,.35]
            target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,points=points,time=np.arange(90)/30,
                reader_sha256=np.array(sha(READER_DIRECTORY/"final.pt")),source_sha256=np.array(sha(path)))
            count+=1
        print(json.dumps(dict(stage="common_rgb_readout",model=name,finite=count,total=len(rows))),flush=True)
    return rows,settings


def measure_case(task):
    from experiments.keycld_corpus.physics_score import measure,residuals
    from experiments.con_corpus.crank_readout import reconstruct
    from .optimize_visual_measurement import reference
    from .score_visual_calibration import metric_map
    row,name=task
    destination=DIRECTORY/"measurements"/name/(row["case_id"]+".json")
    if destination.exists():return json.loads(destination.read_text())
    path=DIRECTORY/"readouts"/name/(row["case_id"]+".npz")
    base=dict(reader="common_planar_64",model=name,case_id=row["case_id"],category=row["category"],condition=row["condition"],
              assembly_id=row["assembly_id"],track="external_RGB_measurement",source_sha256=sha(__file__))
    if not path.exists():
        result=dict(base,finite=False,reason="nonfinite_model_forecast")
        write(destination,result);return result
    with np.load(path) as d:x,t=d["points"],d["time"]
    params,truth,_,_,_,_,_=reference(row)
    limits=json.loads((DATA/"protocol.json").read_text())["evaluation"]["limits_SI"]
    v,a=planar_derivatives(t,x,params["config"],"cartesian_spline5")
    s=slice(5,-5)
    g,vg,ag=residuals(x,v,a,params["config"])
    rms=lambda values:float(np.sqrt(np.mean(values**2,axis=0)).max())
    point=float(np.sqrt(np.mean(np.sum((x[3:]-truth[3:])**2,axis=-1))))
    metrics=dict(geometry=float(np.max(np.abs(g[3:]))),constraint_velocity=rms(vg[s]),
        constraint_acceleration=rms(ag[s]),prediction=rms((x[3:]-truth[3:]).reshape(-1,6)),dynamics=None,energy=None)
    native_qualified=False
    if metrics["geometry"]<=limits["geometry_m"]:
        m=measure(t[s],x[s],v[s],a[s],truth[s],params,params["command_speed_rad_s"],limits)
        native_qualified=m["coverage"]==1
        metrics.update(dynamics=m["metrics"]["force_balance_Nm"],energy=m["metrics"]["work_energy_J"])
    result=dict(base,finite=True,native_metrics=metrics,native_qualified=native_qualified,
        point_euclidean_rmse_m=point,readout_sha256=sha(path),derivative="cartesian_spline5; interior frames 5:85",
        geometry_and_point_frames="all future frames 3:90")
    try:
        corrected,_=reconstruct(x[s],v[s],a[s],params["config"])
        m=measure(t[s],*corrected,truth[s],params,params["command_speed_rad_s"],limits)
        result.update(intervention_metrics=metric_map(m["metrics"]),intervention_qualified=m["coverage"]==1,
            correction_point_rms_m=float(np.sqrt(np.mean(np.sum((corrected[0]-x[s])**2,axis=-1)))))
    except ValueError as error:
        result.update(intervention_metrics={},intervention_qualified=False,intervention_failure=str(error))
    write(destination,result);return result


def run():
    rows,settings=read_frames()
    tasks=[(r,name) for name,_,_ in settings for r in rows]
    all_rows=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i,f in enumerate(as_completed([pool.submit(measure_case,t) for t in tasks])):
            all_rows.append(f.result())
            if (i+1)%108==0:print(json.dumps(dict(scored=i+1,total=len(tasks))),flush=True)
    summary=[]
    for name,_,_ in settings:
        group=[r for r in all_rows if r["model"]==name];finite=[r for r in group if r["finite"]]
        summary.append(dict(model=name,total=len(group),finite=len(finite),
            point_rmse_mean_mm=1000*float(np.mean([r["point_euclidean_rmse_m"] for r in finite])),
            geometry_mean_mm=1000*float(np.mean([r["native_metrics"]["geometry"] for r in finite])),
            native_qualified=sum(r["native_qualified"] for r in finite),
            intervention_qualified=sum(r["intervention_qualified"] for r in finite)))
    write(DIRECTORY/"summary.json",dict(summary=summary,rows=all_rows,
        instrument_sha256=sha(READER_DIRECTORY/"final.pt"),test_references_used_for_fitting=False,
        model_intrinsic_error_claim=False,note="Paired measurement-chain output; compare to reference controls, never subtract RMSE"))
    print(json.dumps(summary),flush=True)


if __name__=="__main__":run()
