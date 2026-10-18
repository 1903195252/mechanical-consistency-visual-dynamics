"""Final fixed-instrument check on reference videos of model-test identities.

These data do not fit heads, choose hyperparameters or enter noise quantiles.
The only per-assembly calibration uses one observed image at known CAD reset.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import cv2
import numpy as np
import torch
from .visual_calibration import ROOT,OUT,write,sha
from .refine_spatial_reader import cache_features
from .geometry_spatial_reader import evaluate,nominal
from .calibrate_spatial_reset import align
from .spatial_instrument_score import measure_reader


DIRECTORY=OUT/"spatial_confirmation"


def score(task):return measure_reader(task,source=__file__)


def run():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    manifest=json.loads((ROOT/"output/model_expansion/spatial/dataset_manifest.json").read_text())
    rows=[dict(r,calibration_role="confirmation_only") for r in manifest["videos"] if r["split"]=="test" and r["view"]=="front"]
    write(DIRECTORY/"specification.json",dict(source_sha256=sha(__file__),cases=[r["video_id"] for r in rows],
        calibration_quantile_fit=False,training_or_selection=False,
        frozen_heads={f:sha(OUT/"optimization/geometry_spatial_reader"/(f+"_head.pt")) for f in ("H-A","H-B")}))
    cache_features(rows)
    evaluate(rows)
    offsets={}
    for row in rows:
        if row["condition"]!="low_speed":continue
        path=OUT/"readouts/geometry_spatial_384"/(row["video_id"]+".npz")
        with np.load(path) as data:observed=data["qpos"][0].astype(float)
        known=nominal(row).astype(float)
        with np.load(ROOT/row["evaluator_directory"]/"physics.npz") as data:
            if not np.allclose(data["qpos"][0],known,atol=1e-7,rtol=0):raise ValueError("Known reset condition fails")
        offsets[row["assembly_id"]]=dict(observed_reset=observed.tolist(),known_reset=known.tolist(),calibration_video=row["video_id"])
    write(DIRECTORY/"calibration_offsets.json",dict(offsets=offsets,uses_future_reference=False))
    tasks=[]
    for width in (384,256):
        reader=dict(id="reset_spatial_"+str(width),kind="spatial",native_resolution=width,encoder_resolution=384)
        for row in rows:
            source=OUT/"readouts"/("geometry_spatial_"+str(width))/(row["video_id"]+".npz")
            with np.load(source) as data:q,t=data["qpos"],data["time"]
            offset=offsets[row["assembly_id"]]
            q=align(q,np.array(offset["observed_reset"]),np.array(offset["known_reset"]))
            target=OUT/"readouts"/reader["id"]/source.name;target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,qpos=q,time=t,calibration_sha256=np.array(sha(DIRECTORY/"calibration_offsets.json")))
            tasks.append((row,reader))
    with ProcessPoolExecutor(max_workers=6) as pool:measured=list(pool.map(score,tasks))
    summary=[]
    for width in (384,256):
        for family in ("H-A","H-B"):
            group=[r for r in measured if r["reader_id"]=="reset_spatial_"+str(width) and r["category"]==family]
            summary.append(dict(width=width,family=family,total=len(group),
                point_rmse_mean_mm=1000*float(np.mean([r["localization"]["point_euclidean_rmse"] for r in group])),
                native_qualified=sum(r["native_qualified"] for r in group),
                intervention_qualified=sum(r["intervention_qualified"] for r in group),
                native_geometry_max_mm=1000*max(r["native_metrics"]["geometry"] for r in group)))
    write(DIRECTORY/"summary.json",dict(summary=summary,measurements=measured,
        frozen_before_confirmation=True,these_rows_must_not_fit_noise_profiles=True))
    print(json.dumps(summary),flush=True)


if __name__=="__main__":run()
