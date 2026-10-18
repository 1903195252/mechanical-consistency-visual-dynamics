"""Continuous SI scoring of completed high-resolution native forecasts.

Correct-reference calibration for the external RGB reader is a separate track;
these native-state scores never borrow its error profile or change tolerances.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import numpy as np
from .common import ROOT,OUT,dataset,sha,write


def score(args):
    from experiments.con_corpus.score_readout import case
    from experiments.closed_chain_eval.score_reports import planar_report
    from .bind_evaluator import verify as verify_evaluator
    evaluator = verify_evaluator()
    data=dataset(args.width)
    folder=OUT/"runs"/f"{args.model}_{args.width}_seed{args.seed}"
    inference=json.loads((folder/"inference_report.json").read_text())
    assert inference["completed"]==162
    calibration=json.loads((ROOT/"output/keycld_corpus/calibration.json").read_text())
    readout_calibration=json.loads((ROOT/"output/con_corpus/dynamics_extension/readout_calibration.json").read_text())
    assert calibration["passed"] and readout_calibration["passed"]
    limits=json.loads((ROOT/"output/keycld_corpus/protocol.json").read_text())["evaluation"]["limits_SI"]
    permutation=[0,1,2]
    if args.model=="keycld":
        permutations=list(itertools.permutations(range(3)));errors=np.zeros(6)
        for r in data.records:
            if r["split"]!="validation":continue
            with np.load(folder/"predictions/validation"/(r["case_id"]+".npz")) as z: points=z["observed_keypoints"]
            with np.load(ROOT/"output/keycld_corpus"/r["state_labels"]) as z: truth=z["x"][:3].astype(float)*.65+[.2,.35]
            for i,p in enumerate(permutations):
                error=np.mean((points[:,p]-truth)**2)
                errors[i]+=error if np.isfinite(error) else np.inf
        if not np.isfinite(errors).any():raise ValueError("No finite validation keypoint permutation")
        permutation=list(permutations[int(np.argmin(errors))])
    pack=json.loads((ROOT/"output/con_corpus/evaluator/manifest.json").read_text())
    references={r["case_id"]:r for r in pack["references"]}
    identity=dict(model=args.model,width=args.width,seed=args.seed,checkpoint_sha256=inference["checkpoint_sha256"],
        protocol_sha256=sha(ROOT/"experiments/planar_resolution/protocol.json"),source_sha256=sha(__file__),
        permutation=permutation,scoring_version="continuous-si-1",
        evaluator_binding_sha256=sha(OUT/"evaluator_binding.json"))
    destination=folder/"measurements";destination.mkdir(parents=True,exist_ok=True)
    test=[r for r in data.records if r["split"]=="test"]
    tasks=[(r,references[r["case_id"]],folder,permutation,limits,destination,identity) for r in test]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(case,tasks))
    reports={}
    for track in ("raw","conditioned"):
        track_rows=[dict({k:r[k] for k in ("case_id","assembly_id","category","condition")},**r[track]) for r in rows]
        reports[track]=planar_report(dict(identity,rows=track_rows),limits,sha(ROOT/"output/keycld_corpus/calibration.json"))
    visual=[]
    for record in test:
        path=folder/"predictions/test"/(record["case_id"]+".npz")
        meta=json.loads(path.with_suffix(".json").read_text())
        assert sha(path)==meta["prediction_sha256"]
        images=np.load(data.root/record["images"],mmap_mode="r")
        target=images[3:].astype(np.float32)/255
        # Same 90-frame temporal-variance mask as the frozen 64px diagnosis.
        mask=np.mean(np.var(images.astype(np.float32)/255,axis=0),axis=-1)>1e-4
        persistence=(images[2].astype(np.float32)/255-target)**2
        metrics=dict(persistence_rgb_mse=float(persistence.mean()),
            persistence_motion_mask_mse=float(persistence[:,mask].mean()) if mask.any() else None,
            rgb_mse=None,motion_mask_mse=None,finite=meta["finite"])
        if meta["finite"]:
            with np.load(path) as pred: rgb=pred["rgb_prediction"].astype(np.float32)/255
            assert rgb.shape==target.shape
            error=(rgb-target)**2
            metrics.update(rgb_mse=float(error.mean()),motion_mask_mse=float(error[:,mask].mean()) if mask.any() else None)
        visual.append(dict(case_id=record["case_id"],category=record["category"],condition=record["condition"],**metrics))
    means={}
    for key in ("rgb_mse","motion_mask_mse","persistence_rgb_mse","persistence_motion_mask_mse"):
        groups=defaultdict(list)
        for row in visual:
            if row[key] is not None:groups[(row["category"],row["condition"])].append(row[key])
        means[key]=float(np.mean([np.mean(v) for v in groups.values()])) if groups else None
    write(folder/"scores.json",dict(identity,native=reports["raw"],reconstructed=reports["conditioned"],
        visual=dict(means=means,finite=sum(r["finite"] for r in visual),total=len(test),rows=visual),
        external_reader_score="separate independently calibrated measurement track"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=("keycld","con"),required=True)
    parser.add_argument("--width",type=int,choices=(256,512),required=True)
    parser.add_argument("--seed",type=int,choices=(42,43,44),required=True)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if args.workers<1:parser.error("workers must be positive")
    score(args)
