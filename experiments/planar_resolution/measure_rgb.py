"""Paired high-resolution RGB measurements using the frozen external reader.

Reference and generated videos pass through the same instrument. Cartesian
and CAD-projected derivatives are retained as separate diagnostic controls.
Native-state scores and the registered engineering tolerances are unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time
import numpy as np
from .common import ROOT, OUT, dataset, sha, write


def configure_precision():
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def read_points(net, video, batch_size):
    import torch
    pieces = []
    with torch.no_grad():
        for first in range(0,len(video),batch_size):
            batch = torch.from_numpy(np.array(video[first:first+batch_size])).cuda().permute(0,3,1,2).float()/255
            pieces.append(net(batch)[0].cpu().numpy())
    return np.concatenate(pieces)*.65+[.2,.35]


def save_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def measurement(task):
    from experiments.keycld_corpus.physics_score import measure, METRICS
    from experiments.closed_chain_eval.optimize_visual_measurement import reference
    from experiments.closed_chain_eval.measurement_derivatives import planar_derivatives
    from experiments.closed_chain_eval.optimize_visual_phase import project_track, INTERIOR
    row, folder, name, selected, identity = task
    path = folder / "readouts" / name / (row["case_id"] + ".npz")
    output = folder / "measurements" / name / (row["case_id"] + ".json")
    if output.exists():
        cached = json.loads(output.read_text())
        assert cached["identity"] == identity
        return cached
    base = {k: row[k] for k in ("case_id", "assembly_id", "category", "condition")}
    methods = list(dict.fromkeys(["raw_cartesian_spline5", "cartesian_spline5", selected]))
    failed = dict(metrics={k: None for k in METRICS}, passed={k: False for k in METRICS},
                  coverage=0., joint_pass=False)
    if not path.exists():
        result = dict(base, input=name, identity=identity, finite=False,
                      measurements={m: dict(failed, failure_reason="nonfinite model forecast") for m in methods})
        write(output, result)
        return result
    with np.load(path) as source:
        x, t = source["points"], source["time"]
        assert str(source["reader_sha256"]) == identity["reader_sha256"]
    params, truth, *_ = reference(row)
    limits = json.loads((ROOT / "output/keycld_corpus/protocol.json").read_text())["evaluation"]["limits_SI"]
    measurements, corrections = {}, {}
    for method in methods:
        try:
            if method == "raw_cartesian_spline5":
                v, a = planar_derivatives(t, x, params["config"], "cartesian_spline5")
                values = x, v, a
            else:
                values = project_track(x, t, params["config"], method)
                corrections[method] = float(np.sqrt(np.mean(np.sum((values[0][INTERIOR]-x[INTERIOR])**2, axis=-1))))
            measurements[method] = measure(t[INTERIOR], *(p[INTERIOR] for p in values), truth[INTERIOR],
                params, params["command_speed_rad_s"], limits)
        except (ValueError, np.linalg.LinAlgError) as error:
            measurements[method] = dict(failed, failure_reason=str(error))
    result = dict(base, input=name, identity=identity, finite=True, measurements=measurements,
        correction_point_rms_m=corrections, readout_sha256=sha(path),
        future_point_euclidean_rmse_m=float(np.sqrt(np.mean(np.sum((x[3:]-truth[3:])**2, axis=-1)))))
    write(output, result)
    return result


def run(args):
    import torch
    from .reader import ResolutionReader
    from .bind_evaluator import verify
    from experiments.closed_chain_eval.score_reports import planar_report
    verify()
    configure_precision()
    folder = OUT / "runs" / f"{args.model}_{args.width}_seed{args.seed}"
    inference = json.loads((folder / "inference_report.json").read_text())
    assert inference["completed"] == 162
    reader = OUT / "readers" / f"reader_{args.width}"
    checkpoint = reader / "final.pt"
    reader_hash = sha(checkpoint)
    reader_report = json.loads((reader / "training_report.json").read_text())
    assert reader_report["checkpoint_sha256"] == reader_hash
    calibration = json.loads((reader / "calibration.json").read_text())
    assert calibration["reader"]["checkpoint_sha256"] == reader_hash
    runtime_calibration = json.loads((reader / "runtime_calibration.json").read_text())
    assert runtime_calibration["complete"] and runtime_calibration["total"] == 54
    runtime = runtime_calibration["identity"]
    assert runtime["reader_sha256"] == reader_hash
    assert runtime["torch_version"] == torch.__version__ and runtime["device"] == torch.cuda.get_device_name()
    assert runtime["batch_size"] == args.batch_size and not runtime["cudnn_allow_tf32"] and not runtime["matmul_allow_tf32"]
    selection = reader / "measurement_refinement/selection.json"
    selected = json.loads(selection.read_text())["selected"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["spec_sha256"] == sha(reader / "specification.json")
    net = ResolutionReader(args.width).cuda().eval()
    net.load_state_dict(saved["model"])
    data = dataset(args.width)
    rows = [r for r in data.records if r["split"] == "test"]
    assert len(rows) == 108
    destination = folder / "external_rgb"
    identity = dict(model=args.model, seed=args.seed, width=args.width,
        checkpoint_sha256=inference["checkpoint_sha256"], reader_sha256=reader_hash,
        reader_calibration_sha256=sha(reader / "runtime_calibration.json"),
        base_reader_calibration_sha256=sha(reader / "calibration.json"),
        derivative_selection_sha256=sha(selection), selected_derivative=selected,
        evaluator_binding_sha256=sha(OUT / "evaluator_binding.json"), source_sha256=sha(__file__),
        scored_frames=list(range(5,85)), original_coordinates_retained=True,
        engineering_tolerances_changed=False, references_used_for_fit=False,
        paired_RMSE_subtraction=False, torch_version=torch.__version__, batch_size=args.batch_size,
        device=torch.cuda.get_device_name(), precision="FP32",
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    if (destination / "specification.json").exists():
        assert json.loads((destination / "specification.json").read_text()) == identity
    write(destination / "specification.json", identity)
    started = time.monotonic()
    for index, row in enumerate(rows):
        prediction = folder / "predictions/test" / (row["case_id"] + ".npz")
        meta = json.loads(prediction.with_suffix(".json").read_text())
        assert sha(prediction) == meta["prediction_sha256"]
        assert meta["checkpoint_sha256"] == inference["checkpoint_sha256"]
        assert sha(data.root / row["images"]) == row["images_sha256"]
        images = np.load(data.root / row["images"], mmap_mode="r")
        inputs = [("reference", images, row["images_sha256"])]
        if meta["finite"]:
            with np.load(prediction) as source:
                assert np.allclose(source["rgb_time"], np.arange(3,90)/30)
                generated = np.concatenate((images[:3], source["rgb_prediction"]))
            assert generated.shape == images.shape
            inputs.append(("generated", generated, meta["prediction_sha256"]))
        for name, video, source_hash in inputs:
            path = destination / "readouts" / name / (row["case_id"] + ".npz")
            if path.exists():
                with np.load(path) as cached:
                    assert str(cached["reader_sha256"]) == reader_hash and str(cached["source_sha256"]) == source_hash
                continue
            save_npz(path, points=read_points(net,video,args.batch_size), time=np.arange(90)/30,
                     reader_sha256=np.array(reader_hash), source_sha256=np.array(source_hash))
        write(destination / "status.json", dict(stage="reading_RGB", completed=index+1,total=108))
    del net
    torch.cuda.empty_cache()
    tasks = [(r, destination, name, selected, identity) for name in ("reference","generated") for r in rows]
    # Spawn after CUDA initialization; fork can inherit unsafe CUDA state.
    import multiprocessing
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        measurements = list(pool.map(measurement, tasks))
    limits = json.loads((ROOT / "output/keycld_corpus/protocol.json").read_text())["evaluation"]["limits_SI"]
    reports = {}
    for name in ("reference","generated"):
        reports[name] = {}
        for method in dict.fromkeys(["raw_cartesian_spline5","cartesian_spline5",selected]):
            selected_rows = [dict({k:r[k] for k in ("case_id","assembly_id","category","condition")},
                **r["measurements"][method]) for r in measurements if r["input"]==name]
            reports[name][method] = planar_report(dict(rows=selected_rows),limits,identity["reader_calibration_sha256"])
    write(destination / "report.json",dict(identity=identity,total=108,summary=reports,rows=measurements,
        elapsed_s=time.monotonic()-started, native_scores_changed=False,
        attribution="Paired instrument controls; residual reference error remains measured separately"))
    write(destination / "status.json",dict(stage="complete",total=108,elapsed_s=time.monotonic()-started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("keycld","con"), required=True)
    parser.add_argument("--width", type=int, choices=(256,512), required=True)
    parser.add_argument("--seed", type=int, choices=(42,43,44), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 1:
        parser.error("positive batch size and workers required")
    run(args)
