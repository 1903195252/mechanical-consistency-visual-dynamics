"""Replay held-out reader calibration with the exact inference backend."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import time
import numpy as np
from .common import ROOT, OUT, dataset, sha, write
from .measure_rgb import configure_precision, read_points, save_npz, measurement


def run(width, batch_size, workers):
    import torch
    from .reader import ResolutionReader
    from .bind_evaluator import verify
    from experiments.closed_chain_eval.visual_calibration import load_protocol
    verify()
    configure_precision()
    folder = OUT/"readers"/f"reader_{width}"
    checkpoint = folder/"final.pt"
    reader_hash = sha(checkpoint)
    assert json.loads((folder/"training_report.json").read_text())["checkpoint_sha256"] == reader_hash
    selection = folder/"measurement_refinement/selection.json"
    selected = json.loads(selection.read_text())["selected"]
    identity = dict(reader_sha256=reader_hash, width=width, torch_version=torch.__version__,
        device=torch.cuda.get_device_name(), batch_size=batch_size, precision="FP32",
        cudnn_allow_tf32=False, matmul_allow_tf32=False,
        derivative_selection_sha256=sha(selection), source_sha256=sha(__file__),
        measurement_source_sha256=sha(ROOT/"experiments/planar_resolution/measure_rgb.py"),
        evaluator_binding_sha256=sha(OUT/"evaluator_binding.json"),
        fitting_performed=False, engineering_tolerances_changed=False)
    destination=folder/"runtime_measurement"
    if (destination/"specification.json").exists():
        assert json.loads((destination/"specification.json").read_text()) == identity
    write(destination/"specification.json",identity)
    data=dataset(width)
    records={r["case_id"]:r for r in data.records}
    rows=load_protocol()["planar_records"]
    assert len(rows)==54 and all(records[r["case_id"]]["split"]=="calibration" for r in rows)
    net=ResolutionReader(width).cuda().eval()
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    assert saved["spec_sha256"]==sha(folder/"specification.json")
    net.load_state_dict(saved["model"])
    started=time.monotonic()
    for index,row in enumerate(rows):
        record=records[row["case_id"]]
        path=destination/"readouts/reference"/(row["case_id"]+".npz")
        if not path.exists():
            source=data.root/record["images"]
            assert sha(source)==record["images_sha256"]
            video=np.load(source,mmap_mode="r")
            save_npz(path,points=read_points(net,video,batch_size),time=np.arange(90)/30,
                reader_sha256=np.array(reader_hash),source_sha256=np.array(record["images_sha256"]))
        write(destination/"status.json",dict(stage="reference_readout",completed=index+1,total=54))
    del net
    torch.cuda.empty_cache()
    tasks=[(r,destination,"reference",selected,identity) for r in rows]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        measurements=list(pool.map(measurement,tasks))
    roles={r["case_id"]:r["calibration_role"] for r in rows}
    summaries=[]
    for role in ("calibration","verification"):
        group=[r for r in measurements if roles[r["case_id"]]==role]
        for method in dict.fromkeys(["raw_cartesian_spline5","cartesian_spline5",selected]):
            values=[r["measurements"][method] for r in group]
            means={}
            for metric in values[0]["metrics"]:
                finite=[m["metrics"][metric] for m in values if m["metrics"][metric] is not None]
                means[metric]=float(np.mean(finite)) if finite else None
            summaries.append(dict(role=role,method=method,total=len(group),
                all_six_pass=sum(m["joint_pass"] for m in values),mean=means))
    write(folder/"runtime_calibration.json",dict(complete=True,total=54,identity=identity,
        selected_derivative=selected,summary=summaries,rows=measurements,elapsed_s=time.monotonic()-started,
        interpretation="Independent runtime-specific measurement controls; complete means replay complete, not universal physical qualification"))
    write(destination/"status.json",dict(stage="complete",total=54))
    print(json.dumps(dict(complete=True,total=54,summary=summaries)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width",type=int,choices=(256,512),required=True)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if args.batch_size<1 or args.workers<1: parser.error("positive batch size and workers required")
    run(args.width,args.batch_size,args.workers)
