"""Select a visual derivative on validation identities, then replay calibration.

The projected track uses public CAD and measured phase. Native coordinates and
their geometric errors are retained separately. No model forecasts are opened.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
from .common import ROOT, OUT, dataset, sha, write
from experiments.closed_chain_eval.optimize_visual_measurement import reference
from experiments.closed_chain_eval.optimize_visual_phase import project_track, CANDIDATES, INTERIOR
from experiments.closed_chain_eval.measurement_derivatives import planar_derivatives
from experiments.closed_chain_eval.visual_calibration import load_protocol


def read_development(width, batch_size):
    import torch
    from .reader import ResolutionReader
    torch.set_num_threads(4)
    data=dataset(width)
    folder=OUT/"readers"/f"reader_{width}"
    checkpoint=folder/"final.pt"
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    assert saved["spec_sha256"]==sha(folder/"specification.json")
    checkpoint_hash=sha(checkpoint)
    net=ResolutionReader(width).cuda().eval()
    net.load_state_dict(saved["model"])
    rows=[r for r in data.records if r["split"]=="validation"]
    held={r["assembly_id"] for r in load_protocol()["planar_records"]}
    assert len(rows)==54 and not held & {r["assembly_id"] for r in rows}
    spec=dict(width=width,reader_sha256=checkpoint_hash,candidates=CANDIDATES,
        common_frames=list(range(5,85)),selection_split="validation",selection_ids=[r["case_id"] for r in rows],
        objective="mean P/V/A MSE divided by squared engineering scales 0.01m/0.01m/s/0.1m/s2",
        fidelity_gate="all six criteria pass on every development exact-coordinate reference",
        forecasts_used=False,engineering_thresholds_changed=False,quality="clean reference RGB",
        source_sha256=sha(__file__),phase_source_sha256=sha(ROOT/"experiments/closed_chain_eval/optimize_visual_phase.py"))
    destination=folder/"measurement_refinement"
    if (destination/"specification.json").exists():
        assert json.loads((destination/"specification.json").read_text())==spec
    write(destination/"specification.json",spec)
    for index,row in enumerate(rows):
        path=destination/"development_readouts"/(row["case_id"]+".npz")
        if path.exists():
            with np.load(path) as z:assert str(z["reader_sha256"])==checkpoint_hash
            continue
        images=np.load(data.root/row["images"],mmap_mode="r")
        with np.load(data.root/row["observations"]) as z:t=z["t"].astype(float)
        pieces=[]
        with torch.no_grad():
            for start in range(0,len(images),batch_size):
                x=torch.from_numpy(np.array(images[start:start+batch_size])).cuda().permute(0,3,1,2).float()/255
                points,_=net(x)
                pieces.append(points.cpu().numpy())
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,points=np.concatenate(pieces)*.65+[.2,.35],time=t,
                            reader_sha256=np.array(checkpoint_hash))
        print(json.dumps(dict(stage="development_readout",width=width,completed=index+1,total=len(rows))),flush=True)
    write(destination/"development_report.json",dict(width=width,records=len(rows),reader_sha256=checkpoint_hash,
          specification_sha256=sha(destination/"specification.json")))


def evaluate(task):
    from experiments.keycld_corpus.physics_score import measure,residuals
    width,row,methods,development=task
    folder=OUT/"readers"/f"reader_{width}"
    source=(folder/"measurement_refinement/development_readouts" if development else folder/"reference_readouts")/(row["case_id"]+".npz")
    with np.load(source) as z:points,t=z["points"],z["time"]
    params,truth,v,a,*_=reference(row)
    config=params["config"]
    limits=json.loads((ROOT/"output/keycld_corpus/protocol.json").read_text())["evaluation"]["limits_SI"]
    results={}
    for method in methods:
        values=project_track(points,t,config,method)
        exact=project_track(truth,t,config,method)
        numerical=measure(t[INTERIOR],*(x[INTERIOR] for x in exact),truth[INTERIOR],params,params["command_speed_rad_s"],limits)
        errors=[float(np.mean((x[INTERIOR]-y[INTERIOR])**2)) for x,y in zip(values,(truth,v,a))]
        result=dict(objective=sum(e/s**2 for e,s in zip(errors,(.01,.01,.1))),
            pva_mse=errors,numerical_reference=numerical,
            correction_point_rms_m=float(np.sqrt(np.mean(np.sum((values[0][INTERIOR]-points[INTERIOR])**2,axis=-1)))))
        if not development:
            result["measurement"]=measure(t[INTERIOR],*(x[INTERIOR] for x in values),truth[INTERIOR],params,params["command_speed_rad_s"],limits)
        results[method]=result
    g,_,_=residuals(points,np.zeros_like(points),np.zeros_like(points),config)
    return dict(case_id=row["case_id"],assembly_id=row["assembly_id"],category=row["category"],
        calibration_role=row.get("calibration_role"),readout_sha256=sha(source),
        raw_geometry_max_m=float(np.max(np.abs(g[INTERIOR]))),candidates=results)


def optimize(width,workers):
    folder=OUT/"readers"/f"reader_{width}"/"measurement_refinement"
    spec=json.loads((folder/"specification.json").read_text())
    assert spec["source_sha256"]==sha(__file__) and spec["candidates"]==CANDIDATES
    development=[r for r in dataset(width).records if r["split"]=="validation"]
    assert spec["selection_ids"]==[r["case_id"] for r in development]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(evaluate,[(width,r,CANDIDATES,True) for r in development]))
    eligible=[m for m in CANDIDATES if all(r["candidates"][m]["numerical_reference"]["joint_pass"] for r in rows)]
    if not eligible:raise ValueError("No derivative passes the exact-coordinate fidelity gate")
    objective={m:float(np.mean([r["candidates"][m]["objective"] for r in rows])) for m in eligible}
    selected=min(objective,key=objective.get)
    write(folder/"selection.json",dict(selected=selected,eligible=eligible,objective=objective,rows=rows,
        specification_sha256=sha(folder/"specification.json")))
    methods=list(dict.fromkeys(["cartesian_spline5","phase_sg7d3",selected]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        verification=list(pool.map(evaluate,[(width,r,methods,False) for r in load_protocol()["planar_records"]]))
    summaries=[]
    for role in ["calibration","verification"]:
        group=[r for r in verification if r["calibration_role"]==role]
        for method in methods:
            measurements=[r["candidates"][method]["measurement"] for r in group]
            keys=measurements[0]["metrics"]
            summaries.append(dict(role=role,method=method,total=len(group),
                all_six_pass=sum(x["joint_pass"] for x in measurements),
                exact_coordinate_all_six_pass=sum(r["candidates"][method]["numerical_reference"]["joint_pass"] for r in group),
                mean={k:float(np.mean([x["metrics"][k] for x in measurements])) for k in keys},
                maximum={k:float(max(x["metrics"][k] for x in measurements)) for k in keys}))
    write(folder/"verification.json",dict(width=width,selected=selected,summary=summaries,rows=verification,
        selection_sha256=sha(folder/"selection.json"),engineering_thresholds_changed=False,
        qualification_is_post_projection=True,raw_geometry_preserved=True))
    print(json.dumps(dict(selected=selected,summary=summaries)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width",type=int,choices=(256,512),required=True)
    parser.add_argument("--stage",choices=("read-development","optimize"),required=True)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    if args.stage=="read-development":read_development(args.width,args.batch_size)
    else:optimize(args.width,args.workers)
