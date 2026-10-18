"""Replay the 54 independent reference sequences at the reader's actual width."""
import argparse
import json
import numpy as np
from .common import ROOT, OUT, dataset, sha, write


def run(width, batch_size):
    import torch
    from .reader import ResolutionReader
    from experiments.closed_chain_eval.visual_calibration import load_protocol
    from experiments.closed_chain_eval.score_visual_calibration import planar_case
    torch.set_num_threads(4)
    data = dataset(width)
    records = {r["case_id"]:r for r in data.records}
    folder = OUT / "readers" / f"reader_{width}"
    checkpoint = folder / "final.pt"
    saved = torch.load(checkpoint,map_location="cpu")
    assert saved["spec_sha256"]==sha(folder/"specification.json")
    reader = dict(id=f"shared_planar_{width}",kind="planar",native_resolution=width,
                  checkpoint_sha256=sha(checkpoint),model="independent RGB coordinates")
    net=ResolutionReader(width).cuda().eval()
    net.load_state_dict(saved["model"])
    rows=load_protocol()["planar_records"]
    assert len(rows)==54
    results=[]
    for row in rows:
        record=records[row["case_id"]]
        assert record["split"]=="calibration"
        path=folder/"reference_readouts"/(row["case_id"]+".npz")
        path.parent.mkdir(parents=True,exist_ok=True)
        images=np.load(data.root/record["images"],mmap_mode="r")
        with np.load(data.root/record["observations"]) as z: times=z["t"]
        pieces=[]
        with torch.no_grad():
            for first in range(0,len(images),batch_size):
                x=torch.from_numpy(np.array(images[first:first+batch_size])).cuda().permute(0,3,1,2).float()/255
                points,_=net(x)
                pieces.append(points.cpu().numpy())
        points=np.concatenate(pieces)*.65+[.2,.35]
        np.savez_compressed(path,points=points,time=times.astype(float),reader_sha256=reader["checkpoint_sha256"])
        measured=planar_case(row,reader,path)
        # Existing raw-measurement implementation uses the legacy grid only for
        # its pixel-equivalent label, never for geometry/effort/energy scores.
        measured["pixel_span_m"]=1.3/(width-1)
        measured["localization"]["point_rmse_pixels"]=measured["localization"]["point_euclidean_rmse"]/measured["pixel_span_m"]
        result=dict(case_id=row["case_id"],assembly_id=row["assembly_id"],category=row["category"],
            calibration_role=row["calibration_role"],reader=reader,**measured)
        write(folder/"measurements"/(row["case_id"]+".json"),result)
        results.append(result)
        print(json.dumps(dict(width=width,completed=len(results),total=54)),flush=True)
    write(folder/"calibration.json",dict(width=width,reader=reader,rows=results,
        uncertainty_changes_tolerances=False,measurement_qualification="inspect SI error propagation before model attribution"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width",type=int,choices=(256,512),required=True)
    parser.add_argument("--batch-size",type=int,default=2)
    args=parser.parse_args()
    if args.batch_size<1: parser.error("batch size must be positive")
    run(args.width,args.batch_size)
