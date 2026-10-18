"""Remeasure existing LingBot RGB with improved, independently checked readers.

The geometric-coordinate and known-reset instruments are separate tracks.
Forecasts and forecaster weights are unchanged; evaluator labels are opened only
by the SI scoring stage after every forecast has been read.
"""
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import sys

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .visual_calibration import ROOT,OUT,sha,write
from .geometry_spatial_reader import GeometryHead,nominal,DIRECTORY as HEADS
from .calibrate_spatial_reset import align


DIRECTORY=OUT/"paired_lingbot"


def run():
    from experiments.model_expansion.profile_vjepa import UPSTREAM,WEIGHTS,clean_weights
    from experiments.model_expansion.spatial_vjepa import camera,pooled,DIM
    from experiments.model_expansion.checkpoint_io import load_checkpoint
    from experiments.model_expansion.evaluate_spatial import evaluate_case
    torch.set_num_threads(4);cv2.setNumThreads(1)
    manifest=json.loads((ROOT/"output/model_expansion/spatial/dataset_manifest.json").read_text())
    rows=[r for r in manifest["videos"] if r["split"]=="test" and r["view"]=="front"]
    offsets=json.loads((OUT/"spatial_confirmation/calibration_offsets.json").read_text())["offsets"]
    sys.path.insert(0,str(UPSTREAM))
    from src.hub.backbones import vjepa2_1_vit_base_384
    encoder,predictor=vjepa2_1_vit_base_384(pretrained=False)
    weights=torch.load(WEIGHTS,map_location="cpu",weights_only=True)
    encoder.load_state_dict(clean_weights(weights["ema_encoder"]),strict=True)
    del weights,predictor
    gc.collect();encoder.eval().requires_grad_(False).cuda()
    mean=torch.tensor([.485,.456,.406],device="cuda").view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device="cuda").view(1,3,1,1)
    heads={}
    for f in ("H-A","H-B"):
        s=load_checkpoint(HEADS/(f+"_head.pt"));head=GeometryHead(s["metadata"]["nq"]).cuda().eval()
        head.load_state_dict(s["model"]);heads[f]=head,s["metadata"]
    sources={}
    for mode in ("zero_shot","adapted"):
        for i,row in enumerate(rows):
            source=ROOT/"output/model_expansion/lingbot_spatial"/mode/"predictions"/(row["video_id"]+"_rgb.npz")
            sources[str(source.relative_to(ROOT))]=sha(source)
            targets={track:DIRECTORY/mode/track/"predictions"/row["category"]/(row["video_id"]+".npz") for track in ("geometry","reset")}
            if all(path.exists() for path in targets.values()):continue
            with np.load(source) as data:images=data["rgb"]
            if len(images)!=120:raise ValueError("Expected 120 frozen forecast frames")
            head,m=heads[row["category"]];q0=nominal(row)
            context=torch.tensor((np.r_[camera(row),q0]-m["cmean"])/m["cstd"],device="cuda")
            decoded=[]
            with torch.inference_mode():
                for first in range(3,120,8):
                    frames=np.stack([cv2.resize(im,(384,384),interpolation=cv2.INTER_CUBIC) for im in images[first:first+8]])
                    x=torch.from_numpy(frames).cuda().permute(0,3,1,2).float()/255
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        h=F.layer_norm(encoder(((x-mean)/std).unsqueeze(2).repeat(1,1,2,1,1)),(DIM,))
                    h=pooled(h.to(torch.float16).float()).to(torch.float16).float()
                    y=head(h,context.expand(len(h),-1))
                    decoded.append(y.cpu().numpy())
            q=np.concatenate(decoded)*m["ystd"]+m["ymean"]+q0
            norm=np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
            if not np.isfinite(q).all() or np.any(norm<1e-8):raise ValueError("Invalid instrument output")
            q[:,3:7]/=norm
            # Read exactly the three observed states; no reference future state
            # enters the observer or the forecaster.
            with np.load(OUT/"readouts/geometry_spatial_384"/(row["video_id"]+".npz")) as data:prefix=data["qpos"][:3].copy()
            q=np.concatenate((prefix,q)).astype(float)
            offset=offsets[row["assembly_id"]]
            options=dict(geometry=q,reset=align(q,np.array(offset["observed_reset"]),np.array(offset["known_reset"])))
            for track,path in targets.items():
                path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,qpos=options[track],time=np.arange(120)/30,observed_frames=np.array(3),
                                    forecast_sha256=np.array(sources[str(source.relative_to(ROOT))]),
                                    reader_sha256=np.array(sha(HEADS/(row["category"]+"_head.pt"))))
            if (i+1)%10==0:print(json.dumps(dict(mode=mode,read=i+1,total=30)),flush=True)
    del encoder,heads
    gc.collect();torch.cuda.empty_cache()
    write(DIRECTORY/"reading_complete.json",dict(source_sha256=sha(__file__),forecasts=sources,
        model_forecasts_changed=False,score_truth_opened_during_reading=False,
        geometry="fixed CAD coordinate chart",reset="one known-pose calibration exposure per assembly, all conditions share it"))
    summary=[]
    for mode in ("zero_shot","adapted"):
        for track in ("geometry","reset"):
            folder=DIRECTORY/mode/track
            with ProcessPoolExecutor(max_workers=6) as pool:
                results=list(pool.map(evaluate_case,[(r,"score",str(folder)) for r in rows]))
            for family in ("H-A","H-B"):
                r=[v for v in results if v["category"]==family]
                write(folder/(family+"_score.json"),r)
                summary.append(dict(mode=mode,reader=track,category=family,total=len(r),
                    point_rmse_mean_mm=float(np.mean([v["native"]["point_rmse_mm"] for v in r])),
                    native_qualified=sum(v["native"]["mechanical_coverage"] for v in r),
                    projected_qualified=sum(v["kinematic_intervention"].get("mechanical_coverage",False) for v in r)))
    write(DIRECTORY/"summary.json",dict(summary=summary,formal_model_ranking=False,
        note="read alongside matched reference-instrument controls; remaining instrument errors are not model-only errors"))
    print(json.dumps(summary),flush=True)


if __name__=="__main__":run()
