"""Improve image-quality robustness using only controlled training corruptions."""
import argparse
import copy
import json
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .planar_visual_reader import CoordinateReader, DATA, DIRECTORY as BASE
from .visual_calibration import ROOT, OUT, sha, write, load_protocol


DIRECTORY = OUT / "optimization/robust_planar_reader"
QUALITIES = dict(clean=(0.,1.), mild_blur=(.5,1.), blur=(1.,1.), blur_low_contrast=(1.,.6))


def degradation(x, sigma, contrast):
    if sigma > 0:
        size=7
        u=torch.arange(-(size//2),size//2+1,device=x.device).float()
        g=torch.exp(-u.square()/(2*sigma*sigma));g/=g.sum()
        kernel=(g[:,None]*g[None,:]).expand(3,1,size,size)
        x=F.conv2d(F.pad(x,(3,3,3,3),mode="reflect"),kernel,groups=3)
    return ((x-128/255)*contrast+128/255).clamp(0,1)


def train():
    spec=dict(source_sha256=sha(__file__),initial_checkpoint_sha256=sha(BASE/"final.pt"),seed=142,
        epochs=8,batch_size=128,learning_rate=.0002,final_learning_rate=.00001,fit_split="train",
        augmentation="half clean batches; half Gaussian sigma U(0.2,1.5)px, contrast U(0.5,1.1), noise 1/255",
        checkpoint_selection="fixed final epoch; no model forecast used for fitting or checkpoint selection",
        role="development robustness refinement; generated-image domain must be tested separately")
    path=DIRECTORY/"specification.json"
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError("Robustness specification changed")
    write(path,spec)
    if (DIRECTORY/"final.pt").exists():return
    torch.manual_seed(142);rng=np.random.default_rng(142)
    state=torch.load(BASE/"final.pt",map_location="cpu")
    net=CoordinateReader().cuda().to(memory_format=torch.channels_last);net.load_state_dict(state["model"])
    rows=[r for r in json.loads((DATA/"dataset_manifest.json").read_text())["records"] if r["split"]=="train"]
    images,ys=[],[]
    for row in rows:
        with np.load(DATA/row["observations"]) as data:images.append(data["x"])
        with np.load(DATA/row["state_labels"]) as data:ys.append(data["x"][:len(images[-1])])
    images=torch.from_numpy(np.concatenate(images)).cuda();ys=torch.from_numpy(np.concatenate(ys)).cuda()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,8,eta_min=.00001)
    scaler=torch.cuda.amp.GradScaler()
    history=[];started=time.monotonic()
    for epoch in range(8):
        total=0.
        for batch in torch.randperm(len(images),device="cuda").split(128):
            x=images[batch].permute(0,3,1,2).float()/255
            if rng.random()<.5:
                x=degradation(x,float(rng.uniform(.2,1.5)),float(rng.uniform(.5,1.1)))
                x=(x+torch.randn_like(x)/255).clamp(0,1)
            target=ys[batch]
            distance=((target[:,:,None,:]-net.grid)*31.5).square().sum(-1)
            distribution=(-distance/(2*.75**2)).softmax(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                point,logit=net(x.contiguous(memory_format=torch.channels_last))
                loss=-(distribution*logit.log_softmax(-1)).sum(-1).mean()+25*(point-target).square().mean()
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
            scaler.step(optimizer);scaler.update();total+=float(loss.detach())*len(batch)
        scheduler.step()
        r=dict(epoch=epoch+1,loss=total/len(images),seconds=round(time.monotonic()-started,1));history.append(r)
        print(json.dumps(r),flush=True)
    torch.save(dict(model={k:v.cpu() for k,v in net.state_dict().items()},spec_sha256=sha(path)),DIRECTORY/"final.pt")
    write(DIRECTORY/"training_report.json",dict(history=history,checkpoint_sha256=sha(DIRECTORY/"final.pt"),
                                              training_frames=len(images),epochs=8))


def stress():
    protocol=load_protocol()
    rows=protocol["planar_records"]
    results=[]
    for name,path in [("baseline",BASE/"final.pt"),("robust",DIRECTORY/"final.pt")]:
        net=CoordinateReader().cuda().eval();net.load_state_dict(torch.load(path,map_location="cpu")["model"])
        for quality,(sigma,contrast) in QUALITIES.items():
            for row in rows:
                with np.load(DATA/row["observations"]) as d:images=d["x"]
                with np.load(DATA/row["state_labels"]) as d:truth=d["x"]*.65+[.2,.35]
                predicted=[]
                for first in range(0,len(images),45):
                    x=torch.from_numpy(images[first:first+45]).cuda().permute(0,3,1,2).float()/255
                    with torch.no_grad():q=net(degradation(x,sigma,contrast))[0]
                    predicted.append(q.cpu().numpy()*.65+[.2,.35])
                point=np.concatenate(predicted)
                r=dict(reader=name,quality=quality,case_id=row["case_id"],category=row["category"],assembly_id=row["assembly_id"],
                    calibration_role=row["calibration_role"],point_rmse_m=float(np.sqrt(np.mean(np.sum((point[3:]-truth[3:])**2,axis=-1)))))
                results.append(r)
                dest=DIRECTORY/"stress_readouts"/name/quality/(row["case_id"]+".npz")
                dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,points=point,time=np.arange(90)/30)
            group=[r for r in results if r["reader"]==name and r["quality"]==quality and r["calibration_role"]=="verification"]
            print(json.dumps(dict(reader=name,quality=quality,verification_point_rmse_mm=1000*np.mean([r["point_rmse_m"] for r in group]))),flush=True)
    write(DIRECTORY/"stress.json",dict(rows=results,qualities=QUALITIES,selected_using_these_errors=False,
        forecast_data_used_for_fit=False,scope="controlled blur/contrast, not a universal guarantee for generated images"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--stage",choices=("all","stress"),default="all")
    args=parser.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1)
    if args.stage=="all":train()
    stress()
