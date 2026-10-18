"""Train a local appearance-based subpixel correction on training images.

Training crops are randomly displaced from known training landmarks. The task
is to recover the displacement from RGB appearance, not to satisfy a physical
equation. Development identities choose correction strength; evaluation is held
out. This supplements the frozen whole-image reader without altering it.
"""
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .planar_visual_reader import DATA
from .visual_calibration import ROOT, OUT, sha, write, load_protocol


DIRECTORY = OUT / "optimization/subpixel_reader"
BASE = OUT / "optimization/planar_reader"


class PatchReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(3*11*11+8, 256), nn.SiLU(),
                                     nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 2))

    def forward(self, patches, point_type, family, centers):
        context = torch.cat((patches.flatten(1)-.5, F.one_hot(point_type,3).float(),
                             F.one_hot(family,3).float(), centers/31.5-1), -1)
        return self.network(context)


def crop(images, centers):
    y,x = torch.meshgrid(torch.arange(-5,6,device=images.device), torch.arange(-5,6,device=images.device),indexing="ij")
    offset = torch.stack((x,y), -1).float()
    grid = (centers[:,None,None,:] + offset)/31.5-1
    return F.grid_sample(images,grid,mode="bilinear",padding_mode="border",align_corners=True)


def training():
    spec = dict(seed=42,steps=16000,batch_size=512,patch_width=11,jitter_std_pixels=.35,jitter_clip_pixels=1.25,
        loss="pixel-offset MSE",outputs="independent 2D subpixel correction for each landmark",
        fit_split="train",selection_split="original validation",source_sha256=sha(__file__),
        no_geometry_or_dynamics_constraints=True,learning_rate=.001,final_learning_rate=.00001)
    path = DIRECTORY / "specification.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Subpixel instrument specification changed")
    write(path,spec)
    if (DIRECTORY/"final.pt").exists():
        return
    torch.manual_seed(42)
    rows = [r for r in json.loads((DATA/"dataset_manifest.json").read_text())["records"] if r["split"]=="train"]
    images, labels, families = [],[],[]
    for row in rows:
        with np.load(DATA/row["observations"]) as data: images.append(data["x"])
        with np.load(DATA/row["state_labels"]) as data:
            labels.append((data["x"][:len(images[-1])]*[1,-1]+1)*31.5)
        families.append(np.full(len(images[-1]), ["S-A","S-B","S-J"].index(row["category"])))
    images=torch.from_numpy(np.concatenate(images)).cuda()
    labels=torch.tensor(np.concatenate(labels),dtype=torch.float32,device="cuda")
    families=torch.from_numpy(np.concatenate(families)).cuda()
    net=PatchReader().cuda()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,spec["steps"],eta_min=.00001)
    started=time.monotonic(); history=[]
    for step in range(spec["steps"]):
        indices=torch.randint(len(images),(spec["batch_size"],),device="cuda")
        point=torch.randint(3,(spec["batch_size"],),device="cuda")
        jitter=(torch.randn((spec["batch_size"],2),device="cuda")*.35).clamp(-1.25,1.25)
        centers=labels[indices,point]+jitter
        im=images[indices].permute(0,3,1,2).float()/255
        patches=crop(im,centers)
        predicted=net(patches,point,families[indices],centers)
        loss=(predicted+jitter).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();scheduler.step()
        if (step+1)%1000==0:
            row=dict(step=step+1,training_offset_rmse_pixels=float(loss.detach())**.5,
                     seconds=round(time.monotonic()-started,1))
            history.append(row);print(json.dumps(row),flush=True)
    torch.save(dict(model={k:v.cpu() for k,v in net.state_dict().items()},spec_sha256=sha(path)),DIRECTORY/"final.pt")
    write(DIRECTORY/"training_report.json",dict(history=history,checkpoint_sha256=sha(DIRECTORY/"final.pt"),
        augmented_patch_exposures=spec["steps"]*spec["batch_size"],physical_frames=len(images),test_or_calibration_fit=False))


def correct(net,images,points,family):
    centers=(points-[.2,.35])/[.65,-.65]*31.5+31.5
    result=[]
    for first in range(0,len(images),30):
        im=torch.from_numpy(images[first:first+30]).cuda().permute(0,3,1,2).float()/255
        im=im.repeat_interleave(3,dim=0)
        center=torch.tensor(centers[first:first+30].reshape(-1,2),dtype=torch.float32,device="cuda")
        pt=torch.arange(3,device="cuda").repeat(len(im)//3)
        fam=torch.full_like(pt,["S-A","S-B","S-J"].index(family))
        with torch.no_grad(): correction=net(crop(im,center),pt,fam,center).cpu().numpy().reshape(-1,3,2)
        result.append(correction*[.65/31.5,-.65/31.5])
    return np.concatenate(result)


def evaluate():
    state=torch.load(DIRECTORY/"final.pt",map_location="cpu")
    net=PatchReader().cuda().eval();net.load_state_dict(state["model"])
    rows=[r for r in json.loads((DATA/"dataset_manifest.json").read_text())["records"] if r["split"]=="validation"]
    shift=json.loads((BASE/"selection.json").read_text())["selected"]
    results=[]
    for row in rows:
        with np.load(BASE/"development"/(row["case_id"]+".npz")) as data: points=data[shift]
        with np.load(DATA/row["observations"]) as data: images=data["x"]
        with np.load(DATA/row["state_labels"]) as data: truth=data["x"]*.65+[.2,.35]
        offset=correct(net,images,points,row["category"])
        errors={str(weight):float(np.sqrt(np.mean(np.sum((points[3:]+weight*offset[3:]-truth[3:])**2,axis=-1))))
                for weight in [0.,.25,.5,.75,1.]}
        results.append(dict(case_id=row["case_id"],point_rmse_m=errors))
        path=DIRECTORY/"development"/(row["case_id"]+".npz");path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,points=points,offset=offset)
    objective={key:float(np.mean([r["point_rmse_m"][key] for r in results])) for key in results[0]["point_rmse_m"]}
    selected=min(objective,key=objective.get)
    write(DIRECTORY/"selection.json",dict(selected_weight=float(selected),development_rmse_m=objective,rows=results,
          checkpoint_sha256=sha(DIRECTORY/"final.pt")))
    print(json.dumps(dict(selected_weight=selected,development_rmse_m=objective)),flush=True)
    for row in load_protocol()["planar_records"]:
        with np.load(BASE/"held_out"/(row["case_id"]+".npz")) as data: points,t=data["points"],data["time"]
        with np.load(DATA/row["observations"]) as data: images=data["x"]
        offset=correct(net,images,points,row["category"])
        path=DIRECTORY/"held_out"/(row["case_id"]+".npz");path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,points=points+float(selected)*offset,time=t,offset=offset,
                            selection_sha256=np.array(sha(DIRECTORY/"selection.json")))


if __name__=="__main__":
    torch.set_num_threads(4)
    training()
    evaluate()
