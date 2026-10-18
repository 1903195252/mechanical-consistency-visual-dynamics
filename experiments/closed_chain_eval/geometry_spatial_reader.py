"""Fit spatial measurement coordinates relative to the calibrated CAD chart.

Tree-joint coordinates have assembly-specific zero configurations. The observer
gets the CAD's fixed nominal configuration, not a measured trajectory state,
and estimates unconstrained pose offsets from each RGB frame. No IK projection,
mass, friction, command, time or future state is supplied to the head.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json

import mujoco
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .visual_calibration import ROOT, OUT, load_protocol, write, sha
from .refine_spatial_reader import DIRECTORY as FEATURES_ROOT, PoseHead


DIRECTORY = OUT / "optimization/geometry_spatial_reader"


class GeometryHead(nn.Module):
    def __init__(self,nq):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(12288+13+nq,512),nn.GELU(),nn.Linear(512,256),nn.GELU(),nn.Linear(256,nq))

    def forward(self,x,c):
        return self.network(torch.cat((x,c),-1))


def nominal(row):
    model=mujoco.MjModel.from_xml_path(str(ROOT/row["evaluator_directory"]/"model.xml"))
    return model.qpos0.astype(np.float32).copy()


def train(family,rows):
    from experiments.model_expansion.spatial_vjepa import labels,camera,normalized
    path=DIRECTORY/(family+"_head.pt")
    if path.exists():return
    torch.manual_seed(42)
    xs,ys,cs=[],[],[]
    for row in rows:
        if row["split"]!="train":raise ValueError("Training references only")
        q0=nominal(row); q=labels(row)[0][:,:len(q0)]
        c=np.r_[camera(row),q0]
        for width in (256,384):
            xs.append(np.load(FEATURES_ROOT/"features"/str(width)/(row["video_id"]+".npy")))
            ys.append(q-q0);cs.append(np.broadcast_to(c,(len(q),len(c))))
    ym,ystd=normalized(np.concatenate(ys));cm,cstd=normalized(np.concatenate(cs))
    x=torch.from_numpy(np.concatenate(xs)).float().cuda()
    y=torch.from_numpy((np.concatenate(ys)-ym)/ystd).float().cuda()
    c=torch.from_numpy((np.concatenate(cs)-cm)/cstd).float().cuda()
    net=GeometryHead(len(ym)).cuda()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,240,eta_min=.000003)
    logs=[]
    for epoch in range(240):
        total=0.
        for batch in torch.randperm(len(x),device="cuda").split(256):
            optimizer.zero_grad(set_to_none=True)
            loss=F.mse_loss(net(x[batch],c[batch]),y[batch]);loss.backward();optimizer.step()
            total+=float(loss.detach())*len(batch)
        scheduler.step()
        if (epoch+1)%40==0:
            r=dict(family=family,epoch=epoch+1,training_loss=total/len(x));logs.append(r);print(json.dumps(r),flush=True)
    metadata=dict(ymean=ym,ystd=ystd,cmean=cm,cstd=cstd,nq=len(ym))
    torch.save(dict(model={k:v.cpu() for k,v in net.state_dict().items()},metadata=metadata,
                    specification_sha256=sha(DIRECTORY/"specification.json")),path)
    write(DIRECTORY/(family+"_training.json"),dict(logs=logs,checkpoint_sha256=sha(path),frames=len(x)))
    del x,y,c,net,optimizer
    gc.collect();torch.cuda.empty_cache()


def evaluate(records):
    from experiments.model_expansion.spatial_vjepa import camera
    from experiments.model_expansion.checkpoint_io import load_checkpoint
    from .read_visual_calibration import save
    heads={}
    for f in ("H-A","H-B"):
        s=load_checkpoint(DIRECTORY/(f+"_head.pt"));m=s["metadata"]
        head=GeometryHead(m["nq"]).cuda().eval();head.load_state_dict(s["model"])
        heads[f]=head,m
    readers=[]
    for width in (384,256):
        reader=dict(id="geometry_spatial_"+str(width),kind="spatial",native_resolution=width,encoder_resolution=384,
            role="external RGB pose instrument with calibrated CAD coordinate chart",
            specification_sha256=sha(DIRECTORY/"specification.json"))
        readers.append(reader)
        for row in records:
            head,m=heads[row["category"]];q0=nominal(row)
            context=(np.r_[camera(row),q0]-m["cmean"])/m["cstd"]
            x=np.load(FEATURES_ROOT/"features"/str(width)/(row["video_id"]+".npy"))
            with torch.no_grad():
                output=head(torch.from_numpy(x).float().cuda(),torch.tensor(context,device="cuda").expand(len(x),-1))
            q=output.cpu().numpy()*m["ystd"]+m["ymean"]+q0
            q[:,3:7]/=np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
            save(reader,row,qpos=q,time=np.arange(120)/30)
    tasks=[(row,r) for r in readers for row in records]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i,f in enumerate(as_completed([pool.submit(score,t) for t in tasks])):
            f.result()
            if (i+1)%12==0:print(json.dumps(dict(scored=i+1,total=len(tasks))),flush=True)
    write(DIRECTORY/"readers.json",readers)


def score(task):
    from .refine_spatial_reader import score_case
    # Use the same scorer with an explicitly matched nominal-chart prefix.
    # Its adapter supports reader-specific prefixes through the small helper.
    from .spatial_instrument_score import measure_reader
    return measure_reader(task,source=__file__)


def run():
    torch.set_num_threads(4)
    p=load_protocol();m=json.loads((ROOT/"output/model_expansion/spatial/dataset_manifest.json").read_text())
    rows=[r for r in m["videos"] if r["split"]=="train" and r["view"]=="front"]
    spec=dict(source_sha256=sha(__file__),seed=42,epochs=240,input_resolutions=[256,384],
        train_ids=[r["video_id"] for r in rows],outputs="unconstrained qpos offsets from CAD qpos0",
        observer_geometry="fixed nominal pose/coordinate chart from the mechanism model, no future state",
        physical_equations_used_in_fit=False,closure_loss=False,final_epoch_selection=True,
        status="development refinement; inspect final held-out errors before using physical claims")
    write(DIRECTORY/"specification.json",spec)
    for family in ("H-A","H-B"):train(family,[r for r in rows if r["category"]==family])
    evaluate(p["spatial_records"])


if __name__=="__main__":run()
