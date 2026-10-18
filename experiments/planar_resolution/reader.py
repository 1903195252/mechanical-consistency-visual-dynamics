"""Resolution-specific external reader: streaming frames, full effective batch.

The architecture and training loss match the original independent coordinate
reader. Only the grid resolution changes. Gradient accumulation preserves the
128-frame effective batch without keeping the training corpus in GPU memory.
"""
import argparse
from functools import lru_cache
import json
import math
import time
import numpy as np
from .common import ROOT, OUT, dataset, training_records, sha, write
import torch
from experiments.closed_chain_eval.planar_visual_reader import CoordinateReader


class ResolutionReader(CoordinateReader):
    def __init__(self, width):
        super().__init__()
        y, x = torch.meshgrid(torch.linspace(1,-1,width), torch.linspace(-1,1,width), indexing="ij")
        self.grid = torch.stack((x,y), -1).reshape(width*width,2)


class Frames:
    def __init__(self, width):
        self.data = dataset(width)
        self.rows = training_records(self.data)
        self.width = width
        original = ROOT / "output/keycld_corpus"
        labels = []
        for row in self.rows:
            path = original / row["state_labels"]
            assert sha(path) == row["state_labels_sha256"]
            with np.load(path) as z:
                labels.append(z["x"][:50])
        self.labels = np.concatenate(labels).astype(np.float32)

    @lru_cache(maxsize=16)
    def images(self, index):
        return np.load(self.data.root / self.rows[index]["images"], mmap_mode="r", allow_pickle=False)

    def batch(self, indices):
        images = np.stack([self.images(int(i)//50)[int(i)%50] for i in indices])
        tensor = torch.from_numpy(images).cuda().permute(0,3,1,2).float().div_(255).contiguous(memory_format=torch.channels_last)
        return tensor, torch.from_numpy(self.labels[indices]).cuda()


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Reader preflight requires CUDA")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    rng = np.random.RandomState(42)
    frames = Frames(args.width)
    count = len(frames.labels)
    assert count == 40650
    folder = OUT / ("preflight" if args.mode == "benchmark" else "readers") / f"reader_{args.width}"
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / ("benchmark.json" if args.mode == "benchmark" else "training_report.json")
    if report_path.exists():
        raise FileExistsError(report_path)
    spec = dict(width=args.width, seed=42, epochs=32, effective_batch=128, microbatch=args.microbatch,
        frames=count, architecture="original 16/32/64/96 U-Net, GroupNorm, full-resolution soft-argmax",
        target_sigma_pixels=.75, coordinate_loss_weight=25., lr=.002, final_lr=.0002,
        fit_split="train", selection="fixed final epoch", precision="AMP float16; heatmap/readout float32",
        protocol_sha256=sha(ROOT / "experiments/planar_resolution/protocol.json"), source_sha256=sha(__file__))
    if (folder / "specification.json").exists():
        assert json.loads((folder / "specification.json").read_text()) == spec
    write(folder / "specification.json", spec)
    net = ResolutionReader(args.width).cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 32, eta_min=.0002)
    scaler = torch.cuda.amp.GradScaler()
    samples, history, first_epoch = [], [], 0
    resume = folder / "resume.pt"
    if args.mode == "train" and resume.exists():
        saved = torch.load(resume, map_location="cpu")
        assert saved["spec_sha256"] == sha(folder / "specification.json")
        net.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        rng.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        first_epoch, history = saved["epoch"], saved["history"]
    started = time.monotonic()
    for epoch in range(first_epoch, 32):
        order = rng.permutation(count)
        for offset in range(0,count,128):
            if args.mode == "benchmark" and len(samples) == args.steps + 2:
                break
            tick = time.monotonic()
            indices = order[offset:offset+128]
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            scale_before = scaler.get_scale()
            for first in range(0,len(indices),args.microbatch):
                part = indices[first:first+args.microbatch]
                images, target = frames.batch(part)
                distance = ((target[:,:,None,:]-net.grid)*((args.width-1)/2)).square().sum(-1)
                distribution = (-distance/(2*.75**2)).softmax(-1)
                with torch.cuda.amp.autocast():
                    prediction, logits = net(images)
                    loss = -(distribution*logits.log_softmax(-1)).sum(-1).mean()
                    loss = loss + 25.*(prediction-target).square().mean()
                    loss = loss*len(part)/len(indices)
                scaler.scale(loss).backward()
                total += float(loss.detach())
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            # AMP may skip its first update while choosing a safe dynamic scale.
            row = dict(epoch=epoch+1, batch=offset//128+1, seconds=time.monotonic()-tick,
                       loss=total, frames=len(indices), optimizer_update=bool(scaler.get_scale() >= scale_before),
                       gradient_finite=bool(torch.isfinite(grad_norm)))
            if not np.isfinite(total):
                raise FloatingPointError("Reader loss is not finite")
            if args.mode == "benchmark":
                samples.append(row)
                write(folder / "timing_partial.json", samples)
                print(json.dumps(row), flush=True)
        if args.mode == "benchmark":
            break
        schedule.step()
        history.append(dict(epoch=epoch+1, last_loss=total, elapsed_s=time.monotonic()-started))
        saved = dict(model=net.state_dict(), optimizer=optimizer.state_dict(), scheduler=schedule.state_dict(),
                     scaler=scaler.state_dict(), numpy_rng=rng.get_state(), rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(), epoch=epoch+1, history=history,
                     spec_sha256=sha(folder / "specification.json"))
        torch.save(saved, resume.with_suffix(".partial"))
        resume.with_suffix(".partial").replace(resume)
        write(folder / "status.json", history[-1])
    report = dict(mode=args.mode, scientific_training_result=args.mode=="train", width=args.width,
        device=torch.cuda.get_device_name(), torch_version=torch.__version__, samples=samples,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30, elapsed_s=time.monotonic()-started)
    if args.mode == "benchmark":
        steady = samples[2:]
        if not all(s["gradient_finite"] and s["optimizer_update"] for s in steady):
            raise FloatingPointError("Benchmark steady updates skipped by AMP; recheck precision before timing")
        times = [s["seconds"] for s in steady]
        report.update(steady_mean_s=float(np.mean(times)), p10_p90_s=np.quantile(times,[.1,.9]).tolist(),
            full_32_epoch_hours=float(np.mean(times)*math.ceil(count/128)*32/3600),
            estimate_excludes=["reader development and independent calibration", "checkpoint I/O", "thermal/power variation"])
    else:
        torch.save(dict(model=net.cpu().state_dict(), spec_sha256=sha(folder/"specification.json")),folder/"final.pt")
        report.update(history=history, checkpoint_sha256=sha(folder/"final.pt"))
    write(report_path, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width",type=int,choices=(256,512),required=True)
    parser.add_argument("--mode",choices=("benchmark","train"),required=True)
    parser.add_argument("--microbatch",type=int,default=2)
    parser.add_argument("--steps",type=int,default=3)
    args=parser.parse_args()
    if args.microbatch < 1 or args.steps < 3:
        parser.error("microbatch >= 1 and steps >= 3 required")
    run(args)
