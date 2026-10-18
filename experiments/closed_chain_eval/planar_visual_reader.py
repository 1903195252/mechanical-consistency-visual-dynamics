"""Train a shared RGB coordinate instrument, independent of candidate models.

Only the original training identities supply supervised coordinates. The final
epoch is fixed before calibration; calibration/verification videos are decoded
only after training. No kinematic closure is imposed on the three outputs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .visual_calibration import ROOT, OUT, load_protocol, readout_path, sha, write


DIRECTORY = OUT / "shared_planar_reader"
DATA = ROOT / "output/keycld_corpus"


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.SiLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.SiLU())


class CoordinateReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.down0, self.down1 = block(3, 16), block(16, 32)
        self.down2, self.down3 = block(32, 64), block(64, 96)
        self.up2, self.up1, self.up0 = block(160, 64), block(96, 32), block(48, 16)
        self.head = nn.Conv2d(16, 3, 1)
        y, x = torch.meshgrid(torch.linspace(1, -1, 64), torch.linspace(-1, 1, 64), indexing="ij")
        self.register_buffer("grid", torch.stack((x, y), -1).reshape(4096, 2))

    def forward(self, images):
        a = self.down0(images)
        b = self.down1(F.avg_pool2d(a, 2))
        c = self.down2(F.avg_pool2d(b, 2))
        d = self.down3(F.avg_pool2d(c, 2))
        up = lambda x, skip: torch.cat((F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip), 1)
        x = self.up0(up(self.up1(up(self.up2(up(d, c)), b)), a))
        logits = self.head(x).flatten(2).float()
        return logits.softmax(-1) @ self.grid, logits


def specification():
    protocol = load_protocol()
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    rows = [row for row in manifest["records"] if row["split"] == "train"]
    if {r["assembly_id"] for r in rows} & {r["assembly_id"] for r in protocol["planar_records"]}:
        raise ValueError("Instrument fitting overlaps calibration identities")
    spec = dict(schema="shared-planar-visual-instrument/1", id="shared_planar_64", seed=42,
        architecture="four-level 16/32/64/96 U-Net, GroupNorm, three 64px heatmaps, soft-argmax",
        epochs=32, batch_size=128, learning_rate=.002, final_learning_rate=.0002,
        optimizer="AdamW", weight_decay=.0001, target_sigma_pixels=.75, coordinate_loss_weight=25.,
        fit_split="train", fit_sequences=len(rows), fit_assemblies=len({r["assembly_id"] for r in rows}),
        selection="fixed final epoch, no held-out selection", inputs="64x64 RGB only",
        outputs="independent B,C,P coordinates; no manifold projection or geometry input",
        source_sha256=sha(__file__), dataset_manifest_sha256=sha(DATA / "dataset_manifest.json"),
        calibration_protocol_sha256=sha(OUT / "protocol.json"),
        training_records=[dict(case_id=r["case_id"], observations_sha256=r["observations_sha256"],
                               state_labels_sha256=r["state_labels_sha256"]) for r in rows])
    destination = DIRECTORY / "training_specification.json"
    if destination.exists() and json.loads(destination.read_text()) != spec:
        raise ValueError("Instrument specification changed; use a new version")
    write(destination, spec)
    return spec, rows


def train():
    spec, rows = specification()
    checkpoint = DIRECTORY / "final.pt"
    if checkpoint.exists():
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This registered training run requires the local CUDA device")
    torch.set_num_threads(4)
    torch.manual_seed(spec["seed"])
    np.random.seed(spec["seed"])
    torch.backends.cudnn.benchmark = True
    images, labels = [], []
    for row in rows:
        for key in ("observations", "state_labels"):
            if sha(DATA / row[key]) != row[key + "_sha256"]:
                raise ValueError("Training data changed: " + row["case_id"])
        with np.load(DATA / row["observations"]) as data:
            images.append(data["x"])
        with np.load(DATA / row["state_labels"]) as data:
            labels.append(data["x"][:len(images[-1])])
    images = torch.from_numpy(np.concatenate(images)).to("cuda")
    labels = torch.from_numpy(np.concatenate(labels)).to("cuda")
    if labels.abs().max() > 1:
        raise ValueError("Training keypoint outside the registered field of view")
    net = CoordinateReader().cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(net.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, spec["epochs"], eta_min=spec["final_learning_rate"])
    scaler = torch.cuda.amp.GradScaler()
    history, first_epoch = [], 0
    resume = DIRECTORY / "training.pt"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu")
        if saved["spec_sha256"] != sha(DIRECTORY / "training_specification.json"):
            raise ValueError("Checkpoint has a different training specification")
        net.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        history, first_epoch = saved["history"], saved["epoch"]
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
    started = time.monotonic()
    for epoch in range(first_epoch, spec["epochs"]):
        net.train()
        order = torch.randperm(len(images), device="cuda")
        total_loss = 0.
        total_error = 0.
        for indices in order.split(spec["batch_size"]):
            x = images[indices].permute(0, 3, 1, 2).float().div_(255).contiguous(memory_format=torch.channels_last)
            target = labels[indices]
            distance = ((target[:, :, None, :] - net.grid) * 31.5).square().sum(-1)
            distribution = (-distance / (2 * spec["target_sigma_pixels"] ** 2)).softmax(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                predicted, logits = net(x)
                loss = -(distribution * logits.log_softmax(-1)).sum(-1).mean()
                loss = loss + spec["coordinate_loss_weight"] * (predicted - target).square().mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(indices)
            total_error += float((predicted.detach() - target).square().sum(-1).sum())
        scheduler.step()
        row = dict(epoch=epoch + 1, loss=total_loss / len(images),
                   training_point_rmse_mm=(total_error / (len(images) * 3)) ** .5 * 650,
                   seconds=round(time.monotonic() - started, 1))
        history.append(row)
        print(json.dumps(row), flush=True)
        state = dict(model=net.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(), epoch=epoch + 1, history=history, rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(), spec_sha256=sha(DIRECTORY / "training_specification.json"))
        torch.save(state, resume.with_suffix(".partial"))
        resume.with_suffix(".partial").replace(resume)
        write(DIRECTORY / "training_status.json", dict(row, target_epochs=spec["epochs"]))
    torch.save(dict(model={k: v.cpu() for k, v in net.state_dict().items()},
                    spec_sha256=sha(DIRECTORY / "training_specification.json")), checkpoint)
    write(DIRECTORY / "training_report.json", dict(history=history, final_checkpoint_sha256=sha(checkpoint),
        parameters=sum(p.numel() for p in net.parameters()), training_frames=len(images),
        device=torch.cuda.get_device_name(), torch_version=torch.__version__, precision="CUDA AMP float16; readout float32",
        held_out_data_used_for_fitting_or_selection=False))


def read_and_score():
    from .read_visual_calibration import save
    from .score_visual_calibration import case
    from . import score_visual_calibration
    spec, _ = specification()
    torch.set_num_threads(4)
    checkpoint = DIRECTORY / "final.pt"
    data = torch.load(checkpoint, map_location="cpu")
    if data["spec_sha256"] != sha(DIRECTORY / "training_specification.json"):
        raise ValueError("Reader specification does not match final weights")
    net = CoordinateReader().cuda().eval()
    net.load_state_dict(data["model"])
    protocol = load_protocol()
    reader = dict(id=spec["id"], kind="planar", model="shared supervised RGB coordinate reader",
                  native_resolution=64, encoder_resolution=64, seed=42,
                  checkpoint_sha256=sha(checkpoint), specification_sha256=data["spec_sha256"],
                  role="external_to_KeyCLD_and_CON", normalization_of_own_model_allowed=False)
    write(DIRECTORY / "reader.json", reader)
    for row in protocol["planar_records"]:
        path = readout_path(reader, row)
        if path.exists():
            with np.load(path) as saved:
                if str(saved["reader_sha256"]) != sha(checkpoint):
                    raise ValueError("Readout checkpoint changed")
            continue
        if sha(DATA / row["observations"]) != row["observations_sha256"]:
            raise ValueError("Reference video changed")
        with np.load(DATA / row["observations"]) as data:
            x, t = data["x"], data["t"]
        with torch.no_grad():
            tensor = torch.from_numpy(x).cuda().permute(0, 3, 1, 2).float() / 255
            points, _ = net(tensor)
        save(reader, row, time=t.astype(float), points=points.cpu().numpy() * .65 + [.2, .35],
             reader_sha256=np.array(sha(checkpoint)))
    tasks = [(row, reader, sha(OUT / "protocol.json"), sha(score_visual_calibration.__file__))
             for row in protocol["planar_records"]]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for index, future in enumerate(as_completed([pool.submit(case, task) for task in tasks])):
            future.result()
            if (index + 1) % 9 == 0:
                print(json.dumps(dict(scored=index + 1, total=len(tasks))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "calibrate", "all"), default="all")
    args = parser.parse_args()
    if args.stage in ("train", "all"):
        train()
    if args.stage in ("calibrate", "all"):
        read_and_score()
