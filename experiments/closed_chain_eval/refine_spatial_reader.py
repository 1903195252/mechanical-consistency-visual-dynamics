"""Refine the external spatial instrument on training images only.

Fit poses, not unobservable single-frame velocities; include both actual input
resolutions; use float32 for the small regression head. Candidate forecasters,
test images and physics thresholds are unchanged. New heads are separate files.
"""
import gc
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .visual_calibration import ROOT, OUT, sha, write, load_protocol


DIRECTORY = OUT / "optimization/spatial_reader"


class PoseHead(nn.Module):
    def __init__(self, nq):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(12288 + 13, 512), nn.GELU(),
                                     nn.Linear(512, 256), nn.GELU(), nn.Linear(256, nq))

    def forward(self, x, camera):
        return self.network(torch.cat((x, camera), -1))


def cache_features(records):
    from experiments.model_expansion.profile_vjepa import UPSTREAM, WEIGHTS, clean_weights
    from experiments.model_expansion.spatial_vjepa import pooled, DIM
    missing = [(row, width) for row in records for width in (256, 384)
               if not (DIRECTORY / "features" / str(width) / (row["video_id"] + ".npy")).exists()]
    if not missing:
        return
    sys.path.insert(0, str(UPSTREAM))
    from src.hub.backbones import vjepa2_1_vit_base_384
    encoder, predictor = vjepa2_1_vit_base_384(pretrained=False)
    weights = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    encoder.load_state_dict(clean_weights(weights["ema_encoder"]), strict=True)
    del weights, predictor
    gc.collect()
    encoder.eval().requires_grad_(False).cuda()
    mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1)
    started = time.monotonic()
    for index, (row, width) in enumerate(missing):
        cap = cv2.VideoCapture(str(ROOT / row["video"]))
        images = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if width == 256:
                    frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
                images.append(cv2.resize(frame, (384, 384), interpolation=cv2.INTER_CUBIC))
        finally:
            cap.release()
        if len(images) != 120:
            raise ValueError("Incomplete reference video")
        features = []
        with torch.no_grad():
            for first in range(0, 120, 8):
                x = torch.from_numpy(np.stack(images[first:first + 8])).cuda().permute(0, 3, 1, 2).float() / 255
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = F.layer_norm(encoder(((x-mean)/std).unsqueeze(2).repeat(1, 1, 2, 1, 1)), (DIM,))
                    h = h.to(torch.float16).float()
                features.append(pooled(h).cpu().numpy().astype(np.float16))
        path = DIRECTORY / "features" / str(width) / (row["video_id"] + ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.concatenate(features))
        if (index + 1) % 12 == 0 or index + 1 == len(missing):
            print(json.dumps(dict(stage="features", completed=index + 1, total=len(missing),
                                  seconds=round(time.monotonic() - started, 1))), flush=True)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()


def train(family, rows):
    from experiments.model_expansion.spatial_vjepa import camera, labels
    from experiments.model_expansion.checkpoint_io import load_checkpoint
    path = DIRECTORY / (family + "_pose_head.pt")
    if path.exists():
        return
    torch.manual_seed(42)
    original = load_checkpoint(ROOT / "output/model_expansion/spatial" / (family + "_readout.pt"))
    metadata = original["metadata"]
    nq = int(metadata["nq"])
    net = PoseHead(nq).cuda()
    state = original["readout"].copy()
    for name in ("network.4.weight", "network.4.bias"):
        state[name] = state[name][:nq]
    net.load_state_dict(state, strict=True)
    pool_x, pool_c, pool_y = [], [], []
    for row in rows:
        if row["split"] != "train" or row["category"] != family:
            raise ValueError("Readout fitting only accepts the registered training family")
        y = labels(row)[0][:, :nq]
        c = (camera(row) - metadata["cmean"]) / metadata["cstd"]
        for width in (256, 384):
            pool_x.append(np.load(DIRECTORY / "features" / str(width) / (row["video_id"] + ".npy")))
            pool_y.append((y - metadata["ymean"][:nq]) / metadata["ystd"][:nq])
            pool_c.append(np.broadcast_to(c, (120, len(c))))
    x = torch.from_numpy(np.concatenate(pool_x)).float().cuda()
    y = torch.from_numpy(np.concatenate(pool_y)).float().cuda()
    c = torch.from_numpy(np.concatenate(pool_c)).float().cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-5, weight_decay=1e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 160, eta_min=1e-6)
    history = []
    for epoch in range(160):
        net.train()
        total = 0.
        for batch in torch.randperm(len(x), device="cuda").split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(net(x[batch], c[batch]), y[batch])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        schedule.step()
        if (epoch + 1) % 20 == 0:
            item = dict(family=family, epoch=epoch + 1, train_pose_mse=total / len(x))
            history.append(item)
            print(json.dumps(item), flush=True)
    torch.save(dict(model={k: v.cpu() for k, v in net.state_dict().items()}, metadata=metadata,
                    specification_sha256=sha(DIRECTORY / "specification.json")), path)
    write(DIRECTORY / (family + "_training.json"), dict(history=history, checkpoint_sha256=sha(path),
        supervised_frames=len(x), physical_frames=len(x)//2, epochs=160, precision="float32 head",
        selected_using_validation_or_test=False, output="qpos only; quaternion unit normalization at inference"))
    del x, y, c, net, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def evaluate(records):
    from experiments.model_expansion.spatial_vjepa import camera
    from experiments.model_expansion.checkpoint_io import load_checkpoint
    from .read_visual_calibration import save
    from . import score_visual_calibration as scoring
    readers = []
    heads = {}
    for family in ("H-A", "H-B"):
        saved = load_checkpoint(DIRECTORY / (family + "_pose_head.pt"))
        net = PoseHead(int(saved["metadata"]["nq"])).cuda().eval()
        net.load_state_dict(saved["model"])
        heads[family] = net, saved["metadata"]
    for width in (384, 256):
        reader = dict(id="refined_spatial_" + str(width), kind="spatial", model="resolution-matched pose-only instrument",
            native_resolution=width, encoder_resolution=384, role="external_pose_instrument",
            checkpoint_sha256={f: sha(DIRECTORY / (f + "_pose_head.pt")) for f in heads},
            specification_sha256=sha(DIRECTORY / "specification.json"))
        readers.append(reader)
        for row in records:
            net, metadata = heads[row["category"]]
            h = np.load(DIRECTORY / "features" / str(width) / (row["video_id"] + ".npy"))
            c = torch.tensor((camera(row) - metadata["cmean"]) / metadata["cstd"], device="cuda").expand(120, -1)
            with torch.no_grad():
                q = net(torch.from_numpy(h).float().cuda(), c).cpu().numpy()
            nq = int(metadata["nq"])
            q = q * metadata["ystd"][:nq] + metadata["ymean"][:nq]
            q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
            save(reader, row, qpos=q, time=np.arange(120)/30)
    write(DIRECTORY / "readers.json", readers)
    # The original scorer hard-codes the old observer's prefix. Measure this
    # new pipeline through its own wrapper so the 256px prefix is also refined.
    tasks = [(row, reader) for reader in readers for row in records]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for index, future in enumerate(as_completed([pool.submit(score_case, task) for task in tasks])):
            future.result()
            if (index + 1) % 12 == 0:
                print(json.dumps(dict(stage="score", completed=index + 1, total=len(tasks))), flush=True)


def score_case(task):
    from .visual_calibration import readout_path, error_statistics
    from experiments.model_expansion.evaluate_spatial import Audit
    row, reader = task
    audit = Audit(row)
    path = readout_path(reader, row)
    with np.load(path) as data:
        q, t = data["qpos"].astype(float), data["time"]
    if reader["native_resolution"] == 256:
        with np.load(OUT / "readouts/refined_spatial_384" / path.name) as data:
            q[:3] = data["qpos"][:3]
    with np.load(ROOT / row["evaluator_directory"] / "physics.npz") as data:
        truth = data["qpos"][np.arange(120)*8]
    measured = audit.measure(t, q, truth)
    points, ref = np.stack([audit.points(p) for p in q]), np.stack([audit.points(p) for p in truth])
    from .rescore_continuous import spatial_values
    original = json.loads((OUT / "measurements/shared_spatial_384" / (row["video_id"] + ".json")).read_text())
    result = dict(reader_id=reader["id"], kind="spatial", category=row["category"], assembly_id=row["assembly_id"],
        case_id=row["video_id"], calibration_role=row["calibration_role"],
        identity=dict(protocol_sha256=sha(OUT / "protocol.json"), scorer_sha256=sha(__file__), readout_sha256=sha(path)),
        localization=error_statistics(points[3:] - ref[3:]), native_metrics=spatial_values(measured),
        native_qualified=measured.get("mechanical_coverage", False),
        numerical_reference_exact=original["numerical_reference_exact"],
        numerical_reference_sampled=original["numerical_reference_sampled"],
        native_raw_equation_diagnostics=dict(dynamics=measured.get("raw_equivalent_effort_rmse"), energy=measured.get("raw_work_energy_max_J")))
    try:
        projected = np.stack([audit.project(p) for p in q])
        corrected = audit.measure(t, projected, truth)
        corrected_points = np.stack([audit.points(p) for p in projected])
        result.update(intervention_metrics=spatial_values(corrected), intervention_reachable=True,
                      intervention_qualified=corrected.get("mechanical_coverage", False),
                      correction_point_rms_m=float(np.sqrt(np.mean(np.sum((corrected_points[3:] - points[3:])**2, axis=-1)))))
    except ValueError as error:
        result.update(intervention_metrics={}, intervention_reachable=False, intervention_qualified=False,
                      intervention_failure=str(error))
    write(OUT / "measurements" / reader["id"] / (row["video_id"] + ".json"), result)
    return result


def run():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    protocol = load_protocol()
    manifest = json.loads((ROOT / "output/model_expansion/spatial/dataset_manifest.json").read_text())
    rows = [r for r in manifest["videos"] if r["split"] == "train" and r["view"] == "front"]
    spec = dict(source_sha256=sha(__file__), seed=42, epochs=160, fitting_split="train/front", input_widths=[256,384],
        fit_video_ids=[r["video_id"] for r in rows], outputs="qpos only, no closure constraint",
        prior="original training-fitted pose/velocity head; shared hidden layers retained, velocity outputs removed",
        final_checkpoint_selection="fixed epoch", precision="unchanged BF16 encoder; float32 pose head",
        learning_rate=3e-5, final_learning_rate=1e-6, weight_decay=.001,
        calibration_protocol_sha256=sha(OUT / "protocol.json"))
    path = DIRECTORY / "specification.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Refinement specification changed")
    write(path, spec)
    cache_features(rows)
    for family in ("H-A", "H-B"):
        train(family, [r for r in rows if r["category"] == family])
    cache_features(protocol["spatial_records"])
    evaluate(protocol["spatial_records"])


if __name__ == "__main__":
    run()
