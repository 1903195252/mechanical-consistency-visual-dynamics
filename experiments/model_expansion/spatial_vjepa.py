"""V-JEPA 2.1-B + released 24-layer AC predictor, spatial adaptation.

Cached framewise encodings preserve the official frame-causal AC recipe.
The additional physical readout is fitted on training states, and never enforces
closed-chain geometry. Test inference opens only prefix features and commands.
"""

import argparse
import gc
import json
import math
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .prepare_spatial import ROOT, DESTINATION, write_json, sha256
from .profile_vjepa import clean_weights, WEIGHTS, UPSTREAM
from .checkpoint_io import load_checkpoint


FEATURES = DESTINATION / "features"
TOKENS, DIM = 576, 768
SEQUENCE_EXPOSURES, BATCH, SEED = 3000, 8, 42


def deadline():
    from experiments.runtime import check_deadline
    check_deadline()


def records(category=None):
    data = json.loads((DESTINATION / "dataset_manifest.json").read_text())["videos"]
    return [r for r in data if (category is None or r["category"] == category)
            and (r["split"] == "train" or r["view"] == "front")]


def cache_path(row):
    return FEATURES / (row["video_id"] + ".npy")


def camera(row):
    public = json.loads((ROOT / row["public_inputs"]).read_text())
    cam = public["observed_camera_frames"][0]
    return np.r_[np.asarray(cam["R_world_to_camera"]).ravel(),
                 cam["t_world_to_camera_m"], cam["scale_m"]].astype(np.float32)


def inputs(row):
    """Only public sidecar and planned commands; no evaluation reference."""
    public = json.loads((ROOT / row["public_inputs"]).read_text())
    with np.load(ROOT / row["commands"]) as data:
        if row["category"] == "H-B":
            p, v = "shoulder_angle_command_rad", "shoulder_velocity_command_rad_s"
        else:
            p, v = "commanded_prismatic_displacement_m", "commanded_prismatic_velocity_m_s"
        times = np.arange(120) / 30
        values = np.concatenate([data[p], data[v]], axis=1)
        action = np.stack([np.interp(times, data["time"], values[:, j])
                           for j in range(values.shape[1])], axis=1)
    return action.astype(np.float32), public


def labels(row):
    if row["split"] == "test":
        raise ValueError("Test labels are available only to the separate evaluator")
    compact = DESTINATION / "training_labels" / (row["assembly_id"] + "__" + row["condition"] + ".npz")
    if compact.exists():
        with np.load(compact) as data:
            return data["labels"], data["state"], data["qi"], data["vi"]
    directory = ROOT / row["evaluator_directory"]
    with np.load(directory / "physics.npz") as data:
        ix = np.arange(120) * 8
        q, v = data["qpos"][ix], data["qvel"][ix]
    # Every spatial model has one free root joint followed by scalar joints.
    if row["category"] == "H-A":
        qi, vi = np.arange(9, 25, 3), np.arange(8, 24, 3)
    else:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(directory / "model.xml"))
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drive_%d" % i)
                  for i in range(3)]
        qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    state = np.concatenate([q[:, qi], v[:, vi]], axis=1)
    return np.concatenate([q, v], axis=1).astype(np.float32), state.astype(np.float32), qi, vi


def pooled(features):
    return F.adaptive_avg_pool2d(features.reshape(-1, 24, 24, DIM).permute(0, 3, 1, 2),
                                (4, 4)).flatten(1)


def cache(category):
    sys.path.insert(0, str(UPSTREAM))
    from src.hub.backbones import vjepa2_1_vit_base_384
    FEATURES.mkdir(parents=True, exist_ok=True)
    encoder, predictor = vjepa2_1_vit_base_384(pretrained=False)
    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    encoder.load_state_dict(clean_weights(checkpoint["ema_encoder"]), strict=True)
    predictor.load_state_dict(clean_weights(checkpoint["predictor"]), strict=True)
    del checkpoint, predictor
    gc.collect()
    encoder.eval().requires_grad_(False).cuda()
    mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1)
    rows = records(category)
    cv2.setNumThreads(1)
    start = time.monotonic()
    for number, row in enumerate(rows):
        deadline()
        count = 3 if row["split"] == "test" else 120
        output = cache_path(row)
        if output.exists():
            existing = np.load(output, mmap_mode="r")
            if existing.shape == (count, TOKENS, DIM):
                continue
            raise ValueError("Stale feature dimensions: %s" % output)
        capture = cv2.VideoCapture(str(ROOT / row["video"]))
        frames = []
        try:
            for _ in range(count):
                ok, image = capture.read()
                if not ok:
                    raise ValueError("Incomplete video: %s" % row["video"])
                frames.append(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                         (384, 384), interpolation=cv2.INTER_CUBIC))
        finally:
            capture.release()
        temporary = output.with_suffix(".tmp.npy")
        target = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                           shape=(count, TOKENS, DIM))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for first in range(0, count, 32):
                x = torch.from_numpy(np.stack(frames[first:first + 32])).cuda()
                x = (x.permute(0, 3, 1, 2).float() / 255 - mean) / std
                # Independent duplicated-frame tubelets do not expose future frames.
                z = encoder(x.unsqueeze(2).repeat(1, 1, 2, 1, 1))
                z = F.layer_norm(z, (DIM,))
                if not torch.isfinite(z).all():
                    raise FloatingPointError("Encoder produced nonfinite features")
                target[first:first + len(x)] = z.float().cpu().numpy().astype(np.float16)
        target.flush()
        del target
        temporary.replace(output)
        if (number + 1) % 10 == 0 or number + 1 == len(rows):
            print(json.dumps({"stage": "features", "category": category,
                              "videos": number + 1, "total": len(rows),
                              "seconds": time.monotonic() - start}), flush=True)
    write_json(DESTINATION / (category + "_cache.json"), {
        "completed": True, "videos": len(rows),
        "test_frames_encoded": 3, "train_validation_frames_encoded": 120,
        "encoder": "official V-JEPA 2.1-B/16 at 384px; strict checkpoint loading",
        "checkpoint_sha256": sha256(WEIGHTS),
        "sampling": "30 Hz, one duplicated-frame tubelet per real frame",
        "seconds": time.monotonic() - start,
    })


class Readout(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(16 * DIM + 13, 512), nn.GELU(),
                                     nn.Linear(512, 256), nn.GELU(), nn.Linear(256, output_dim))

    def forward(self, visual, calibration):
        return self.network(torch.cat([visual, calibration], dim=-1))


def normalized(values):
    mean = values.mean(0)
    std = np.maximum(values.std(0), 1e-3)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_readout(category):
    sys.path.insert(0, str(UPSTREAM))
    from src.models.ac_predictor import vit_ac_predictor
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    train_rows = [r for r in records(category) if r["split"] == "train"]
    pool_x, pool_c, pool_y, commands, states, tracks = [], [], [], [], [], []
    qi = vi = None
    for row in train_rows:
        deadline()
        z = np.load(cache_path(row), mmap_mode="r")
        y, state, qi, vi = labels(row)
        action, _ = inputs(row)
        with torch.no_grad():
            for first in range(0, 120, 32):
                p = pooled(torch.tensor(np.array(z[first:first + 32]), device="cuda").float())
                pool_x.append(p.cpu().numpy())
        pool_c.append(np.tile(camera(row), (120, 1)))
        pool_y.append(y)
        commands.append(action)
        states.append(state)
        tracks.append(z)
    yall, call = np.concatenate(pool_y), np.concatenate(pool_c)
    ymean, ystd = normalized(yall)
    cmean, cstd = normalized(call)
    amean, astd = normalized(np.concatenate(commands))
    smean, sstd = normalized(np.concatenate(states))
    nq = (pool_y[0].shape[1] + 1) // 2
    head = Readout(yall.shape[1]).cuda()
    hx = torch.tensor(np.concatenate(pool_x), device="cuda")
    hc = torch.tensor((call - cmean) / cstd, device="cuda")
    hy = torch.tensor((yall - ymean) / ystd, device="cuda")
    head_optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    for epoch in range(60):
        deadline()
        order = torch.randperm(len(hx), device="cuda")
        losses = []
        for batch in order.split(256):
            head_optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.mse_loss(head(hx[batch], hc[batch]), hy[batch])
            loss.backward()
            head_optimizer.step()
            losses.append(loss.detach())
        if (epoch + 1) % 10 == 0:
            print(json.dumps({"stage": "readout", "category": category, "epoch": epoch + 1,
                              "training_loss": torch.stack(losses).mean().item()}), flush=True)
    head.eval().requires_grad_(False)
    metadata = dict(ymean=ymean, ystd=ystd, cmean=cmean, cstd=cstd,
                    amean=amean, astd=astd, smean=smean, sstd=sstd,
                    nq=nq, qi=np.asarray(qi), vi=np.asarray(vi),
                    category=category, seed=SEED)
    # Readout-only validation measures coordinate anchoring on held-out identities.
    validation = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in records(category):
            if row["split"] != "validation":
                continue
            z = np.load(cache_path(row), mmap_mode="r")
            y, _, _, _ = labels(row)
            cp = torch.tensor((camera(row) - cmean) / cstd, device="cuda").expand(120, -1)
            zp = pooled(torch.tensor(np.array(z), device="cuda").float())
            prediction = head(zp, cp).float().cpu().numpy() * ystd + ymean
            validation.append({"video_id": row["video_id"],
                               "platform_position_rmse_mm": float(np.sqrt(
                                   np.mean(np.sum((prediction[:, :3] - y[:, :3])**2, axis=-1))) * 1000)})
    write_json(DESTINATION / (category + "_readout_validation.json"), validation)
    torch.save({"readout": head.cpu().state_dict(), "metadata": metadata},
               DESTINATION / (category + "_readout.pt"))
    del head, head_optimizer, hx, hc, hy, pool_x, pool_c, pool_y
    gc.collect()
    torch.cuda.empty_cache()
    return metadata


def train(category):
    sys.path.insert(0, str(UPSTREAM))
    from src.models.ac_predictor import vit_ac_predictor
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    readout_path = DESTINATION / (category + "_readout.pt")
    if readout_path.exists():
        metadata = load_checkpoint(readout_path)["metadata"]
    else:
        metadata = fit_readout(category)
    rows = [r for r in records(category) if r["split"] == "train"]
    tracks = [np.load(cache_path(row), mmap_mode="r") for row in rows]
    commands = np.stack([inputs(row)[0] for row in rows])
    states = np.stack([labels(row)[1] for row in rows])
    amean, astd = metadata["amean"], metadata["astd"]
    smean, sstd = metadata["smean"], metadata["sstd"]
    epochs = math.ceil(SEQUENCE_EXPOSURES / len(rows))

    predictor = vit_ac_predictor(img_size=(384, 384), patch_size=16,
        num_frames=16, tubelet_size=2, embed_dim=DIM, predictor_embed_dim=1024,
        depth=24, num_heads=16, action_embed_dim=len(amean),
        is_frame_causal=True, use_activation_checkpointing=False,
        use_rope=True, use_extrinsics=False).cuda()
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=4.25e-4,
                                 weight_decay=.04, fused=True)
    commands = (commands - amean) / astd
    states = (states - smean) / sstd
    logs, updates, exposures = [], 0, 0
    start_epoch, prior_seconds = 0, 0.
    checkpoint_path = DESTINATION / (category + "_dynamics.pt")
    if checkpoint_path.exists():
        saved = load_checkpoint(checkpoint_path)
        predictor.load_state_dict(saved["predictor"], strict=True)
        for group in saved["optimizer"]["param_groups"]:
            group["fused"], group["foreach"] = True, False
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["epoch"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        del saved
        logs = [r for r in json.loads((DESTINATION / (category + "_training.json")).read_text())
                if r["epoch"] <= start_epoch]
        updates, exposures, prior_seconds = logs[-1]["updates"], logs[-1]["sequence_exposures"], logs[-1]["elapsed_s"]
        print(json.dumps({"stage": "resume", "category": category, "epoch": start_epoch}), flush=True)
    start = time.monotonic()
    for epoch in range(start_epoch, epochs):
        deadline()
        predictor.train()
        warmup_epochs = max(1, round(epochs * .05))
        lr = 4.25e-4 * min((epoch + 1) / warmup_epochs, 1) * .5 * (1 + math.cos(math.pi * epoch / epochs))
        for group in optimizer.param_groups:
            group["lr"] = lr
        order = np.random.permutation(len(tracks))
        losses = []
        for first in range(0, len(order), BATCH):
            deadline()
            remaining_exposures = SEQUENCE_EXPOSURES - exposures
            if remaining_exposures <= 0:
                break
            batch = order[first:first + min(BATCH, remaining_exposures)]
            offsets = np.random.randint(0, 113, len(batch))
            z = torch.tensor(np.stack([tracks[i][k:k + 8] for i, k in zip(batch, offsets)]), device="cuda")
            z = z.flatten(1, 2)
            a = torch.tensor(np.stack([commands[i, k:k + 7] for i, k in zip(batch, offsets)]), device="cuda")
            s = torch.tensor(np.stack([states[i, k:k + 7] for i, k in zip(batch, offsets)]), device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                teacher = F.layer_norm(predictor(z[:, :-TOKENS], a, s), (DIM,))
                two = torch.cat([z[:, :TOKENS], teacher[:, :TOKENS]], dim=1)
                second = F.layer_norm(predictor(two, a[:, :2], s[:, :2]), (DIM,))[:, -TOKENS:]
                ar = torch.cat([teacher[:, :TOKENS], second], dim=1)
                loss = F.l1_loss(teacher, z[:, TOKENS:]) + F.l1_loss(ar, z[:, TOKENS:3 * TOKENS])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite dynamics loss")
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
            updates += 1
            exposures += len(batch)
        row = {"epoch": epoch + 1, "loss": torch.stack(losses).mean().item(),
               "updates": updates, "sequence_exposures": exposures,
               "elapsed_s": prior_seconds + time.monotonic() - start}
        logs.append(row)
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(json.dumps(dict(stage="dynamics", category=category, **row)), flush=True)
            write_json(DESTINATION / (category + "_training.json"), logs)
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            temporary = DESTINATION / (category + "_dynamics.tmp.pt")
            torch.save({"predictor": predictor.state_dict(), "optimizer": optimizer.state_dict(),
                        "metadata": metadata, "epoch": epoch + 1,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                        "numpy_rng": np.random.get_state(), "registered_epochs": epochs}, temporary)
            temporary.replace(DESTINATION / (category + "_dynamics.pt"))
    # Full-precision inference weights are sufficient to reproduce forecasts;
    # the larger optimizer checkpoint remains available for training continuation.
    torch.save({"predictor": predictor.state_dict(), "metadata": metadata,
                "epoch": epochs, "registered_epochs": epochs},
               DESTINATION / (category + "_predictor.pt"))


def infer(category):
    sys.path.insert(0, str(UPSTREAM))
    from src.models.ac_predictor import vit_ac_predictor
    # Locally generated training checkpoints contain NumPy normalization arrays.
    readout = load_checkpoint(DESTINATION / (category + "_readout.pt"))
    checkpoint_path = DESTINATION / (category + "_dynamics.pt")
    if not checkpoint_path.exists():
        checkpoint_path = DESTINATION / (category + "_inference_weights.pt")
    saved = load_checkpoint(checkpoint_path)
    if saved["epoch"] != saved["registered_epochs"]:
        raise ValueError("Incomplete training checkpoint is not a final experiment")
    final_epoch = saved["epoch"]
    m = readout["metadata"]
    head = Readout(len(m["ymean"])).cuda().eval()
    head.load_state_dict(readout["readout"], strict=True)
    predictor = vit_ac_predictor(img_size=(384, 384), patch_size=16,
        num_frames=16, tubelet_size=2, embed_dim=DIM, predictor_embed_dim=1024,
        depth=24, num_heads=16, action_embed_dim=len(m["amean"]),
        is_frame_causal=True, use_rope=True, use_extrinsics=False).cuda().eval()
    predictor.load_state_dict(saved["predictor"], strict=True)
    del saved, readout
    gc.collect()
    dest = DESTINATION / "predictions" / category
    dest.mkdir(parents=True, exist_ok=True)
    rows = [r for r in records(category) if r["split"] == "test"]
    # Batching across all held-out conditions amortizes the autoregressive launch overhead.
    z = torch.tensor(np.stack([np.load(cache_path(r)) for r in rows]), device="cuda")
    if z.shape[1] != 3:
        raise ValueError("Inference must have only three observed frames")
    actions = np.stack([inputs(r)[0] for r in rows])
    actions = torch.tensor((actions - m["amean"]) / m["astd"], device="cuda")
    cams = torch.tensor((np.stack([camera(r) for r in rows]) - m["cmean"]) / m["cstd"], device="cuda")
    ym = torch.tensor(m["ymean"], device="cuda")
    ys = torch.tensor(m["ystd"], device="cuda")
    sm = torch.tensor(m["smean"], device="cuda")
    ss = torch.tensor(m["sstd"], device="cuda")
    qi, vi, nq = m["qi"], m["vi"], int(m["nq"])
    predictions, state_history = [], []

    def decode(feature):
        result = head(pooled(feature.float()), cams).float() * ys + ym
        quaternion = result[:, 3:7]
        result[:, 3:7] = F.normalize(quaternion, dim=-1)
        state = torch.cat([result[:, qi], result[:, nq + vi]], dim=-1)
        return result, (state - sm) / ss

    start = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for frame in range(3):
            y, state = decode(z[:, frame])
            predictions.append(y.cpu().numpy())
            state_history.append(state)
        history = z
        for frame in range(3, 120):
            deadline()
            window = min(7, history.shape[1])
            context = history[:, -window:].flatten(1, 2)
            states = torch.stack(state_history[-window:], dim=1)
            nxt = F.layer_norm(predictor(context, actions[:, frame - window:frame], states), (DIM,))[:, -TOKENS:]
            y, state = decode(nxt)
            predictions.append(y.cpu().numpy())
            state_history.append(state)
            history = torch.cat([history[:, -6:], nxt.unsqueeze(1)], dim=1)
    predictions = np.stack(predictions, axis=1)
    for row, trajectory in zip(rows, predictions):
        np.savez_compressed(dest / (row["video_id"] + ".npz"),
                            time=np.arange(120) / 30, qpos=trajectory[:, :nq],
                            readout_velocity=trajectory[:, nq:], observed_frames=np.array(3))
    write_json(DESTINATION / (category + "_inference.json"), {
        "cases": len(rows), "completed_frames_per_case": 120, "observed_frames": 3,
        "future_steps": 117, "context_frames": 7, "seconds": time.monotonic() - start,
        "future_truth_opened": False, "final_epoch": final_epoch,
        "physical_readout": "training-supervised free Cartesian root pose and scalar tree joints",
        "quaternion_handling": "unit normalization only; no closed-chain projection",
    })
    print(json.dumps({"stage": "inference", "category": category,
                      "cases": len(rows), "seconds": time.monotonic() - start}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["cache", "train", "infer"])
    parser.add_argument("--category", choices=["H-A", "H-B"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    {"cache": cache, "train": train, "infer": infer}[args.stage](args.category)


if __name__ == "__main__":
    main()
