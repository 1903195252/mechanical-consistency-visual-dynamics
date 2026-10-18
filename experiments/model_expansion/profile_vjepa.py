"""Measure the official encoder and full AC predictor on a real training clip.

This produces resource measurements, not model-quality or publication results.
The AC predictor is initialized for this profile; no trained dynamics are claimed.
"""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from .prepare_spatial import ROOT, DESTINATION, sha256, write_json


UPSTREAM = ROOT / "output/model_expansion/upstream/vjepa2"
WEIGHTS = ROOT / "output/model_expansion/weights/vjepa2_1_vitb_dist_vitG_384.pt"


def load_video_frames(path, indices):
    capture = cv2.VideoCapture(str(path))
    result = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = capture.read()
            if not success:
                raise ValueError("Cannot decode frame %d of %s" % (index, path))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result.append(cv2.resize(rgb, (384, 384), interpolation=cv2.INTER_CUBIC))
    finally:
        capture.release()
    tensor = torch.from_numpy(np.stack(result)).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    return (tensor - mean) / std


def memory():
    torch.cuda.synchronize()
    return {
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20,
        "free_device_MiB": torch.cuda.mem_get_info()[0] / 2**20,
    }


def clean_weights(state):
    return {key.replace("module.", "").replace("backbone.", ""): value
            for key, value in state.items()}


def profile(args, report):
    sys.path.insert(0, str(UPSTREAM))
    from src.hub.backbones import vjepa2_1_vit_base_384
    from src.models.ac_predictor import vit_ac_predictor

    torch.manual_seed(42)
    torch.set_num_threads(4)
    manifest = json.loads((DESTINATION / "dataset_manifest.json").read_text())
    sample = next(row for row in manifest["videos"] if row["category"] == "H-B"
                  and row["split"] == "train" and row["view"] == "front"
                  and row["condition"] == "low_speed")
    report.update(
        video=sample["video"], assembly=sample["assembly_id"], split=sample["split"],
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
        checkpoint_sha256=sha256(WEIGHTS),
        checkpoint_url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        total_device_MiB=torch.cuda.get_device_properties(0).total_memory / 2**20,
        precision="float32 weights, bfloat16 autocast", resolution=[384, 384],
    )
    print("Loading official checkpoint", flush=True)
    encoder, pretraining_predictor = vjepa2_1_vit_base_384(pretrained=False)
    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    # Same keys and strict loading as the official Hub entrypoint; its network
    # downloader is bypassed to load the locally verified checkpoint.
    encoder.load_state_dict(clean_weights(checkpoint["ema_encoder"]), strict=True)
    pretraining_predictor.load_state_dict(clean_weights(checkpoint["predictor"]), strict=True)
    report["strict_checkpoint_load"] = True
    report["encoder_parameters"] = sum(p.numel() for p in encoder.parameters())
    del checkpoint, pretraining_predictor
    gc.collect()
    encoder.eval().requires_grad_(False).cuda()

    # The inference check sees exactly the declared prefix. Each observed frame
    # is duplicated into one tubelet, as in the official frame-causal AC recipe.
    prefix = load_video_frames(ROOT / sample["video"], list(range(sample["observed_prefix_frames"])))
    torch.cuda.reset_peak_memory_stats()
    timings, outputs = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for frame in prefix:
            clip = frame.unsqueeze(0).unsqueeze(2).repeat(1, 1, 2, 1, 1).cuda()
            torch.cuda.synchronize()
            start = time.perf_counter()
            feature = encoder(clip)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
            outputs.append(feature.cpu())
    report["prefix_encoding"] = dict(
        observed_frames=sample["observed_prefix_frames"],
        future_frames_read=False,
        tubelet_policy="duplicate each observed frame independently",
        feature_shape=list(outputs[0].shape),
        per_frame_seconds=timings,
        finite=all(torch.isfinite(out).all().item() for out in outputs),
        **memory(),
    )
    write_json(args.output, report)
    print("Prefix encoding:", json.dumps(report["prefix_encoding"]), flush=True)
    if not args.ac_step:
        return

    # A training-only resource profile can use training targets and proprioception.
    # Eight frames at 4 Hz follow the released AC clip length and sample rate.
    frame_indices = np.rint(np.arange(8) * 30 / 4).astype(int)
    frames = load_video_frames(ROOT / sample["video"], frame_indices)
    features = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for frame in frames:
            clip = frame.unsqueeze(0).unsqueeze(2).repeat(1, 1, 2, 1, 1).cuda()
            features.append(F.layer_norm(encoder(clip), (encoder.embed_dim,)).cpu())
    features = torch.cat(features, dim=1)
    embed_dim = encoder.embed_dim
    del encoder, clip, feature, outputs
    gc.collect()
    torch.cuda.empty_cache()

    import mujoco
    directory = ROOT / sample["evaluator_directory"]
    model = mujoco.MjModel.from_xml_path(str(directory / "model.xml"))
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drive_%d" % i)
              for i in range(3)]
    qi = model.jnt_qposadr[joints]
    vi = model.jnt_dofadr[joints]
    physics = np.load(directory / "physics.npz")
    commands = np.load(ROOT / sample["commands"])
    sample_times = frame_indices / 30.0

    def interpolate(values, times):
        return np.stack([np.interp(sample_times, times, values[:, i])
                         for i in range(values.shape[1])], axis=1)

    action = np.concatenate([
        interpolate(commands["shoulder_angle_command_rad"], commands["time"]),
        interpolate(commands["shoulder_velocity_command_rad_s"], commands["time"]),
    ], axis=1)
    state = np.concatenate([
        interpolate(physics["qpos"][:, qi], physics["time"]),
        interpolate(physics["qvel"][:, vi], physics["time"]),
    ], axis=1)
    physics.close()
    commands.close()
    predictor = vit_ac_predictor(
        img_size=(384, 384), patch_size=16, num_frames=16, tubelet_size=2,
        embed_dim=embed_dim, predictor_embed_dim=1024, depth=24, num_heads=16,
        action_embed_dim=6, use_activation_checkpointing=True,
        is_frame_causal=True, use_rope=True, use_extrinsics=False,
    ).cuda().train()
    report["ac_training_configuration"] = {
        "parameters": sum(p.numel() for p in predictor.parameters()),
        "initialization": "random; resource profile only",
        "depth": 24, "width": 1024, "heads": 16, "batch_size": 1,
        "frames": 8, "frame_indices": frame_indices.tolist(),
        "nominal_fps": 4, "activation_checkpointing": True,
        "loss": "official teacher-forcing L1 plus two-step autoregressive L1",
        "adaptation": "6D Delta command/state interface and 768D ViT-B features",
        "state_input": "training-only shoulder angles and velocities; SI units",
        "encoder_during_training": "frozen features cached; encoder unloaded",
    }
    write_json(args.output, report)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=4.25e-4,
                                 weight_decay=.04, foreach=False)
    target = features.to("cuda")
    actions = torch.tensor(action[:-1], dtype=torch.float32, device="cuda").unsqueeze(0)
    states = torch.tensor(state[:-1], dtype=torch.float32, device="cuda").unsqueeze(0)
    tokens = 24 * 24

    def predict(z, a, s):
        return F.layer_norm(predictor(z, a, s), (embed_dim,))

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    print("Profiling full 24-layer AC forward, backward, and AdamW update", flush=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        teacher = predict(target[:, :-tokens], actions, states)
        first_two = torch.cat([target[:, :tokens], teacher[:, :tokens]], dim=1)
        second = predict(first_two, actions[:, :2], states[:, :2])[:, -tokens:]
        autoregressive = torch.cat([teacher[:, :tokens], second], dim=1)
        loss = F.l1_loss(teacher, target[:, tokens:]) + F.l1_loss(
            autoregressive, target[:, tokens:3 * tokens])
    loss.backward()
    gradients_finite = all(p.grad is None or torch.isfinite(p.grad).all().item()
                           for p in predictor.parameters())
    if not gradients_finite or not torch.isfinite(loss).item():
        raise FloatingPointError("Non-finite AC loss or gradient")
    optimizer.step()
    torch.cuda.synchronize()
    report["ac_training_step"] = dict(
        seconds=time.perf_counter() - start,
        completed_optimizer_updates=1, gradients_finite=gradients_finite,
        **memory(),
    )
    print("AC training:", json.dumps(report["ac_training_step"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ac-step", action="store_true")
    parser.add_argument("--output", type=Path, default=DESTINATION / "vjepa_local_profile.json")
    args = parser.parse_args()
    report = {"purpose": "local resource profile; not scientific model results", "status": "running"}
    try:
        profile(args, report)
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        if torch.cuda.is_initialized():
            report["failure_memory"] = memory()
        traceback.print_exc()
        raise
    finally:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
