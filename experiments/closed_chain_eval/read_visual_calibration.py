"""Decode all held-out reference frames using unchanged frozen visual readers."""
import argparse
import gc
import json
import os
import pickle
import sys
import tarfile
import time

import numpy as np

from .visual_calibration import ROOT, OUT, load_protocol, readout_path, sha, write


def prepare_con_runtime():
    """Restore the exact imports used by the original training deployment.

    The upstream package exports unrelated optional models by default. The
    recorded deployment already restricts those exports to the trained model.
    Restore that package, with all encoder/model implementation bytes unchanged.
    """
    required = {"src/__init__.py", "src/models/__init__.py",
                "src/models/autoencoders/__init__.py", "src/models/autoencoders/vae.py",
                "src/models/autoencoders/simple_cnn.py", "src/models/neural_odes/__init__.py",
                "src/models/neural_odes/con_iae.py", "src/models/neural_odes/neural_ode_base.py",
                "src/models/utils.py", "src/models/dynamics_autoencoder.py"}
    with tarfile.open(ROOT / "output/con_corpus/deployment.tar.gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith("official/"):
                continue
            relative = member.name[len("official/"):]
            if relative in required:
                path = OUT / "runtime" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                content = archive.extractfile(member).read()
                if not path.exists() or path.read_bytes() != content:
                    path.write_bytes(content)


def save(reader, row, **arrays):
    path = readout_path(reader, row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, protocol_sha256=np.array(sha(OUT / "protocol.json")), **arrays)
    temporary.replace(path)


def done(reader, row):
    path = readout_path(reader, row)
    if not path.exists():
        return False
    with np.load(path) as data:
        if str(data["protocol_sha256"]) != sha(OUT / "protocol.json"):
            raise ValueError("Cached readout has a different calibration protocol")
    return True


def planar(protocol, model_name):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if model_name == "con":
        prepare_con_runtime()
    sys.path.insert(0, str(ROOT / "output/keycld_corpus/runtime"))
    sys.path.insert(0, str(OUT / "runtime"))
    import jax
    import jax.numpy as jnp
    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree_util.tree_map
    records = protocol["planar_records"]
    start = time.monotonic()
    for reader in protocol["readers"]:
        if reader["kind"] != "planar" or reader["model"] != model_name:
            continue
        folder = ROOT / reader["folder"]
        if sha(folder / "params.p") != reader["checkpoint_sha256"]:
            raise ValueError("Reader checkpoint changed")
        with (folder / "params.p").open("rb") as stream:
            params = pickle.load(stream)
        if model_name == "keycld":
            from keycld import util
            from experiments.keycld_corpus.model import ControlledKeyCLD
            permutation = json.loads((folder / "scores.json").read_text())["reader_permutation_from_validation"]

            @jax.jit
            def encode(images, geometry):
                model = ControlledKeyCLD(geometry)
                return util.map_to_keypoints(model.encoder(params, images))[0]

            def points(latent):
                return np.asarray(latent).reshape(-1, 3, 2)[:, permutation] * .65 + [.2, .35]
        else:
            from experiments.con_corpus.common import model
            network = model()
            affine = np.asarray(json.loads((folder / "readout.json").read_text())["affine"])

            @jax.jit
            def encode(images, geometry):
                return network.apply({"params": params}, images * 2 - 1, method=network.encode)

            def points(latent):
                decoded = np.asarray(latent) @ affine[:8] + affine[8]
                return decoded.reshape(-1, 3, 2) * .65 + [.2, .35]

        for index, row in enumerate(records):
            if done(reader, row):
                continue
            source = ROOT / "output/keycld_corpus" / row["observations"]
            if sha(source) != row["observations_sha256"]:
                raise ValueError("Reference RGB changed")
            with np.load(source) as data:
                images = data["x"].astype(np.float32) / 255
                geometry = data["geometry"]
            latent = np.concatenate([np.asarray(encode(jnp.asarray(images[i:i + 15]), geometry))
                                     for i in range(0, len(images), 15)])
            result = points(latent)
            save(reader, row, points=result, time=np.arange(len(result)) / 30,
                 finite=np.array(bool(np.isfinite(result).all())))
            if (index + 1) % 6 == 0 or index == len(records) - 1:
                print(json.dumps(dict(reader=reader["id"], videos=index + 1, total=len(records),
                                      seconds=round(time.monotonic() - start, 1))), flush=True)
        del params
        jax.clear_caches()
        gc.collect()


def spatial(protocol, batch_size):
    import cv2
    import torch
    import torch.nn.functional as F
    from experiments.model_expansion.profile_vjepa import UPSTREAM, WEIGHTS, clean_weights
    from experiments.model_expansion.spatial_vjepa import Readout, pooled, camera, DIM
    from experiments.model_expansion.checkpoint_io import load_checkpoint
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
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
    heads = {}
    for family in ("H-A", "H-B"):
        checkpoint = load_checkpoint(ROOT / "output/model_expansion/spatial" / (family + "_readout.pt"))
        metadata = checkpoint["metadata"]
        head = Readout(len(metadata["ymean"])).cuda().eval().requires_grad_(False)
        head.load_state_dict(checkpoint["readout"], strict=True)
        heads[family] = head, metadata
    started = time.monotonic()
    readers = [r for r in protocol["readers"] if r["kind"] == "spatial"]
    with torch.inference_mode():
        for reader in readers:
            for index, row in enumerate(protocol["spatial_records"]):
                if done(reader, row):
                    continue
                cap = cv2.VideoCapture(str(ROOT / row["video"]))
                images = []
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if reader["native_resolution"] == 256:
                            frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
                        images.append(cv2.resize(frame, (384, 384), interpolation=cv2.INTER_CUBIC))
                finally:
                    cap.release()
                if len(images) != 120:
                    raise ValueError("Incomplete reference video")
                head, metadata = heads[row["category"]]
                calibration = torch.tensor((camera(row) - metadata["cmean"]) / metadata["cstd"], device="cuda")
                outputs = []
                for first in range(0, 120, batch_size):
                    x = torch.from_numpy(np.stack(images[first:first + batch_size])).cuda().permute(0, 3, 1, 2).float() / 255
                    x = (x - mean) / std
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        feature = F.layer_norm(encoder(x.unsqueeze(2).repeat(1, 1, 2, 1, 1)), (DIM,))
                        feature = feature.to(torch.float16).float()
                        value = head(pooled(feature), calibration.expand(len(x), -1))
                    outputs.append(value.float().cpu().numpy())
                result = np.concatenate(outputs) * metadata["ystd"] + metadata["ymean"]
                norm = np.linalg.norm(result[:, 3:7], axis=1, keepdims=True)
                if not np.isfinite(result).all() or np.any(norm < 1e-8):
                    raise FloatingPointError("Invalid observer quaternion/state")
                result[:, 3:7] /= norm
                nq = int(metadata["nq"])
                save(reader, row, qpos=result[:, :nq], readout_velocity=result[:, nq:],
                     time=np.arange(120) / 30, finite=np.array(True))
                print(json.dumps(dict(reader=reader["id"], videos=index + 1, total=24,
                                      seconds=round(time.monotonic() - started, 1))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", required=True, choices=("keycld", "con", "spatial"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    protocol = load_protocol()
    if args.reader == "spatial":
        spatial(protocol, args.batch_size)
    else:
        planar(protocol, args.reader)
    write(OUT / (args.reader + "_readout_complete.json"), dict(complete=True, protocol_sha256=sha(OUT / "protocol.json")))
