"""Independent reference-video calibration of frozen physical readouts.

The specification is frozen before reference videos are decoded. No candidate
rollout or formal test identity is used to estimate the measurement errors.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/closed_chain_eval/visual_calibration"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def partition(rows):
    """Alternate sorted identity hashes within each family, never frames/views."""
    families = defaultdict(set)
    for row in rows:
        if row["split"] not in ("calibration", "validation"):
            raise ValueError("Only reference calibration/validation identities are allowed")
        families[row["category"]].add(row["assembly_id"])
    roles = {}
    for family, identities in sorted(families.items()):
        ordered = sorted(identities, key=lambda key: hashlib.sha256(("visual-calibration-42:" + key).encode()).hexdigest())
        if len(ordered) < 4:
            raise ValueError("Insufficient distinct identities for calibration and verification")
        roles.update({identity: "calibration" if i % 2 == 0 else "verification"
                      for i, identity in enumerate(ordered)})
    return [dict(row, calibration_role=roles[row["assembly_id"]]) for row in rows]


def prepare():
    planar_path = ROOT / "output/keycld_corpus/dataset_manifest.json"
    spatial_path = ROOT / "output/model_expansion/spatial/dataset_manifest.json"
    planar = json.loads(planar_path.read_text())["records"]
    spatial = json.loads(spatial_path.read_text())["videos"]
    planar_rows = partition([r for r in planar if r["split"] == "calibration"])
    spatial_rows = partition([r for r in spatial if r["split"] == "validation" and r["view"] == "front"])
    selected = {r["assembly_id"] for r in planar_rows + spatial_rows}
    excluded = {r["assembly_id"] for r in planar + spatial if r["split"] in ("train", "test")}
    if selected & excluded:
        raise ValueError("Calibration identities overlap model training/test identities")
    readers = []
    for model, arm, base in (("keycld", "video_euler", "output/keycld_corpus"),
                              ("con", "video_con_iae", "output/con_corpus")):
        for seed in (42, 43, 44):
            folder = ROOT / base / "runs" / (arm + "_seed" + str(seed))
            training = json.loads((folder / "training_report.json").read_text())
            if sha(folder / "params.p") != training["checkpoint_sha256"]:
                raise ValueError("Frozen planar checkpoint changed")
            mapping = "scores.json" if model == "keycld" else "readout.json"
            readers.append(dict(id=model + "_64_seed" + str(seed), kind="planar", model=model,
                seed=seed, native_resolution=64, encoder_resolution=64, folder=str(folder.relative_to(ROOT)),
                checkpoint_sha256=training["checkpoint_sha256"], mapping_sha256=sha(folder / mapping),
                role="model_owned_coordinate_readout_diagnostic",
                normalization_of_own_model_allowed=False))
    spatial_root = ROOT / "output/model_expansion/spatial"
    for width in (384, 256):
        readers.append(dict(id="shared_spatial_" + str(width), kind="spatial", model="V-JEPA 2.1-B/16 observer",
            native_resolution=width, encoder_resolution=384, seed=42,
            encoder_sha256=sha(ROOT / "output/model_expansion/weights/vjepa2_1_vitb_dist_vitG_384.pt"),
            readout_sha256={family: sha(spatial_root / (family + "_readout.pt")) for family in ("H-A", "H-B")},
            role="external_to_LingBot; model_owned_in_V-JEPA",
            normalization_of_own_model_allowed=False))
    protocol = dict(schema="visual-readout-calibration/1", seed=42,
        purpose="held-out correct reference RGB through unchanged readers; no dynamics rollout or fitting",
        split_rule="alternate sorted SHA256(identity) within family; retain all conditions together",
        quantile=.95, quantile_rule="higher empirical quantile; not a guaranteed 95% confidence bound",
        planar_frames=list(range(90)), spatial_frames=list(range(120)),
        temporal_statistics="future/interior as declared for each measurement; no independent-frame confidence claims",
        planar_derivatives=dict(fps=30, window=7, degree=3, position_filter=False, edge=3),
        spatial_derivatives="unchanged evaluate_spatial.Audit.derivatives; exact same 30Hz SG pipeline",
        engineering_thresholds_changed=False, test_data_used=False, calibrate_on_model_forecasts=False,
        uncertainty_interpretation="total readout error: systematic bias and centered variation reported separately",
        downstream_tracks=["raw_readout", "declared_kinematic_intervention"],
        source_manifests={str(planar_path.relative_to(ROOT)): sha(planar_path),
                          str(spatial_path.relative_to(ROOT)): sha(spatial_path)},
        readers=readers, planar_records=planar_rows, spatial_records=spatial_rows)
    destination = OUT / "protocol.json"
    if destination.exists() and json.loads(destination.read_text()) != protocol:
        raise ValueError("Existing calibration protocol differs; do not mix results")
    write(destination, protocol)
    return protocol


def load_protocol():
    protocol = json.loads((OUT / "protocol.json").read_text())
    for path, expected in protocol["source_manifests"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Reference manifest changed")
    return protocol


def readout_path(reader, row):
    return OUT / "readouts" / reader["id"] / (row.get("case_id", row.get("video_id")) + ".npz")


def error_statistics(error):
    """Bias/spread of vector errors; variation is not claimed to be pure noise."""
    values = np.asarray(error, dtype=float).reshape(len(error), -1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite coordinate errors require an explicit failure record")
    bias = values.mean(0)
    centered = values - bias
    points = np.asarray(error).reshape(len(error), -1, np.asarray(error).shape[-1])
    return dict(coordinate_bias=bias.tolist(), coordinate_bias_rms=float(np.sqrt(np.mean(bias**2))),
        coordinate_rmse=float(np.sqrt(np.mean(values**2))),
        centered_coordinate_rmse=float(np.sqrt(np.mean(centered**2))),
        point_euclidean_rmse=float(np.sqrt(np.mean(np.sum(points**2, axis=-1)))),
        point_error_p95=float(np.quantile(np.linalg.norm(points, axis=-1), .95, method="higher")),
        covariance=(centered.T @ centered / len(values)).tolist(),
        lag1_cross_covariance=(centered[1:].T @ centered[:-1] / (len(values) - 1)).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        protocol = prepare()
        print(json.dumps({"readers": len(protocol["readers"]), "planar_reference_videos": len(protocol["planar_records"]),
                          "spatial_reference_videos": len(protocol["spatial_records"])}))
