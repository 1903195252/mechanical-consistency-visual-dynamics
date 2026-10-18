"""CLI for physical-coordinate predictions; model inference is a separate stage."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from . import PROTOCOL_VERSION
from .core import MetricSpec
from .continuous import continuous_scorecard
from .planar import evaluate_planar


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True, help="npz: t, q (T,3)")
    parser.add_argument("--reference", type=Path, required=True, help="evaluator-only npz: t, q, motor_tau (T,)")
    parser.add_argument("--config", type=Path, required=True, help="audited 4R/3R1P SI configuration")
    parser.add_argument("--profile", type=Path, help="registered calibration_id and six MetricSpec entries")
    parser.add_argument("--prefix-frames", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text()) if args.profile else {}
    specs = {name: MetricSpec(**value) for name, value in profile["metrics"].items()} if profile else None
    with np.load(args.prediction, allow_pickle=False) as predicted, np.load(args.reference, allow_pickle=False) as reference:
        if not np.array_equal(predicted["t"], reference["t"]):
            raise ValueError("Prediction timestamps must equal the registered reference grid")
        report = evaluate_planar(predicted["t"], predicted["q"], reference["q"], reference["motor_tau"],
                                 json.loads(args.config.read_text()), args.prefix_frames, specs,
                                 profile.get("calibration_id"))
    if report.get("score_status") == "calibrated":
        report = continuous_scorecard(report)
    paths = {"prediction": args.prediction, "reference": args.reference, "mechanism": args.config}
    if args.profile:
        paths["profile"] = args.profile
    report.update(protocol_version=PROTOCOL_VERSION,
                  input_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                      Path(__file__), Path(__file__).with_name("core.py"), Path(__file__).with_name("planar.py"),
                      Path(__file__).parents[1] / "fourbar/mechanics.py",
                      Path(__file__).parents[1] / "keycld_fourbar/extended_mechanics.py")},
                  environment={"python": platform.python_version(), "numpy": np.__version__},
                  evidence_type="single_trajectory_measurement_not_leaderboard")
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "score_status": report["score_status"],
                      "joint_pass": report["joint_pass"],
                      "Scomp": report.get("continuous", {}).get("score_100")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
