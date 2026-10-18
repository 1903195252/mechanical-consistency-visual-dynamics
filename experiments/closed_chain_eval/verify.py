"""Integration audit against stored SI truth; this is not a model result."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np

from .core import equation_residuals
from .evaluate import write_json
from .import_keycld import ROOT, import_evidence
from .planar import evaluate_planar, physics, physics_frame


def main():
    source = ROOT / "output/keycld_fourbar/extended_dynamics"
    configs = json.loads((source / "condition_configs.json").read_text())
    conditions = {}
    for name in configs:
        with np.load(source / "truth" / name / "run_1000.npz", allow_pickle=False) as data:
            truth = {key: data[key] for key in data.files}
        with np.load(source / "observations" / name / "run_1000.npz", allow_pickle=False) as data:
            t = data["t"].astype(float)
            tau = np.full(len(t), float(data["action"][0]))
        config_for_run = json.loads((source / "truth" / name / "run_1000.json").read_text())
        exact, old = [], []
        for q, v, a, command in zip(truth["q"], truth["qdot"], truth["qddot"], tau):
            residuals = equation_residuals(physics_frame(q, v, command, config_for_run), v, a)
            exact.append(float(residuals["dynamics"][0]))
            terms = physics.terms(q, v, a, command, config_for_run)
            old.append(terms["inertia_Nm"] + terms["velocity_quadratic_Nm"] + terms["gravity_Nm"]
                       + terms["friction_Nm"] - terms["external_Nm"] - terms["motor_Nm"])
        np.testing.assert_allclose(exact, old, atol=1e-12, rtol=1e-10)
        report = evaluate_planar(t, truth["q"], truth["q"], tau, config_for_run)
        assert report["all_metrics_available"] and report["joint_pass"] is None
        assert report["metrics"]["prediction"]["value"] == 0.
        static = np.repeat(truth["q"][:1], len(t), axis=0)
        static_report = evaluate_planar(t, static, truth["q"], tau, config_for_run)
        assert static_report["metrics"]["prediction"]["value"] > 0.
        conditions[name] = {
            "case": 1000, "samples": len(t),
            "exact_derivative_equation_max_abs_Nm": float(np.max(np.abs(exact))),
            "new_vs_previous_equation_max_abs_Nm": float(np.max(np.abs(np.asarray(exact) - old))),
            "correct_sampled_trajectory_common_derivative_diagnostics": report["metrics"],
            "static_prediction_error_m": static_report["metrics"]["prediction"]["raw_per_channel"],
            "formal_pass_not_assigned": report["joint_pass"] is None,
        }
    imported = import_evidence(source)
    assert len(imported["scorecards"]) == 2
    assert all(card["verified_score_100"] is None for card in imported["scorecards"])
    counts = [sum(row["expected_trajectories"] for row in card["conditions"]) for card in imported["scorecards"]]
    assert counts == [160, 160]
    # Exercise the actual file/CLI boundary once; fixtures are owned temporary
    # files under the workspace, never additions to the original experiment.
    outdir = ROOT / "output/closed_chain_eval"
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cli_check_", dir=str(outdir)) as temporary:
        temporary = Path(temporary)
        np.savez(temporary / "prediction.npz", t=t, q=truth["q"])
        np.savez(temporary / "reference.npz", t=t, q=truth["q"], motor_tau=tau)
        command = [sys.executable, "-m", "experiments.closed_chain_eval.evaluate",
                   "--prediction", str(temporary / "prediction.npz"),
                   "--reference", str(temporary / "reference.npz"),
                   "--config", str(source / "truth" / name / "run_1000.json"),
                   "--output", str(temporary / "score.json")]
        subprocess.run(command, cwd=str(ROOT), check=True, capture_output=True, text=True)
        cli = json.loads((temporary / "score.json").read_text())
        assert cli["score_status"] == "calibration_pending" and cli["joint_pass"] is None
        assert cli["all_metrics_available"]
    directory = Path(__file__).resolve().parent
    output = outdir / "verification.json"
    write_json(output, {
        "purpose": "Software/equation integration verification, not new model performance",
        "model_inference_or_training_performed": False,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(directory.glob("*.py")) + [directory / "protocol.json"]},
        "physics_dependency_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in (ROOT / "experiments/fourbar/mechanics.py",
                                                   ROOT / "experiments/keycld_fourbar/extended_mechanics.py")},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "cli_check": {"passed": True, "input_track": "exact_state_software_control", "formal_score_assigned": False},
        "conditions": conditions,
        "imported_trajectory_counts": counts,
        "conclusion": "Independent equation agrees with the previous implementation; common derivative errors require calibration; no retrospective total score assigned.",
    })
    print(json.dumps({"output": str(output), "conditions_verified": len(conditions),
                      "maximum_equation_implementation_difference_Nm": max(row["new_vs_previous_equation_max_abs_Nm"] for row in conditions.values())}))


if __name__ == "__main__":
    main()
