"""Archive existing KeyCLD evidence under the common contract without rescoring."""
import argparse
import hashlib
import json
from pathlib import Path

from . import PROTOCOL_VERSION
from .evaluate import write_json

ROOT = Path(__file__).resolve().parents[2]


def import_evidence(source):
    names = ("evaluation_report.json", "field_probes.json", "identifiability_controls.json",
             "dataset_manifest.json", "execution_manifest.json")
    data = {name: json.loads((source / name).read_text()) for name in names}
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}
    report = data["evaluation_report.json"]
    cards = []
    for model, input_track, reader in (
        ("video", "video_primary", "legacy_unvalidated_latent_keypoints"),
        ("state_RK4", "state_diagnostic", "legacy_physical_state_component"),
    ):
        conditions = []
        for name, models in report["conditions"].items():
            result = models[model]
            if len(result["rows"]) != result["run_count"]:
                raise ValueError("Inconsistent source row count")
            conditions.append({
                "condition": name,
                "mechanism_family": "3R1P" if name == "crank_slider" else "4R",
                "expected_trajectories": result["run_count"],
                "finite_trajectories": result["finite_runs"],
                "legacy_geometrically_complete_trajectories": result["complete_physical_runs"],
                "legacy_geometry_gate_m": report["protocol"]["geometry_gate_m"],
                "motor_torque_error_rms_Nm": result["motor_torque_error_rms_Nm"],
                "geometry_residual_rms_m": result["geometry_residual_rms_m"],
                "source_trajectory_rows": result["rows"],
            })
        cards.append({
            "protocol_version": PROTOCOL_VERSION,
            "suite_id": "legacy_keycld_extended_diagnostics",
            "test_manifest_sha256": hashes["dataset_manifest.json"],
            "model": "KeyCLD_" + model,
            "checkpoint_sha256": report["checkpoint_sha256"][model],
            "input_track": input_track,
            "training_track": "source_trained_frozen",
            "information_policy_id": "legacy_three_frame_known_geometry_constant_action",
            "budget_policy_id": "legacy_450_train_50_validation_40_epochs_seed42",
            "reader_id": reader,
            "derivative_policy_id": "legacy_native_model_ODE_derivatives",
            "sampling_policy_id": "deterministic_one_forecast_per_case",
            "calibration_id": None,
            "score_status": "legacy_diagnostic_not_formal_leaderboard",
            "verified_score_100": None,
            "new_training_performed": False,
            "conditions": conditions,
            "missing_common_core_evidence": [
                "Common calibrated video/state reader and derivative path",
                "Velocity and acceleration constraint metrics on that path",
                "Work-energy metric on that path",
                "Complete aligned core predictive-error measurements",
                "Independent calibration thresholds and preregistered normalized scales",
            ],
            "interpretation": (
                "Uninterpretable latent geometry blocks this measurement route; this is not a video-level physical failure score."
                if model == "video" else
                "Exact-state component diagnostic, not end-to-end video learning performance."
            ),
        })
    return {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "Preserve old scientific evidence and distinguish it from new common scoring",
        "source_directory": str(source.resolve()), "source_sha256": hashes,
        "scorecards": cards,
        "state_component_diagnostic_probes": data["field_probes.json"],
        "exact_state_parameter_positive_control": data["identifiability_controls.json"],
        "parameter_control_is_model_score": False,
        "source_execution_manifest": data["execution_manifest.json"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "output/keycld_fourbar/extended_dynamics")
    parser.add_argument("--output", type=Path, default=ROOT / "output/closed_chain_eval/keycld_evidence.json")
    args = parser.parse_args()
    evidence = import_evidence(args.source)
    write_json(args.output, evidence)
    print(json.dumps({"output": str(args.output), "model_cards": len(evidence["scorecards"]),
                      "formal_scores": None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
