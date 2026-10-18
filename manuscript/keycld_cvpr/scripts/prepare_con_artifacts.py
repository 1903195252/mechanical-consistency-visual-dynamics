"""Bind the two-model manuscript tables to the completed CON reports."""
from collections import defaultdict
import ast
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
CON = ROOT / "output/con_corpus"
KEYCLD = ROOT / "output/keycld_corpus"
METRICS = (
    "geometry_m", "velocity_m_s", "acceleration_m_s2",
    "force_balance_Nm", "work_energy_J", "prediction_m",
)
ARMS = {"video_con_iae": "Video + affine", "state_con_iae": "State"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def macro(rows, value):
    groups = defaultdict(list)
    for row in rows:
        number = value(row)
        if number is not None:
            assert math.isfinite(number)
            groups[row["category"], row["condition"]].append(number)
    return fmean(fmean(group) for group in groups.values()) if groups else None


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-13), (actual, expected)


def collect():
    sources = {}

    def source(path):
        sources[str(path.relative_to(ROOT))] = digest(path)
        return read(path)

    manifest = source(CON / "dataset_manifest.json")
    assert digest(CON / "dataset_manifest.json") == digest(KEYCLD / "dataset_manifest.json")
    protocol = source(CON / "protocol.json")
    key_protocol = source(KEYCLD / "protocol.json")
    calibration = source(CON / "calibration.json")
    assert digest(CON / "calibration.json") == digest(KEYCLD / "calibration.json")
    assert calibration["passed"]
    assert protocol["evaluation"]["limits_SI"] == key_protocol["evaluation"]["limits_SI"]
    evaluator = source(CON / "evaluator/manifest.json")
    for path, expected in evaluator["source_hashes"].items():
        if digest(ROOT / path) != expected:
            # Scoring aggregation has been migrated; the measured SI residuals
            # still use the exact original functions retained in the archive.
            archived = ROOT / "output/closed_chain_eval/archive/pre_unified_scoring/evaluator_sources" / path
            assert digest(archived) == expected, path
            original = ast.parse(archived.read_text())
            current = {n.name: n for n in ast.parse((ROOT / path).read_text()).body
                       if isinstance(n, ast.FunctionDef)}
            for node in original.body:
                if isinstance(node, ast.FunctionDef) and node.name not in ("score", "main"):
                    assert ast.dump(node) == ast.dump(current[node.name]), (path, node.name)
    status = read(CON / "status.json")
    expected_cases = {r["case_id"] for r in manifest["records"] if r["split"] == "test"}
    assert len(expected_cases) == 108
    assert {r["case_id"] for r in evaluator["references"]} == expected_cases
    runs, summaries = [], {}
    for arm in ARMS:
        all_rows = []
        for seed in (42, 43, 44):
            folder = CON / "runs" / f"{arm}_seed{seed}"
            trained = source(folder / "training_report.json")
            inferred = source(folder / "inference_report.json")
            scored = source(folder / "scores.json")
            assert trained["arm"] == scored["arm"] == inferred["arm"] == arm
            assert trained["seed"] == scored["seed"] == inferred["seed"] == seed
            assert trained["epochs"] == 50 and trained["updates"] == 2550
            assert trained["samples_seen"] == 40650
            assert trained["identity"]["data"] == digest(CON / "dataset_manifest.json")
            assert trained["identity"]["protocol"] == scored["protocol_sha256"] == digest(CON / "protocol.json")
            assert trained["checkpoint_sha256"] == inferred["checkpoint_sha256"] == scored["checkpoint_sha256"] == digest(folder / 'params.p')
            predictions=list((folder/'predictions').glob('*/*.npz'))
            assert len(predictions)==162
            for prediction in predictions:
                meta=read(prediction.with_suffix('.json'))
                assert digest(prediction)==meta['prediction_sha256']
                assert meta['checkpoint_sha256']==trained['checkpoint_sha256']
            assert scored["calibration_sha256"] == digest(CON / "calibration.json")
            assert scored["evaluator_manifest_sha256"] == digest(CON / "evaluator/manifest.json")
            assert inferred["completed"] == inferred["finite"] == 162
            assert inferred["test_cases"] == 108 and inferred["validation_cases"] == 54
            rows = scored["rows"]
            assert len(rows) == 108 and {r["case_id"] for r in rows} == expected_cases
            key_scores = source(KEYCLD / "runs" / f"video_euler_seed{seed}" / "scores.json")
            assert {r["case_id"] for r in key_scores["rows"]} == expected_cases
            for metric, limit in protocol["evaluation"]["limits_SI"].items():
                for row in rows:
                    value = row["metrics"][metric]
                    assert row["passed"][metric] == (value is not None and value <= limit)
                close(100 * macro(rows, lambda r: r["passed"][metric]), scored["component_scores"][metric])
            assert all(r["future_samples"] == 689 for r in rows)
            assert all(r["metrics"][k] is None for r in rows for k in ("force_balance_Nm", "work_energy_J"))
            assert all(not r["passed"]["geometry_m"] and not r["joint_pass"] for r in rows)
            assert scored["complete_physical_trajectories"] == 0
            close(fmean(scored["component_scores"].values()), scored["composite_score_100"])
            reader = source(folder / "readout.json") if arm == "video_con_iae" else None
            if reader:
                assert reader["fit_partitions"] == ["train"] and not reader["test_labels_used"]
                assert not reader["predictor_uses_reader"]
                assert reader["checkpoint_sha256"] == trained["checkpoint_sha256"]
                assert trained["full_official_video_objective"]
                assert all(r["rgb_mse"] is not None for r in rows)
            runs.append({
                "arm": arm, "seed": seed, "checkpoint_sha256": trained["checkpoint_sha256"],
                "epochs": trained["epochs"], "updates": trained["updates"],
                "samples_seen": trained["samples_seen"], "parameters": trained["parameters"],
                "training_elapsed_s": trained["elapsed_s"], "devices": trained["devices"],
                "test_cases": 108, "validation_cases": 54, "finite_predictions": 162,
                "complete_physical_trajectories": scored["complete_physical_trajectories"],
                "component_scores": scored["component_scores"],
                "composite_score_100": scored["composite_score_100"],
                "point_rms_m": macro(rows, lambda r: r["metrics"]["prediction_m"]),
                "geometry_max_m": macro(rows, lambda r: r["metrics"]["geometry_m"]),
                "rgb_mse": macro(rows, lambda r: r.get("rgb_mse")),
                "rgb_persistence_mse": macro(rows, lambda r: r.get("rgb_persistence_mse")),
                "reader_validation_rms_m": reader["validation_mean_point_rms_m"] if reader else None,
            })
            all_rows.extend(rows)
        stored = status["summaries"][arm]
        summary = {
            "test_predictions": len(all_rows), "complete_physical_trajectories": 0,
            "joint_pass_rate_100": 0,
            "component_scores": {k: 100 * macro(all_rows, lambda r: r["passed"][k]) for k in METRICS},
            "point_rms_m": macro(all_rows, lambda r: r["metrics"]["prediction_m"]),
            "geometry_max_m": macro(all_rows, lambda r: r["metrics"]["geometry_m"]),
            "rgb_mse": macro(all_rows, lambda r: r.get("rgb_mse")),
            "rgb_persistence_mse": macro(all_rows, lambda r: r.get("rgb_persistence_mse")),
            "composite_95pct": stored["intervals"]["composite_95pct"],
        }
        summary["composite_score_100"] = fmean(summary["component_scores"].values())
        close(summary["composite_score_100"], stored["composite_score_100"])
        close(summary["point_rms_m"], stored["raw_point_error_m"]["mean"])
        close(summary["geometry_max_m"], stored["geometry_max_error_m"]["mean"])
        if summary["rgb_mse"] is not None:
            close(summary["rgb_mse"], stored["rgb_future_mse"]["mean"])
            close(summary["rgb_persistence_mse"], stored["rgb_persistence_mse"]["mean"])
            summary["rgb_reduction_percent"] = 100 * (1 - summary["rgb_mse"] / summary["rgb_persistence_mse"])
        summaries[arm] = summary
    return {
        "dataset_identical_to_keycld": True, "calibration_identical_to_keycld": True,
        "thresholds_identical_to_keycld": True, "test_cases_matched_by_id": 108,
        "training_runs": 6, "reported_prediction_archives": 972,
        "reported_test_predictions": 648, "reported_validation_predictions": 324,
        "evidence_basis": "Completed training, inference, scoring, and local SHA-256 verification of all six weights and 972 raw prediction files",
        "raw_archive_recovery_verified": True, "runs": runs, "summary": summaries,
        "interval_source": "output/con_corpus/status.json, paired seed/assembly bootstrap, 2000 draws, seed 20261012",
        "sources": [{"path": p, "sha256": h} for p, h in sorted(sources.items())],
        "literature": {
            "citation_key": "stolzle2024con", "source": "https://arxiv.org/html/2409.08439v2",
            "support": "Sections 2 and 4; Appendix F.1 (equilibrium regime), F.2 (periodic behavior)",
            "interpretation": "Migration test on servo-driven rigid closed chains; equilibrium prior is a possible contributor, not an isolated causal result",
        },
    }


def main():
    from prepare_unified_scoring import planar_estimates
    evidence = collect()
    continuous = planar_estimates()
    score_rows = []
    for arm, label in ARMS.items():
        summary = evidence["summary"][arm]
        components = summary["component_scores"]
        values = ["N/A" if k in ("force_balance_Nm", "work_energy_J") else f"{components[k]:.2f}" for k in METRICS]
        low, high = continuous[arm]["ci95"]
        score_rows.append(" & ".join([label, "0/324", *values, "0", f"{continuous[arm]['mean']:.2f} [{low:.2f}, {high:.2f}]"]) + r" \\")
    (PAPER / "tables/con_score_rows.tex").write_text("\n".join(score_rows) + "\n")
    seed_rows = []
    for run in evidence["runs"]:
        rgb = "N/A" if run["rgb_mse"] is None else f"{run['rgb_mse'] * 1e4:.4f}"
        reader = "N/A" if run["reader_validation_rms_m"] is None else f"{run['reader_validation_rms_m'] * 1000:.2f}"
        seed_rows.append(" & ".join([
            ARMS[run["arm"]], str(run["seed"]), "108/108", "0/108",
            f"{continuous[run['arm']]['by_seed'][str(run['seed'])]:.2f}", f"{run['point_rms_m'] * 1000:.2f}", rgb, reader,
        ]) + r" \\")
    (PAPER / "tables/con_seed_rows.tex").write_text("\n".join(seed_rows) + "\n")
    (PAPER / "sources/con_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return {
        "con_extension": "sources/con_evidence.json: six completed runs, 648 test score rows, shared sample IDs and SI evaluator",
        "con_sources": "output/con_corpus/runs/*/{training_report,inference_report,scores,readout}.json",
        "two_model_comparison": "Same data and independent SI thresholds; model-native budgets and physical readouts are disclosed separately",
    }


if __name__ == "__main__":
    additions = main()
    mapping_path = PAPER / "sources/evidence_map.json"
    mapping = read(mapping_path)
    mapping.update(additions)
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")
    print(json.dumps(additions, indent=2))
