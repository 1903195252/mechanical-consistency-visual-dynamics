"""Re-score saved planar/spatial measurements without running or changing models.

Run from the project root:
    python3 -m experiments.closed_chain_eval.rescore_continuous

Only engineering-normalized continuous scores are applied to current model
results. Independent readout calibration and optimization are separate evidence;
their errors are not subtracted from model RMSE or used to relax physical limits.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import CORE_METRICS, MetricSpec, score_trajectory
from .continuous import (aggregate_continuous, calibrate_noise_profile,
                         continuous_scorecard, pixel_span, SCORING_VERSION)


PLANAR_FIELDS = dict(zip(CORE_METRICS, (
    "geometry_m", "velocity_m_s", "acceleration_m_s2", "force_balance_Nm",
    "work_energy_J", "prediction_m")))
UNITS = dict(zip(CORE_METRICS, ("m", "m/s", "m/s^2", "N m", "J", "m")))


def make_card(values, limits, units, calibration_id, case_key):
    specs = {k: MetricSpec([units[k]], [limits[k]], limit=1.) for k in CORE_METRICS}
    errors = {k: None if values.get(k) is None else [values[k]] for k in CORE_METRICS}
    # Each input is an already reduced trajectory metric. Do not re-average its
    # time/channel statistic or pretend that it is one original video frame.
    card = score_trajectory(errors, specs, dict.fromkeys(CORE_METRICS, 1), calibration_id)
    card["case_key"] = case_key
    return continuous_scorecard(card)


def spatial_values(measured):
    fields = {"geometry": ("geometry_max_mm", .001),
              "constraint_velocity": ("velocity_constraint_max_m_s", 1.),
              "constraint_acceleration": ("acceleration_constraint_max_m_s2", 1.),
              "dynamics": ("effort_rmse", 1.), "energy": ("work_energy_max_J", 1.),
              "prediction": ("point_rmse_mm", .001)}
    return {name: None if measured.get(field) is None else float(measured[field]) * factor
            for name, (field, factor) in fields.items()}


def run(root, destination):
    sources, run_summaries, flat_rows, profiles = {}, [], [], []

    def read(relative):
        path = root / relative
        raw = path.read_bytes()
        sources[str(relative)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    planar_protocol = read(Path("output/keycld_corpus/protocol.json"))
    limits = {k: planar_protocol["evaluation"]["limits_SI"][v] for k, v in PLANAR_FIELDS.items()}
    manifest = read(Path("output/keycld_corpus/dataset_manifest.json"))
    expected = [(r["category"], r["condition"], r["case_id"])
                for r in manifest["records"] if r["split"] == "test"]
    calibration = read(Path("output/keycld_corpus/calibration.json"))
    if not calibration["passed"]:
        raise ValueError("The source planar calibration has not passed")
    calibration_id = sources["output/keycld_corpus/calibration.json"]
    refs = [make_card({k: r["controls"]["oracle"]["metrics"][v] for k, v in PLANAR_FIELDS.items()},
                      limits, UNITS, calibration_id, (r["category"], r["condition"], r["case_id"]))
            for r in calibration["rows"]]
    profiles.append(calibrate_noise_profile(refs, profile_id="planar-reference-numerics",
        measurement_id="reference-native-derivatives-240Hz", source_id=calibration_id,
        split="calibration"))

    def record(cards, expected_keys, identity, calibration_key):
        result = aggregate_continuous(cards, expected_keys, calibration_id=calibration_key)
        result.pop("strata")
        run_summaries.append(dict(identity, **result))
        for card in cards:
            branch = card["continuous"]
            row = dict(identity, family=card["case_key"][0], condition=card["case_key"][1],
                       case_id=card["case_key"][2], complete=card["all_metrics_available"],
                       strict_joint_pass=card["joint_pass"], continuous_score_100=branch["score_100"])
            for metric in CORE_METRICS:
                item = card["metrics"][metric]
                raw = item["raw_per_channel"]
                row[metric + "_SI"] = max(raw) if raw is not None else None
                row[metric + "_limit_SI"] = item["scales"][0] * item["limit"]
                row[metric + "_ratio"] = branch["metrics"][metric]["ratio"]
                row[metric + "_score_100"] = branch["metrics"][metric]["score_100"]
            flat_rows.append(row)

    torque = read(Path("output/keycld_fourbar/scored_evaluation/scores.json"))
    for arm, saved in torque["scorecards"].items():
        cards = [continuous_scorecard(r) for r in saved["rows"]]
        record(cards, [r["case_key"] for r in cards],
               dict(suite="torque-excitation", model="KeyCLD", arm=arm, track="native",
                    subset="all-conditions", seed=None), saved["calibration_id"])

    directory = root / "output/con_corpus/dynamics_extension/physics_readout"
    reports = sorted(directory.glob("*/report.json"))
    if not reports:
        raise FileNotFoundError("Saved planar readout reports are required")
    for path in reports:
        report = read(path.relative_to(root))
        origin = report["identity"]
        for track in ("raw", "conditioned"):
            cards = []
            for row in report["rows"]:
                values = {k: row[track]["metrics"].get(v) for k, v in PLANAR_FIELDS.items()}
                cards.append(make_card(values, limits, UNITS, calibration_id,
                                       (row["category"], row["condition"], row["case_id"])))
            identity = dict(suite="planar-servo", model="KeyCLD" if origin["model"] == "keycld" else "CON",
                            arm=origin["arm"], track=track, subset="all-planar", seed=origin["seed"])
            record(cards, expected, identity, calibration_id)

    spatial_manifest = read(Path("output/vjepa_expansion/dataset_manifest.json"))
    spatial_roots = [("V-JEPA", "AC", seed, Path("output/vjepa_expansion") / ("seed_%d" % seed) / "calibrated")
                     for seed in (42, 43, 44)]
    spatial_roots += [("LingBot", arm, seed, Path("output/lingbot_expansion/runs") /
                      ("seed%d" % seed) / ("step%04d" % step))
                     for arm, step in (("zero_shot", 0), ("adapted", 50)) for seed in (42, 43, 44)]
    for category in ("H-A", "H-B"):
        calibration_path = Path("output/model_expansion/spatial") / (category + "_calibrate.json")
        reference_rows = read(calibration_path)
        if not all(r["reference_30Hz"]["physical_pass"] for r in reference_rows):
            raise ValueError("Spatial reference calibration failed")
        spatial_calibration = sources[str(calibration_path)]

        def spatial_card(measured, row, fallback=None):
            if "engineering_limits" in row:
                local_limits = row["engineering_limits"]
                units = dict(UNITS, dynamics=row["effort_unit"])
            else:
                scale_source = measured if "effort_tolerance" in measured else fallback
                local_limits = dict(limits, dynamics=scale_source["effort_tolerance"],
                                    energy=scale_source["energy_tolerance_J"])
                units = dict(UNITS, dynamics=scale_source["effort_unit"])
            return make_card(spatial_values(measured), local_limits, units, spatial_calibration,
                             (category, row["condition"], row["video_id"]))

        refs = [spatial_card(r["reference_30Hz"], r) for r in reference_rows]
        profiles.append(calibrate_noise_profile(refs, profile_id=category + "-reference-numerics",
            measurement_id=category + "-exact-states-discrete-30Hz", source_id=spatial_calibration,
            split="validation"))
        expected_spatial = [(category, r["condition"], r["video_id"]) for r in spatial_manifest["videos"]
                            if r["split"] == "test" and r["view"] == "front" and r["category"] == category]
        for model, arm, seed, folder in spatial_roots:
            if model == "V-JEPA":
                rows = read(folder / (category + "_score.json"))
            else:
                rows = [read(p.relative_to(root)) for p in sorted((root/folder/'cases').glob(category+'*.json'))]
            if len(rows) != 150:
                raise ValueError("Spatial comparison requires 150 trajectories per family and seed")
            for track in ("native", "readout_persistence", "kinematic_intervention"):
                cards = [spatial_card(r.get(track, {}), r, r.get("native", {})) for r in rows]
                identity = dict(suite="spatial-position", model=model, arm=arm,
                                track=track, subset=category, seed=seed)
                record(cards, expected_spatial, identity, spatial_calibration)

    grouped = defaultdict(list)
    for row in run_summaries:
        grouped[tuple(row[k] for k in ("suite", "model", "arm", "track", "subset"))].append(row)
    summaries = []
    for identity, runs in sorted(grouped.items()):
        if identity[0] in ("planar-servo", "spatial-position") and sorted(r["seed"] for r in runs) != [42, 43, 44]:
            raise ValueError("Corpus comparisons require all three registered seeds")
        item = dict(zip(("suite", "model", "arm", "track", "subset"), identity))
        item.update(seeds=[r["seed"] for r in runs], expected_cases=sum(r["expected_cases"] for r in runs),
                    complete_cases=sum(r["complete_cases"] for r in runs),
                    strict_joint_100=float(np.mean([r["verified_score_100"] for r in runs])),
                    continuous_score_100=float(np.mean([r["continuous"]["score_100"] for r in runs])),
                    components_100={k: float(np.mean([r["continuous"]["components_100"][k]
                                                     for r in runs])) for k in CORE_METRICS})
        summaries.append(item)
    visual_report = root / "output/closed_chain_eval/visual_calibration/summary.json"
    visual_evidence = None
    if visual_report.exists():
        visual = json.loads(visual_report.read_text())
        visual_evidence = dict(report=str(visual_report.relative_to(root)),
            sha256=hashlib.sha256(visual_report.read_bytes()).hexdigest(),
            reference_measurements=visual["reference_measurements"],
            score_application="optimize and remeasure the matching pipeline; never subtract two RMSE values")
    result = {"purpose": "canonical continuous SI evaluation of registered measurements",
              "scoring_version": SCORING_VERSION,
              "rule": "100 / (1 + (error / engineering_limit)^2)",
              "visual_measurement_evidence": visual_evidence,
              "calibration_application_status": "One continuous Scomp; calibration diagnoses instrument error without changing scores",
              "sources": sources, "metric_rows": len(flat_rows), "summaries": summaries,
              "reference_numerical_profiles": profiles}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with (destination / "case_scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    lines = ["# 统一连续 SI 评分", "",
             "使用既有预测与测量，仅更换分数映射。连续分 = 六项 100 / (1 + 容差比²) 的平衡平均；"
             "容差比为 1 时得 50 分。Scomp 是唯一主综合分；严格联合率是辅助指标。独立标定记录测量误差，不另设观测归一化分。", "",
             "原始 SI 误差、严格联合合格率和物理可测资格保持原规则。N/A 分项贡献 0 分并保留六项分母。"
             "均值先按机构族、工况、轨迹平衡，再平均种子。各协议的时间和通道汇总方式保留原定义，分开列示。", "",
             "| 协议/子集 | 模型 | 分支 | 轨道 | 完整可测/总数 | Scomp | 严格联合率 (%) |",
             "|---|---|---|---|---:|---:|---:|"]
    for row in summaries:
        lines.append("| {suite}/{subset} | {model} | {arm} | {track} | {complete_cases}/{expected_cases} | "
                     "{continuous_score_100:.2f} | {strict_joint_100:.2f} |".format(**row))
    lines += ["", "## 分辨率示例", "",
              "下表只复核一像素对应的物理跨度，不把它直接视为闭合残差的实测不确定度。", "",
              "| 图像宽度 | 全宽 0.86 m，边界跨度 (mm/px) | 全宽 1.30 m，边界跨度 (mm/px) | "
              "全宽 1.30 m，首末像素中心跨度 (mm/px) |", "|---:|---:|---:|---:|"]
    for width in (64, 256, 512):
        values = [pixel_span(s, width, convention=c) * 1000 for s, c in
                  ((.86, "pixel_edges"), (1.30, "pixel_edges"), (1.30, "pixel_centers"))]
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} |".format(width, *values))
    lines += ["", "共享平面语料当前采用首末像素中心约定：prepare.py 中的换算为 1.30 / 63。", "",
              "## 标定状态", "",
              "已提取平面 54 条独立校准参考、Stewart/Delta 各 12 条验证参考的数值误差分位数。"
              "这些是精确状态/求导管线的误差，不能代替图像定位和物理读出标定。", "",
              "完整视觉链路的[独立标定与优化](../visual_calibration/results.md)另行报告。"
              "参考视频与模型视频必须经过相同的冻结读取器、分辨率和求导配置；原生 ODE 导数成绩不能套用图像求导的误差尺度。"
              "测量误差会触发读取器／求导优化和新版重评分，不作为模型免责额度；工程门槛不变，也不相减两个 RMSE。", "",
              "本表统一重算所有既有测量的连续 Scomp。KeyCLD、CON 与 V-JEPA 的状态／潜空间读出结果描述各自完整预测管线；"
              "LingBot 的物理量来自外部图像读取器，需要结合上述标定与新版测量判断归因。"
              "公共读取器与原生状态是测量路径的消融设置，共享同一个评分定义。", "",
              "平面采用 64／256 像素，空间模型共享 50 + 50 个测试机构、三个工况和三个种子；所有注册案例保留在评分分母。", ""]
    (destination / "results.md").write_text("\n".join(lines))
    return {"metric_rows": len(flat_rows), "summary_rows": len(summaries),
            "report": str(destination / "results.md")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("output/closed_chain_eval/continuous"))
    args = parser.parse_args()
    root = args.root.resolve()
    destination = args.output if args.output.is_absolute() else root / args.output
    print(json.dumps(run(root, destination), ensure_ascii=False))
