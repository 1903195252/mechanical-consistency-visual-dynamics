"""Render legacy measurements using the sole current continuous SI score."""
import hashlib
import json
import math
from pathlib import Path

from .evaluate import write_json
from .continuous import SCORING_VERSION, continuous_scorecard, aggregate_continuous
from .import_keycld import ROOT


LABELS = {"geometry": "几何约束", "constraint_velocity": "速度约束", "constraint_acceleration": "加速度约束",
          "dynamics": "动力学平衡", "energy": "功—能平衡", "prediction": "条件预测准确性"}
CONDITIONS = {"nominal": "标称无摩擦", "viscous": "黏性摩擦", "coulomb": "库仑摩擦",
              "mixed_friction": "混合摩擦", "heavy_payload": "载荷 0.25 kg", "dense_bars": "密度 ×1.5",
              "geometry_scaled": "尺寸 ×0.8", "crank_slider": "曲柄滑块"}


def main():
    root = ROOT / "output/keycld_fourbar/scored_evaluation"
    report = json.loads((root / "scores.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    video, state = report["scorecards"]["video"], report["scorecards"]["state_RK4"]
    totals, continuous = {}, {}
    for label, card in report["scorecards"].items():
        rows = [continuous_scorecard(row) for row in card["rows"]]
        current = aggregate_continuous(rows, [row["case_key"] for row in rows],
                                       calibration_id=rows[0]["calibration_id"])
        continuous[label] = current["continuous"]
        totals[label] = current["continuous"]["score_100"]
    summary = {
        "kind": "canonical_continuous_si_score",
        "scoring_version": SCORING_VERSION,
        "definition": "Balanced mean of six 100 / (1 + tolerance_ratio ** 2) contributions",
        "metric_weights": {name: 1 / 6 for name in LABELS},
        "added_after_evaluation": True,
        "missing_evidence_policy": "No verified pass; keep all six metrics in denominator and retain coverage",
        "source_scores_sha256": hashlib.sha256((root / "scores.json").read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scorecards": {label: {
            "composite_score_100": totals[label],
            "joint_pass_rate_percent": card["verified_score_100"],
            "component_pass_rates": card["component_scores"],
            "continuous_components_100": continuous[label]["components_100"],
        } for label, card in report["scorecards"].items()},
    }
    write_json(root / "score_summary.json", summary)
    expressions = {label: " + ".join("%g" % continuous[label]["components_100"][key] for key in LABELS)
                   for label, card in report["scorecards"].items()}
    lines = ["# KeyCLD 已执行的动力学诊断评分", "",
             "**综合总分：视频输入 → 当前物理关键点接口 %.2f / 100；真实状态 RK4 组件对照 %.2f / 100。**" %
             (totals["video"], totals["state_RK4"]), "",
             "综合总分 =（几何约束分 + 速度约束分 + 加速度约束分 + 动力学平衡分 + 功—能平衡分 + 条件预测准确性分）÷ 6。每项占 1/6，每项为 100 / (1 + 容差比²)，误差等于容差时得 50 分；工程目标和原始测量保持不变。", "",
             "视频接口：(%s) ÷ 6 = **%.2f 分**。状态组件：(%s) ÷ 6 = **%.2f 分**。" %
             (expressions["video"], totals["video"], expressions["state_RK4"], totals["state_RK4"]), "",
             "当前唯一主综合分为连续容差比 Scomp，旧二分综合分已归档。严格联合率为辅助指标，两组均为 **0%（0/160）**。", "",
             "已完成 96 条独立物理校准轨迹及 160 条全新测试轨迹（每个冻结检查点各 160 次，共 320 次预测）。测试种子 3000–3019，8 个工况配对。没有重新训练、挑选测试样本或调整模型权重。", "",
             "各分项与联合通过率都先在每个工况内求平均，再在机构族内等权平均，最后 4R 与 3R+1P 两个机构族等权。缺失物理测量的项目不得分，分母固定为六项；视频接口动力学与能量的 0 分表示未获得核验通过，原始误差仍为缺失值。", "",
             "| 模型/输入 | 综合总分 /100 | 六项联合通过率 | 全部六项通过的轨迹 | 六项均有完整测量的轨迹 |",
             "|---|---:|---:|---:|---:|"]
    for key, label, card in (("video", "视频输入、声明的 B/C/中点接口", video), ("state_RK4", "真实状态监督与初始化的 RK4 对照", state)):
        complete = sum(row["all_metrics_available"] for row in card["rows"])
        lines.append("| %s | %.2f | %.2f%% | %d/160 | %d/160 |" %
                     (label, totals[key], card["verified_score_100"], card["joint_passed_trajectories"], complete))
    lines += ["", "## 分项成绩", "",
              "下面各分项也按机构族、工况等权计算百分制通过率；同时给出未加权的原始通过数与可测数。不可测的分项不给通过证据，仍保留原因和分母。", "",
              "| 项目 | 视频输入接口 /100 | 视频通过数（可测数） | 状态组件 /100 | 状态通过数（可测数） |",
              "|---|---:|---:|---:|---:|"]
    for key, label in LABELS.items():
        a, b = video["component_scores"][key], state["component_scores"][key]
        lines.append("| %s | %.2f | %d/160（%d） | %.2f | %d/160（%d） |" %
                     (label, a["score_100"], a["passed_trajectories"], a["measured_trajectories"],
                      b["score_100"], b["passed_trajectories"], b["measured_trajectories"]))
    lines += ["", "## 固定的诊断通过线", "",
              "这些是本项目预先登记的工程诊断精度，不是从论文引用的标准，也不是依据本次模型分数反推的阈值。先固定阈值并检验校准控制，再执行新的模型预测。", "",
              "| 项目 | 通过线 | 聚合方式 |", "|---|---:|---|"]
    for key, label in LABELS.items():
        reduction = "全段、全通道最大绝对值" if key in ("geometry", "energy") else "各通道轨迹 RMS 的最大值"
        lines.append("| %s | %g %s | %s |" % (label, protocol["limits_SI"][key], protocol["units"][key], reduction))
    lines += ["", "力矩通过线 0.1 N·m 是固定 ±2 N·m 动作上限幅值的 5%；能量通过线 0.02 J 是 2 N·m × 1 rad 参考功的 1%。这里的参考尺度在观察新测试输出前固定。", "",
              "所有未来点在 0.04–0.50 s 的相同 1600 Hz 输出网格评分。模型仍只看到 100 Hz 的前三张图像及动作；高频输出用于测量连续 ODE，不向模型增加真实未来信息。", "",
              "采用模型原生速度/加速度，并保存密集输出求导的一致性审计。它属于原生 ODE 诊断赛道，不能与仅有视频帧、依靠图像差分求导的模型混排。恒定力矩和恒定世界外力的功用端点位移精确计算，摩擦耗散用密集采样积分。", "",
              "## 校准控制", "", "| 控制 | 六项联合通过 |", "|---|---:|"]
    for key, label in (("oracle", "正确的独立物理轨迹"), ("static", "保持最后观测状态"), ("time_scaled_0.8", "保持几何但运动时间缩放为 0.8")):
        row = report["calibration_controls"][key]
        lines.append("| %s | %d/%d |" % (label, row["passed"], row["total"]))
    lines += ["", "时间缩放控制的少数通过样本表示它们在当前时长与精度下仍落在容差内；没有为了清零这些通过样本收紧阈值。", "",
              "校准时发现原 MuJoCo implicitfast 离散轨迹在高速条件下存在累积功—能误差和二阶约束误差，最初只有 58/96 通过。已在模型推理前把本次新数据改为独立理想约束 DOP853 轨迹，并用 MuJoCo 渲染及独立核对质量矩阵、偏置力。原 KeyCLD 训练数据、权重和旧实验结果没有改动。完整失败控制保留于 integrator_audit.json。", "",
              "## 各工况实际结果", "",
              "表中力矩、能量仅统计整段可测轨迹；不可测记为 —，不填零。", "",
              "| 工况 | 视频联合通过 | 状态联合通过 | 状态力矩残差均值 ± 标准差 (N·m) | 状态最大累计能量误差的轨迹均值 (J) |",
              "|---|---:|---:|---:|---:|"]
    for name, label in CONDITIONS.items():
        v, s = video["conditions"][name], state["conditions"][name]
        force, energy = s["metrics"]["dynamics"], s["metrics"]["energy"]
        force_text = "—" if force["mean"] is None else "%.6f ± %.6f" % (force["mean"], force["std"])
        energy_text = "—" if energy["mean"] is None else "%.6f" % energy["mean"]
        lines.append("| %s | %d/20 | %d/20 | %s | %s |" % (label, v["passed"], s["passed"], force_text, energy_text))
    lines += ["", "## 阈值敏感性", "",
              "在预测前登记统一收紧至 0.5 倍、保持原值、放宽至 2 倍三个设置；始终保留原始误差，不重新训练。", "",
              "| 所有通过线倍数 | 视频接口联合通过率（%） | 状态组件联合通过率（%） |", "|---|---:|---:|"]
    for multiplier in protocol["sensitivity_threshold_multipliers"]:
        lines.append("| %g | %.2f | %.2f |" % (multiplier, video["sensitivity_scores_100"][str(multiplier)], state["sensitivity_scores_100"][str(multiplier)]))
    lines += ["", "## 结论范围", "",
              "视频成绩评价的是当前声明的物理关键点输出接口。其几何不合法时，动力学与能量无法获得物理解释，联合通过计为 0；这不等于证明所有生成视频都违反物理，也不代表 KeyCLD 在其他任务上的能力为零。通用生成视频的公共读取器仍需另行校准。", "",
              "状态组件成绩使用真实状态标签训练和真实初态，只用于隔离视觉问题。原训练条件无摩擦且几何/质量固定，摩擦与载荷组属于冻结迁移。当前只有一个训练随机种子，覆盖 4R 与 3R+1P，未覆盖 Delta、Stewart。", "",
              "摩擦奇对称响应与 SI 质量尺度辨识的已有结果见 [扩展动力学报告](../extended_dynamics/results.md)，不把参数输出头是否存在混入六项联合分数。", "",
              "综合总分：score_summary.json；原始 SI 测量及分项、联合通过结果：scores.json；逐轨迹密集预测：predictions/；校准结果：calibration.json；输入和源代码哈希：dataset_manifest.json、execution_manifest.json。综合汇总未改写原始 scores.json，其输入和生成代码哈希单独记录在 score_summary.json。"]
    target = root / "results.md"
    temporary = target.with_suffix(".partial")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(target)
    print(str(target))


if __name__ == "__main__":
    main()
