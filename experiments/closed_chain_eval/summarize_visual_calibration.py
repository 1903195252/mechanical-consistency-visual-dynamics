"""Summarize independent instrument errors, coverage and optimization evidence."""
from collections import defaultdict
import csv
import json

import numpy as np

from .core import CORE_METRICS
from .visual_calibration import ROOT, OUT, load_protocol, sha, write


def empirical_bound(values, quantile=.95):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Bounds require finite nonnegative calibration measurements")
    return float(np.quantile(values, quantile, method="higher"))


def verify_bound(rows, getter, bound):
    """Report trajectory and all-conditions-within-assembly coverage separately."""
    groups = defaultdict(list)
    available, passed = 0, 0
    for row in rows:
        value = getter(row)
        valid = value is not None and np.isfinite(value)
        accepted = bool(valid and value <= bound)
        available += int(valid)
        passed += int(accepted)
        groups[row["assembly_id"]].append(accepted)
    return dict(available=available, passed=passed, total=len(rows),
                assemblies_passed=sum(all(v) for v in groups.values()), assemblies_total=len(groups))


def calibrated_metric(rows, getter):
    calibration = [r for r in rows if r["calibration_role"] == "calibration"]
    verification = [r for r in rows if r["calibration_role"] == "verification"]
    values = [getter(row) for row in calibration]
    valid = [v for v in values if v is not None and np.isfinite(v)]
    if not valid:
        return dict(bound=None, calibration_available=0, calibration_total=len(calibration),
                    full_coverage=False, verification=None)
    bound = empirical_bound(valid)
    return dict(bound=bound, calibration_available=len(valid), calibration_total=len(calibration),
                full_coverage=len(valid) == len(calibration),
                verification=verify_bound(verification, getter, bound))


def summarize():
    protocol = load_protocol()
    readers = list(protocol["readers"])
    extra = OUT / "shared_planar_reader/reader.json"
    if extra.exists():
        readers.append(json.loads(extra.read_text()))
    for relative in ("optimization/optical_motion/reader.json", "optimization/spatial_reader/readers.json",
                     "optimization/geometry_spatial_reader/readers.json", "optimization/reset_calibration/readers.json"):
        path=OUT/relative
        if path.exists():
            entries=json.loads(path.read_text())
            readers.extend(entries if isinstance(entries,list) else [entries])
    sources, summary, profiles, flat = {}, [], [], []
    for reader in readers:
        rows = []
        for row in protocol[reader["kind"] + "_records"]:
            case = row.get("case_id", row.get("video_id"))
            path = OUT / "measurements" / reader["id"] / (case + ".json")
            data = json.loads(path.read_text())
            if data["identity"]["protocol_sha256"] != sha(OUT / "protocol.json"):
                raise ValueError("Mixed calibration protocols")
            sources[str(path.relative_to(ROOT))] = sha(path)
            rows.append(data)
        families = sorted({r["category"] for r in rows})
        for family in families:
            group = [r for r in rows if r["category"] == family]
            verification = [r for r in group if r["calibration_role"] == "verification"]
            loc = calibrated_metric(group, lambda r: r["localization"]["point_euclidean_rmse"])
            item = dict(reader=reader["id"], kind=reader["kind"], category=family, role=reader["role"],
                calibration_cases=sum(r["calibration_role"] == "calibration" for r in group),
                verification_cases=len(verification), localization=loc,
                verification_point_rmse_mean_mm=1000 * np.mean([r["localization"]["point_euclidean_rmse"] for r in verification]),
                verification_coordinate_bias_rms_mean_mm=1000 * np.mean([r["localization"]["coordinate_bias_rms"] for r in verification]),
                verification_centered_coordinate_rms_mean_mm=1000 * np.mean([r["localization"]["centered_coordinate_rmse"] for r in verification]),
                native_qualified=sum(r["native_qualified"] for r in verification),
                intervention_qualified=sum(r["intervention_qualified"] for r in verification),
                correction_point_rms_mean_mm=1000 * np.mean([r["correction_point_rms_m"] for r in verification if r.get("correction_point_rms_m") is not None]))
            summary.append(item)
            for track in ("native", "intervention"):
                metrics = {key: calibrated_metric(group, lambda r, k=key: r.get(track + "_metrics", {}).get(k))
                           for key in CORE_METRICS}
                profiles.append(dict(id=reader["id"] + "/" + family + "/" + track,
                    reader_id=reader["id"], category=family, track=track, metrics=metrics,
                    units=dict(geometry="m",constraint_velocity="m/s",constraint_acceleration="m/s^2",
                               dynamics="N" if family=="H-A" else "N m",energy="J",prediction="m"),
                    complete_six_metric_calibration=all(v["full_coverage"] for v in metrics.values()),
                    model_score_application_enabled=False,
                    application_rule="instrument diagnosis and optimization; does not change engineering pass limits",
                    readout_role=reader["role"],
                    sample_interpretation="empirical trajectory quantiles; correlated conditions; no distribution-free confidence guarantee"))
        for row in rows:
            item = {key: row[key] for key in ("reader_id", "category", "assembly_id", "case_id", "calibration_role")}
            item.update(point_euclidean_rmse_m=row["localization"]["point_euclidean_rmse"],
                        coordinate_bias_rms_m=row["localization"]["coordinate_bias_rms"],
                        centered_coordinate_rmse_m=row["localization"]["centered_coordinate_rmse"],
                        native_qualified=row["native_qualified"], intervention_qualified=row["intervention_qualified"])
            for track in ("native", "intervention", "numerical_reference_sampled"):
                item.update({track + "_" + key: row.get(track + "_metrics" if track != "numerical_reference_sampled" else track, {}).get(key)
                             for key in CORE_METRICS})
            flat.append(item)
    result = dict(protocol_sha256=sha(OUT / "protocol.json"), readout_configurations=len(readers),
                  reference_measurements=len(flat), summaries=summary, profiles=profiles, sources=sources,
                  engineering_thresholds_changed=False, model_error_obtained_by_rmse_subtraction=False,
                  numerical_only_calibration_is_sufficient=False)
    write(OUT / "summary.json", result)
    write(OUT / "calibration_profiles.json", dict(protocol_sha256=result["protocol_sha256"], profiles=profiles))
    with (OUT / "case_measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ["# 视觉读取误差：独立标定与测量链路优化", "",
        "工程闭合、力矩和能量门槛保持原值。本报告测量公共仪器及模型自带读出的误差，并据此改进测量链路；不把读取误差变成模型免责额度。", "",
        "本轮实际完成了真值视频逐帧读取、数值求导优化、独立公共坐标读取器训练、空间坐标零点处理、"
        "分辨率匹配、图像质量鲁棒性训练及已有预测的配对重测。各版本均保存结果，改进无效的方案不替换原测量。"
        "数值参考的全项通过与图像读取的精度分别验收，不能互相替代。", "",
        "改进概览：[独立校准图](calibration_overview.pdf)（另存 PNG / SVG）。", "",
        "## 数据与独立性", "",
        "平面使用原 calibration 分区的 18 个机构、54 条视频：按机构划成 9 个机构／27 条标定与 9 个机构／27 条验证。"
        "空间使用原 validation 的 8 个机构、24 条正面视频：4 个机构／12 条标定与 4 个机构／12 条验证。"
        "同一机构的各工况不跨组；模型训练与正式测试机构不进入误差尺度拟合。"
        "仪器开发期间查看过这些参考上的版本差异，因此这里的改进对比是开发证据；空间端另外冻结仪器后在 10 个模型测试机构上做只读复验，复验数据不拟合噪声分位数。", "",
        "每条平面视频回放全部 90 帧，空间视频回放全部 120 帧。定位统计用未来帧；求导与力学使用各自登记的完整中心窗口。"
        "分位数按轨迹统计并检查独立机构覆盖，不把帧当作独立样本。空间每族仅有 2 个标定和 2 个验证机构，经验 95% 分位数不是 95% 置信保证。", "",
        "KeyCLD/CON 自带编码器及其坐标读出是模型接口诊断，不能作为独立量具替自身放宽标准。"
        "空间 V-JEPA 读取器对 LingBot 是外部量具，对 V-JEPA 自身则属于模型读出路径。"
        "新增平面公共读取器只用原训练机构的 RGB 与坐标标签拟合，固定最终训练轮次，不输入几何、质量、未来状态或强制杆长。"
        "空间 geometry 版本显式使用 CAD 的固定坐标零点，预测自由的坐标偏移；reset 版本另使用一次已知复位姿态的观测标定，后续各工况共享该标定。"
        "reset 是额外登记的测量设置，只适用于确有已知复位姿态的实验，不与原无复位标定配置混称。", "",
        "## 独立验证集的定位结果", "",
        "Point RMSE 为每个物理点的欧氏距离 RMS；bias/spread 是坐标分量 RMS。它们与旧论文的最大坐标通道 RMS 不同，不能直接混成一列。", "",
        "| 读取器 | 机构族 | 验证视频 | Point RMSE 均值 (mm) | Bias (mm) | Centered spread (mm) | 标定 P95 (mm) | 验证覆盖 |", 
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        loc = r["localization"]
        lines.append("| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {}/{} |".format(
            r["reader"], r["category"], r["verification_cases"], r["verification_point_rmse_mean_mm"],
            r["verification_coordinate_bias_rms_mean_mm"], r["verification_centered_coordinate_rms_mean_mm"],
            1000 * loc["bound"], loc["verification"]["passed"], loc["verification"]["total"]))
    lines += ["", "Bias 与 centered spread 分开保留；后者仍含随姿态变化的系统误差，不能全部称作随机噪声。"
              "逐轨迹协方差及一阶时间互协方差位于 measurements/，便于后续传播；像素换算仅描述标定尺度，不是单像素精度下限。", "",
              "## 下游六项误差与几何干预", "",
              "原始读取保留其几何残差。力学不可测项保留为空；原始代数方程诊断不升级为合格 SI 测量。"
              "后验重构使用公共几何，单列修正距离与力学误差，其闭合不能归功于原始视频或模型。", "",
              "| 读取器 | 机构族 | 原始力学可测/验证数 | 重构力学可测/验证数 | 重构修正均值 (mm) |",
              "|---|---|---:|---:|---:|"]
    for r in summary:
        lines.append("| {} | {} | {}/{} | {}/{} | {:.3f} |".format(r["reader"], r["category"], r["native_qualified"],
            r["verification_cases"], r["intervention_qualified"], r["verification_cases"], r["correction_point_rms_mean_mm"]))
    numerical = OUT / "optimization/coordinate_derivatives/verification.json"
    if numerical.exists():
        n = json.loads(numerical.read_text())
        lines += ["", "## 已执行的数值优化", "",
                  "仅在原 validation 机构上选择求导器，原 calibration 的 54 条参考轨迹负责独立检查。"
                  "所有候选用相同的 80 个内部采样点；不修改位置、杆长、动力学方程或工程阈值。", "",
                  "| 求导配置 | 六项通过 | 最大速度约束 (m/s) | 最大加速度约束 (m/s²) | 最大力矩 RMS (N·m) | 最大功—能缺陷 (J) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for method, s in n["summary"].items():
            m = s["maxima"]
            lines.append("| {} | {}/{} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(method,s["joint_pass"],s["total"],
                m["constraint_velocity"],m["constraint_acceleration"],m["dynamics"],m["energy"]))
        lines += ["", "该选择证明精确状态的数值误差可以明显降低；带图像定位误差时另行优化，不能直接把插值求导器当作降噪器。"]
    sensitivity=OUT/"optimization/sensitivity.json"
    if sensitivity.exists():
        controls=json.loads(sensitivity.read_text())["summary"]
        lines += ["", "数值灵敏度检查：正确参考 {}/54 全项通过；5 mm 径向扰动、0.8 倍时间缩放和 0.5 N·m 错误执行器输入分别检出 {}/54、{}/54、{}/54。".format(
            controls["correct"]["joint_pass"],54-controls["radial_defect_5mm"]["joint_pass"],
            54-controls["time_scale_0_8"]["joint_pass"],54-controls["wrong_actuator_plus_0_5Nm"]["joint_pass"])]
    stress=OUT/"optimization/robust_planar_reader/stress.json"
    if stress.exists():
        records=json.loads(stress.read_text())["rows"]
        lines += ["", "## 生成图像质量差异：受控降质标定与改进", "",
                  "对物理状态不变的参考图像施加固定模糊／对比度变化，检验清晰图像标定能否覆盖读取器的图像质量变化。"
                  "鲁棒版仅用训练图像的随机降质继续训练 8 轮，不拟合模型预测或验证标签。", "",
                  "| 参考图像质量 | 原公共读取器定位 RMSE (mm) | 鲁棒版定位 RMSE (mm) |",
                  "|---|---:|---:|"]
        for quality in ("clean","mild_blur","blur","blur_low_contrast"):
            values=[1000*np.mean([r["point_rmse_m"] for r in records if r["reader"]==reader and r["quality"]==quality
                                  and r["calibration_role"]=="verification"]) for reader in ("baseline","robust")]
            lines.append("| {} | {:.3f} | {:.3f} |".format(quality,*values))
        lines += ["", "分辨率、模糊和对比度都属于量具适用条件。该实验覆盖登记的降质范围，不把清晰参考的误差界自动推广到任意生成图像。"
                  "鲁棒版的各质量条件还通过同一 SI 求导和重构路径保存完整六项测量，见 paired_robust_planar_rgb/calibration.json。"]
    phase_path=OUT/"optimization/constrained_visual_phase/verification.json"
    if phase_path.exists():
        phase=json.loads(phase_path.read_text())
        controls=json.loads((phase_path.parent/"sensitivity.json").read_text())
        lines += ["", "## 带视觉噪声的求导优化与保真验收", "",
                  "高阶插值适合精确坐标，却会放大视觉读取噪声。为此，另在后验重构路径上拟合测得的曲柄相位，"
                  "所有候选使用相同的 80 个内部采样点；原始坐标、闭合误差和重构修正距离仍分别保存。"
                  "滤波器不输入驱动力、摩擦或未来真值。", "",
                  "先在原 validation 的 54 条正确轨迹上要求六项全部通过，再以该开发集四种图像质量下的位置、"
                  "速度、加速度误差选择窗口。仅按噪声均方误差选出的 21 帧强平滑版本损害了正确参考，因此被否决；"
                  "最终选择 7 帧三次相位拟合（phase_sg7d3）。以下是另一组 27 条验证视频的后验重构测量均值。", "",
                  "| 图像质量 | 求导方法 | 力矩残差均值 (N·m) | 功—能缺陷均值 (J) | 重构后六项通过 |",
                  "|---|---|---:|---:|---:|"]
        for r in phase["summary"]:
            if r["role"]!="verification":continue
            lines.append("| {} | {} | {:.4f} | {:.4f} | {}/{} |".format(r["quality"],r["method"],
                r["mean"]["force_balance_Nm"],r["mean"]["work_energy_J"],r["all_six_pass"],r["total"]))
        check=controls["summary"]
        lines += ["", "独立精确状态复验：正确轨迹 {}/54 通过；0.8 倍时间缩放、1 Hz / 0.1 rad 相位扰动、"
                  "0.5 N·m 错误输入分别检出 {}/54、{}/54、{}/54；原始坐标上的 5 mm 几何缺陷检出 {}/54。".format(
                      check["correct"]["passed"],54-check["time_scale_0_8"]["passed"],54-check["phase_1Hz_0_1rad"]["passed"],
                      54-check["wrong_actuator_plus_0_5Nm"]["passed"],controls["raw_radial_defects_detected"]), "",
                  "相位拟合降低了视觉测量误差，但参考视频的力矩和能量残差仍高于工程门槛。频率响应也单独保存："
                  "1 Hz 加速度幅值保留约 96.5%，3 Hz 约 72.0%。因此保留未滤波的并行测量，"
                  "不能用平滑路径认证高频响应，或将其几何重构结果当作模型原生闭合。"]
    lines += ["", "## 改进记录与采用规则", "",
              "平移集成、光流融合、局部亚像素校正均在开发集上实际测试；它们未改善本轮读取结果，开发选择保留单帧公共读取器，"
              "不把失败试验包装成提升。数值插值在真值状态上改进明显，但会放大图像噪声，故 `optimized_planar_64` 是候选诊断路径，"
              "不代表已通过所有视觉力学精度验收。", "",
              "空间坐标零点处理同时降低了点位和闭合读出误差；已知复位标定有独立的信息前提，并未在所有指标上进一步改善。"
              "各版本分别报告，不能按每个测试指标挑最有利的版本拼成一行。"]
    confirmation=OUT/"spatial_confirmation/summary.json"
    if confirmation.exists():
        confirmed=json.loads(confirmation.read_text())["summary"]
        lines += ["", "## 冻结仪器后的空间复验", "",
                  "原模型测试机构的正确参考视频只作仪器复验：两族各 5 个机构、15 条视频。没有用于头部拟合、配置选择或误差分位数拟合。", "",
                  "| 输入宽度 | 机构族 | 视频数 | 点位 RMSE (mm) | 原始力学可测 | 重构力学可测 |",
                  "|---:|---|---:|---:|---:|---:|"]
        for r in confirmed:
            lines.append("| {width} | {family} | {total} | {point_rmse_mean_mm:.3f} | {native_qualified}/{total} | {intervention_qualified}/{total} |".format(**r))
    for relative,title in (("paired_planar_rgb","原公共读取器"),("paired_robust_planar_rgb","鲁棒公共读取器")):
        paired=OUT/relative/"summary.json"
        if paired.exists():
            paired_summary=json.loads(paired.read_text())["summary"]
            lines += ["", "## 已有预测配对重测："+title, "",
                      "这是同一冻结公共读取器下的图像测量诊断；模型原生状态与仿射潜空间读出单独报告。"
                      "模型的预测和权重未重训；Reference 是同测试机构的正确视频测量对照，未用于标定量具。"
                      "有限子集的均值与完整分母分开报告，不作为已排除全部读出误差的模型排行榜。", "",
                      "| 来源 | 有限/总数 | 读取点位 RMSE 均值 (mm) | 原始闭合残差均值 (mm) |",
                      "|---|---:|---:|---:|"]
            for r in paired_summary:
                lines.append("| {model} | {finite}/{total} | {point_rmse_mean_mm:.3f} | {geometry_mean_mm:.3f} |".format(**r))
    phase_pair=OUT/"paired_constrained_visual_phase/summary.json"
    if phase_pair.exists():
        rows=json.loads(phase_pair.read_text())["summary"]
        lines += ["", "## 同一相位测量下的模型与正确视频对照", "",
                  "上述冻结配置也回放了全部 648 条模型记录与 108 条匹配正确视频。非有限预测保留在分母中。"
                  "这里报告后验重构路径，原始几何失败没有被重构覆盖；参考残差与模型残差并列，不能相减成模型净误差。", "",
                  "| 来源 | 有限/总数 | 力矩残差均值 (N·m) | 功—能缺陷均值 (J) | 重构后六项通过 |",
                  "|---|---:|---:|---:|---:|"]
        for r in rows:
            lines.append("| {model} | {finite}/{total} | {force_mean_Nm:.4f} | {energy_mean_J:.4f} | {projected_all_six_pass}/{total} |".format(**r))
    lingbot=OUT/"paired_lingbot/summary.json"
    if lingbot.exists():
        items=json.loads(lingbot.read_text())["summary"]
        lines += ["", "## LingBot 既有视频的新读取器重测", "",
                  "零样本与已适应版本共 60 条原预测视频，分别经过 CAD 坐标读出和已知复位标定两个固定设置。"
                  "这些数字是测量链路的输出，须与上面的正确视频复验误差一起解释。新读取器没有进入预测模型或重新生成视频。", "",
                  "| 模型分支 | 读取设置 | 机构族 | 视频数 | 点位 RMSE (mm) | 原始力学可测 | 重构力学可测 |",
                  "|---|---|---|---:|---:|---:|---:|"]
        for r in items:
            lines.append("| {mode} | {reader} | {category} | {total} | {point_rmse_mean_mm:.3f} | {native_qualified}/{total} | {projected_qualified}/{total} |".format(**r))
    lines += ["", "## 归因与后续接入", "",
              "模型结果必须区分：原生状态输出的误差、模型自带坐标适配器的误差、公共视觉量具的误差，以及求导／积分的误差。"
              "模型图像和参考图像须经过同一冻结测量链路。旧的原生 ODE 导数成绩不能套用新的 30 Hz 图像求导误差尺度。", "",
              "两组 RMSE 不能简单相减来得到模型净误差。参考视频误差高时继续优化读取器与时间处理；"
              "归因检查在改进期间保留，防止测量误差被提前写成模型物理失败。", "",
              "完整数据：summary.json、calibration_profiles.json、case_measurements.csv、measurements/。"
              "旧模型预测、旧评分及论文数据保持原文件；新配置只有在匹配的模型输出上重评分后才产生新版模型成绩。", ""]
    (OUT / "results.md").write_text("\n".join(lines))
    print(json.dumps(dict(readers=len(readers), measurements=len(flat), report=str(OUT / "results.md"))))


if __name__ == "__main__":
    summarize()
