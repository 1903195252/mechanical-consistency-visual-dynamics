# 视觉动力学的力学一致性

论文 **Beyond Geometric Closure: Measuring and Diagnosing Mechanical Consistency in Visual Dynamics** 的实验源码与数值证据。

框架分开测量视觉预测、几何可达性和力学一致性，以六项 SI 指标与连续容差比得分评价预测，并通过读取、积分器及力矩干预定位误差来源。

## 一键使用

```bash
# 安装独立 CPU 环境、恢复元数据并自检，无需模型或视频。
./setup.sh

# 准备全部模型环境、固定版本源码及官方预训练权重。
./setup.sh --profile all --device cuda --download-models

# 检查数据划分、数值结果和物理不变量。
./run.sh verify
```

环境按 CPU 审计、JAX 平面模型和 PyTorch 空间模型隔离。可用 `--profile planar`、`--profile vjepa`、`--profile lingbot` 单独准备，用 `--dry-run` 查看计划。实验数据与训练权重单独存放，支持校验导入；发布状态见[数据与模型](docs/data.md)。

## 当前实验

- 平面：343 个机构，KeyCLD 与 CON 均完成 64/256 分辨率和三种子；包含精确状态、Euler/RK4、几何重构与本构探针。
- 空间：50 个 Stewart＋50 个 Delta，每机构三工况，统一 seeds 42/43/44。
- V-JEPA＋AC：每家族每种子 3,000 次曝光，共 900 条预测；内部反馈头与事后校准读取头分开评估。
- LingBot-VA：0/50 更新，每种子 400 次曝光，共 1,800 段视频。
- 主评分：固定工程容差下的连续 Scomp；覆盖率与严格联合合格率作为辅助指标，保留真实视频和精确状态对照。

[架构](docs/architecture.md) · [复现](docs/reproduction.md) · [结果](docs/results.md) · [英文首页](README.md)

```bash
./setup.sh --profile all --device cuda --download-models --artifact-dir /path/to/artifacts/current
python3 scripts/check_artifact_coverage.py
```

数据目录包含 `artifact_manifest.json` 和统一的 `output/` 路径，可直接校验、导入，无需先打成巨型压缩包。SHA-256 校验保证文件完整；覆盖检查另外核对全部种子、测试轨迹、模型权重和参考视频。V-JEPA 的完整优化器状态单独放在可选的 `vjepa-resume` 目录，日常推理与评分不需要导入。
