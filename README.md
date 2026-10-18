# Mechanical Consistency in Visual Dynamics

Code and numerical evidence for **Beyond Geometric Closure: Measuring and Diagnosing Mechanical Consistency in Visual Dynamics**, by Qibin Gu and Yunfeng Yan.

The framework measures geometric closure, tangent velocity, curvature acceleration, reaction-free actuator effort, work–energy balance, and point tracking in SI units. Controlled interventions separate localization, geometric readout, integration, and constitutive behavior. The primary score is continuous Scomp; physical coverage and strict joint qualification accompany it.

[中文](README.zh-CN.md) · [Reproduction](docs/reproduction.md) · [Data and weights](docs/data.md) · [Results](docs/results.md) · [Architecture](docs/architecture.md) · [Third-party sources](THIRD_PARTY.md)

## One-command setup

On Linux, from the repository root:

```bash
./setup.sh
```

This creates an isolated CPU audit environment, restores metadata, and runs physics and result checks. No GPU, videos, or model weights are needed.

For all model environments, pinned upstream sources, and official pretrained weights:

```bash
./setup.sh --profile all --device cuda --download-models
```

Audit, JAX, and PyTorch dependencies stay in separate environments. Downloads are resumable and verified. Select `--profile planar`, `--profile vjepa`, or `--profile lingbot` to install one pathway; `--dry-run` shows the plan.

```bash
./run.sh verify
./run.sh planar -m experiments.planar_resolution.run_keycld --mode train --width 256 --seed 42
./run.sh spatial -m experiments.vjepa_expansion.run train --family H-A --seed 42
```

Training requires the experiment datasets. Official pretrained weights have download URLs; experiment datasets and trained checkpoints are separate artifacts. See [artifact import and hardware requirements](docs/data.md). Blender and FFmpeg are required to render new data.

## Experiments

| Study | Models | Test scope | Comparison |
|---|---|---|---|
| Planar visual dynamics | KeyCLD, CON | 36 held-out assemblies × 3 conditions × 3 seeds | 64 and 256 pixels; native and exact-state controls |
| Torque and structure | KeyCLD | Single-mechanism transfer and shared-assembly probes | Velocity reversal, actuator gain, point-mass capacity |
| Spatial feature prediction | V-JEPA 2.1 + AC | 50 Stewart + 50 Delta assemblies, 3 conditions, 3 seeds | 3,000 exposures per family/seed; 900 forecasts |
| Spatial video generation | LingBot-VA | The same 300 test trajectories and 3 seeds | 0 / 50 updates; 1,800 videos |

The planar corpus has 343 assemblies: 271 training, 18 validation, 18 calibration, and 36 test identities. Spatial experiments share test identities across models. Evaluation includes matched reference replay, fixed-feature readout controls, and CAD/IK reconstruction with its coordinate correction cost.

## Layout

```text
experiments/
  closed_chain_eval/     SI residuals, continuous scoring and measurement
  fourbar/              Analytical mechanics and simulation
  mechanism_corpus/     Mechanism construction, physics and rendering
  keycld_fourbar/        Torque-excitation and structural probes
  keycld_corpus/         Shared-assembly KeyCLD and state controls
  con_corpus/            Native CON and integrator controls
  planar_resolution/     256-pixel training, forecasting and RGB measurement
  model_expansion/       Shared spatial model adapters and mechanics
  lingbot_expansion/     Three-seed 0/50-update video experiment
  vjepa_expansion/       Three-seed AC experiment and fixed-feature readouts
configs/                Revisions, protocols and artifact metadata
data/manifests/         Frozen partitions
results/                Current numerical evidence
manuscript/             Current CVPR/arXiv sources and figure assets
scripts/                Installation, downloads, imports and verification
docs/                   Protocols and usage
```

Generated datasets, weights, predictions, and environments are ignored by Git. Upstream licenses are retained in [THIRD_PARTY.md](THIRD_PARTY.md). Original project code has no assigned redistribution license at this stage.

```bash
./setup.sh --profile all --device cuda --download-models --artifact-dir /path/to/artifacts/current
```

`--artifact-dir` accepts a manifested data directory or a directory containing verified archives. Check all seeds and test cases with `python3 scripts/check_artifact_coverage.py` after import.
