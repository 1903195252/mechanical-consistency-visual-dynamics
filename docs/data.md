# Data and model packages

Artifacts are organized by **model, then experiment type**. Shared datasets and independent measurement components have their own directories. Every leaf directory is a transferable package with a `package.json` description, a file index, model-component index, and canonical `output/` runtime paths.

## Package layout

| Directory | Packages |
|---|---|
| `data/` | Planar 64 px, planar 256 px, shared spatial data |
| `keycld/` | Planar 64 px, planar 256 px, torque and structural diagnostics |
| `con-model/` | Planar 64 px, planar 256 px |
| `vjepa/` | Official encoder, spatial experiments, continuation states for seeds 42/43/44 |
| `lingbot/` | Official Transformer/VAE/text components, adapted weights by seed, 0/50-update forecasts, matched references and protocols |
| `measurement/` | Planar and spatial measurement heads, calibration and controls |

The collection contains 24 packages. The largest is 14.35 GB; every package is below the 50 GB limit. Sizes use decimal GB. Package identifiers, contents and sizes are listed in the [artifact catalog](../configs/artifact_catalog.json).

## Importing packages

From the source repository:

```bash
./setup.sh --profile all --device cuda --download-models --artifact-dir /path/to/artifacts
```

The collection imports the data, final model weights, predictions, measurement components and official pretrained models by default. It imports official components before checking for missing downloads. Setup installs dependencies and prepares the repository; it does not start training.

V-JEPA continuation states are optional and separated by seed:

```bash
python3 scripts/artifacts.py import /path/to/artifacts/vjepa/resume-seed42
```

An individual package also works with `--artifact-dir` or `scripts/artifacts.py import`. For a model-specific workflow, include the corresponding shared data, measurement and pretrained packages. V-JEPA's `*_predictor.pt` files are AC predictors; `*_readout.pt` files are internal autoregressive feedback heads. Independently calibrated measurement heads are indexed separately. LingBot adapted weights are Transformer evaluation checkpoints; V-JEPA continuation packages additionally contain optimizer state.

Official pretrained components can also be fetched with:

```bash
python3 scripts/download_models.py vjepa
python3 scripts/download_models.py lingbot
```

KeyCLD and CON train from their registered random initializations; `setup.sh` installs their upstream implementations. Dataset rendering additionally requires Blender and FFmpeg.

## Transferring one package

Package directories can be copied directly. To create a single upload file:

```bash
python3 scripts/artifacts.py export /path/to/artifacts/con-model/planar-256 con-planar-256.tar
python3 scripts/artifacts.py import con-planar-256.tar
```

Export also enforces the 50 GB archive limit. Each TAR contains relative runtime paths and neutral ownership and timestamp metadata. It can be imported independently; no concatenation of volume fragments is needed. Existing different destination files are preserved and reported.

To regenerate package navigation and file/model indexes:

```bash
python3 scripts/index_artifacts.py /path/to/artifacts
```

The source repository contains code, protocols and numerical results. The model and dataset packages are distributed separately; public download URLs have not yet been assigned.

## Fixed data partitions

| Dataset | Training | Validation | Calibration | Test |
|---|---:|---:|---:|---:|
| Planar | 271 assemblies / 813 sequences | 18 / 54 | 18 / 54 | 36 / 108 |
| Stewart | 14 assemblies / 210 multi-view training videos | 4 assemblies | Disjoint measurement subsets | 50 / 150 front-view trajectories |
| Delta | 4 assemblies / 60 multi-view training videos | 4 assemblies | Disjoint measurement subsets | 50 / 150 front-view trajectories |

Three conditions from the same assembly are grouped observations. Three model seeds produce 450 forecasts per spatial family. Future states and actual efforts are evaluator references; the model receives its observed prefix, public commands, and the declared geometry/camera inputs.

## Hardware

CPU evaluation uses the audit environment. Planar training uses Python 3.12 and JAX; the native CON 256 model has 1.08B parameters and requires a large-memory GPU or its verified optimizer-memory pathway. Spatial models use PyTorch. LingBot adapts the full 5.09B Transformer; model parameters and optimizer state require substantial GPU and host memory. A 96 GiB GPU and at least 384 GiB host RAM match the documented full-model execution setting.

The setup script installs software but does not start training. Run a complete-objective preflight before allocating a long job on a new device. New corpus rendering additionally needs Blender and FFmpeg.
