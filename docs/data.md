# Data, weights and storage

## What is available in the source repository?

Frozen manifests, protocols, model-download metadata, per-case numerical records and the current paper sources are included. Videos, arrays and trained checkpoints are stored outside Git.

Official pretrained weights can be downloaded now:

```bash
python3 scripts/download_models.py all --list
python3 scripts/download_models.py vjepa
python3 scripts/download_models.py lingbot
python3 scripts/download_models.py all --verify-only
```

V-JEPA uses the official B/16 384-pixel checkpoint. LingBot uses the fixed ModelScope revision of Robbyant/lingbot-va-base, including Transformer, VAE, tokenizer and text encoder. All files are verified against the recorded hashes. A partial transfer resumes on the next invocation; an existing mismatched final file is preserved and reported.

KeyCLD and CON training starts from the registered random initialization; their upstream source is installed by `setup.sh`. Experiment-trained weights are separate from these official pretrained weights.

## Experiment artifacts

Experiment data and trained weights are distributed separately from Git. Public download URLs have not yet been assigned. The artifact directory uses canonical runtime paths, a SHA-256 manifest, and a group inventory. Import the directory directly or use an archive:

```bash
python3 scripts/artifacts.py verify /path/to/experiment-bundle.tar
python3 scripts/artifacts.py import /path/to/experiment-bundle.tar
python3 scripts/artifacts.py verify /path/to/artifacts/current
python3 scripts/artifacts.py import /path/to/artifacts/current
```

For the full artifact collection, including already downloaded official models:

```bash
./setup.sh --profile all --device cuda --download-models --artifact-dir /path/to/artifacts
```

The collection index selects `current` and `pretrained` by default. The installer imports them before checking the official download list, so verified local weights are reused. Import `vjepa-resume` explicitly only when continuing AC training. A standalone `current` directory works with the same command; missing official components are then downloaded normally.

A directory or bundle contains `artifact_manifest.json` and the exact `output/...` files it lists. Each entry has `path`, `bytes` and `sha256`; directory inventories also record the experiment group. Imports check containment, sizes, hashes, and conflicting existing files. Matching destination files are preserved. Directory import avoids allocating an additional archive-sized staging file; allow enough disk space for the copied runtime files.

The current core experiment collection contains 27 planar final checkpoints and 2,916 test predictions, six V-JEPA AC predictors with six internal feedback heads and 900 saved feature forecasts, and three adapted LingBot checkpoints with 1,800 generated videos. It includes matched reference RGB, native and reconstructed coordinate trajectories, physical reference states, commands, calibration heads, and all evaluated failure records. Model source and official pretrained components are installed separately by `setup.sh`.

The [artifact catalog](../configs/artifact_catalog.json) lists package sizes, coverage and checkpoint hashes. The core collection is 83.02 GiB; official pretrained components add 24.26 GiB. Supplementary torque-study artifacts include three final checkpoints and all 160 extended-condition observations and references. The original 590 clips and state references from the initial single-mechanism training/test study have not been recovered; their registered manifest and generation recipe are retained.

| Artifact group | Runtime location |
|---|---|
| Shared corpus RGB, public inputs, physical references | `output/mechanism_corpus/` |
| 64-pixel planar observations and state labels | `output/keycld_corpus/` |
| 256-pixel observations | `output/planar_resolution/256/` |
| 256-pixel KeyCLD/CON final weights and predictions | `output/planar_resolution/runs/` |
| Additional spatial assemblies | `output/lingbot_expansion/corpus/` |
| LingBot seeds 42/43/44, 0/50 predictions | `output/lingbot_expansion/runs/` |
| V-JEPA AC predictors, internal heads and features | `output/vjepa_expansion/seed_42/`, `seed_43/`, `seed_44/` |
| Fixed measurement heads | `output/closed_chain_eval/` and `output/lingbot_expansion/delta_instrument/` |
| Official model weights | `output/model_expansion/weights/` |

After import, check both required inputs and the exact experiment case identities:

```bash
python3 scripts/doctor.py --profile all --weights
python3 scripts/check_artifact_coverage.py
```

Coverage can also be checked before import with `python3 scripts/check_artifact_coverage.py /path/to/artifacts/current`. Use `--profile planar`, `vjepa`, or `lingbot` for a selected experiment. Hash verification and case coverage are separate checks: a directory can contain intact files while missing a seed or test trajectory.

The CPU checks run immediately after setup. Model training and full rescoring require the corresponding artifacts at these paths. Inference checkpoints and optimizer-resume states are separate; only the latter support exact training continuation. The optional `vjepa-resume` directory contains the six full AC training states; it is unnecessary for inference or rescoring. LingBot's exported adapted Transformer weights are evaluation checkpoints, not optimizer-resume states.

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

## Preparing a transferable bundle

Create a JSON list of explicit file mappings, for example `[ {"path": "output/model/params.p", "source": "output/local-run/params.p"} ]`. `source` defaults to `path`. Both are relative to the selected source root; only canonical destination paths are stored in the bundle.

```bash
python3 scripts/artifacts.py pack selection.json weights.tar --source-root /path/to/experiment-root
```

Packing checks tensor-file completeness, writes checksums, verifies the archive, and removes no source files. An optional `sha256` in each selection entry also checks against a previously recorded digest. Separate dataset, measurement-head, model-weight and prediction bundles keep downloads selective.
