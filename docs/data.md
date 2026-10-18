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

Public download URLs for the experiment datasets and trained weights have not yet been assigned. The source repository provides a verified local bundle importer:

```bash
python3 scripts/artifacts.py verify /path/to/experiment-bundle.tar
python3 scripts/artifacts.py import /path/to/experiment-bundle.tar
```

A bundle contains `artifact_manifest.json` and the exact `output/...` files it lists. Each entry has `path`, `bytes` and `sha256`. Imports reject path traversal, links, corrupt data, and conflicting existing files. They preserve matching files, so repeated imports are safe.

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

After importing bundles, run `python3 scripts/doctor.py --profile all --weights` to check required data paths and official weight checksums.

The CPU checks run immediately after setup. Model training and full rescoring require the corresponding artifacts at these paths. Inference checkpoints and optimizer-resume states are separate; only the latter support exact training continuation.

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
