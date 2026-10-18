# Reproduction

Run commands from the repository root. The default `./setup.sh` prepares CPU evaluation and runs numerical checks. `./setup.sh --profile all --device cuda --download-models` prepares separate JAX/PyTorch environments and the fixed official pretrained models.

Datasets, experiment checkpoints and measurement heads are imported separately using [docs/data.md](data.md). Setup never changes model architecture, loss horizons, engineering tolerances or test partitions to fit available hardware.

## 1. Inspect and verify

```bash
./run.sh verify
python3 scripts/download_models.py all --list
python3 scripts/check_artifact_coverage.py /path/to/artifacts/current
```

The tests check constraint-reaction cancellation, friction work, differentiated constraints, continuous scoring, and incomplete-case denominators. Result verification reads the included current records without training a model.

Imported forecasts are the frozen evidence for the paper. They can be remeasured and rescored directly. For a new training or generation run, use a separate checkout with the input data and measurement heads; retain the imported forecasts separately. Run fingerprints distinguish the released adapter from the original execution source and prevent silently mixing predictions from different implementations.

## 2. Planar models

At each resolution, KeyCLD uses 40 epochs, batch size 1 and 32,520 updates. CON uses 50 epochs, batch size 16, 2,550 updates and 40,650 sequence exposures. All use final-epoch checkpoints and seeds 42/43/44.

64-pixel commands:

```bash
./run.sh planar -m experiments.keycld_corpus.train --root output/keycld_corpus --arm video_euler --seed 42
./run.sh planar -m experiments.keycld_corpus.infer --root output/keycld_corpus --arm video_euler --seed 42
./run.sh planar -m experiments.con_corpus.prepare
./run.sh planar -m experiments.con_corpus.train --arm video_con_iae --seed 42
./run.sh planar -m experiments.con_corpus.infer --arm video_con_iae --seed 42
```

Repeat for the other seeds. KeyCLD state controls use `state_euler` / `state_rk4`. CON state controls use `state_con_iae`, and paired integrator training uses `train_integrator --solver euler` / `rk4`.

256-pixel commands:

```bash
./run.sh planar -m experiments.planar_resolution.run_keycld --mode train --width 256 --seed 42
./run.sh planar -m experiments.planar_resolution.verify_con_memory --frame-chunk 16
./run.sh planar -m experiments.planar_resolution.run_con --mode train --width 256 --seed 42 --frame-chunk 16
./run.sh planar -m experiments.planar_resolution.forecast --model keycld --width 256 --seed 42
./run.sh audit -m experiments.planar_resolution.score --model keycld --width 256 --seed 42
```

Repeat forecasting/scoring for `--model con` and seeds 43/44. Memory preflight metadata is included. Framewise execution retains the complete native objective and requires the objective/gradient equivalence check.

The `reader`, `calibrate_reader`, `refine_measurement`, and `measure_rgb` modules implement the independent RGB measurement pathway. Reference replay and generated-video measurement use the same frozen reader and differentiator.

## 3. V-JEPA 2.1 + AC

```bash
./run.sh spatial -m experiments.vjepa_expansion.run cache --family H-A
./run.sh spatial -m experiments.vjepa_expansion.run reference --family H-A
./run.sh spatial -m experiments.vjepa_expansion.run train --family H-A --seed 42
./run.sh spatial -m experiments.vjepa_expansion.run infer --family H-A --seed 42
./run.sh spatial -m experiments.vjepa_expansion.measure --family H-A --seed 42
```

Repeat for H-B and seeds 43/44. Each family/seed gets 3,000 sequence exposures. The frozen encoder uses 384-pixel images. Internal state-head feedback remains active during prediction; calibrated measurement heads operate on the completed feature trajectories. Fit those heads on the training/calibration split or import their frozen weights.

## 4. LingBot-VA

The base model, action normalization and text embeddings must be available before caching. The native preparation function builds the frozen prompt embeddings from the official text encoder:

```bash
./run.sh spatial -c "from experiments.model_expansion.lingbot_spatial import prepare_latents; prepare_latents()"
./run.sh spatial -m experiments.lingbot_expansion.run_model cache
./run.sh spatial -m experiments.lingbot_expansion.run_model preflight --seed 42
./run.sh spatial -m experiments.lingbot_expansion.run_model train --seed 42
./run.sh spatial -m experiments.lingbot_expansion.run_model infer --seed 42 --step 0
./run.sh spatial -m experiments.lingbot_expansion.run_model infer --seed 42 --step 50
./run.sh spatial -m experiments.lingbot_expansion.evaluate read --seed 42 --step 50
./run.sh audit -m experiments.lingbot_expansion.evaluate score --seed 42 --step 50
```

Repeat for the other seeds and both inference checkpoints. Training performs 50 updates with effective batch size 8; generation uses 256-pixel images and 117 future frames. Reference RGB and exact/sampled-state controls use the same registered measurement configuration. See the module help for reference-only measurement.

`WAM_STOP_UNIX` optionally supplies a UNIX computation deadline. Without it, launchers run to their registered completion and never shut down the host.

## 5. Data generation and manuscript

`mechanism_corpus` constructs the closed-chain mechanisms and performs physical/geometry checks. The included frozen manifests define the reported cohort; rendering a new cohort is a separate data build.

The checked-in LaTeX tables and figure assets compile without full experimental artifacts:

```bash
cd manuscript/keycld_cvpr
bash build.sh
```

The arXiv source is in `manuscript/keycld_arxiv/source/`. Figure scripts and their numerical inputs are included under the CVPR manuscript. Full remeasurement requires the external datasets and predictions.

## Continuous integration

`ci/github-actions.yml` is the GitHub Actions template for source hygiene, result recomputation and physics tests. Place it at `.github/workflows/checks.yml` in a repository with workflow-write permission to enable it. The same numerical checks run locally through `./run.sh verify`.
