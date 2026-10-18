# Architecture

The numerical evaluator is independent of model training. The six SI residuals and fixed engineering targets are defined in `experiments/closed_chain_eval/`; individual mechanism adapters supply mass matrices, forces, constraint Jacobians and tangent maps.

| Layer | Entry points | Outputs |
|---|---|---|
| Data | `mechanism_corpus`, `fourbar`, frozen manifests | RGB, public commands/calibration, separate physical references |
| Planar models | `keycld_corpus`, `con_corpus`, `planar_resolution` | Native coordinates/derivatives and predicted RGB |
| Spatial models | `vjepa_expansion.run`, `lingbot_expansion.run_model` | Saved feature or RGB trajectories |
| Measurement | `closed_chain_eval`, `lingbot_expansion.evaluate`, `vjepa_expansion.measure` | Frozen physical readouts and matched reference measurements |
| Audit | `closed_chain_eval.continuous`, mechanism-specific scoring | Six SI errors, Scomp, coverage and joint qualification |
| Presentation | `manuscript/keycld_cvpr/scripts` and `sources` | Current tables and figures |

Model inputs and evaluator references have separate files and access paths. A native prediction is checked before CAD/IK reconstruction. Reconstruction is recorded as an intervention with a coordinate correction cost.

V-JEPA's internal state head supplies autoregressive feedback. Its calibrated measurement head is applied to the same saved feature rollout after prediction. LingBot's reference and generated videos pass through the same frozen measurement pipeline.

Installation, upstream downloads and artifact transport live in `scripts/`. Runtime products live in the ignored `output/` tree. `configs/metadata_layout.json` maps included metadata to its required runtime location. Each original model's mathematical modules are obtained from the pinned upstream revisions, rather than copied into the repository.
