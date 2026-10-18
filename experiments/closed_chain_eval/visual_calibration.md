# Independent visual measurement

Calibration uses known geometry and reference sequences with assembly-separated fitting and verification. The registered pipeline applies the same frozen reader and differentiator to reference and generated RGB. Reference replay is reported beside model residuals, rather than subtracted from them.

Planar RGB readers live in `planar_resolution/reader.py`, with per-resolution calibration, coordinate refinement and measurement entrypoints in the same package. Spatial readouts combine frozen V-JEPA features with calibrated CAD-coordinate heads. Delta image-space refinement and temporal differentiation are registered in `configs/spatial_measurement.json`.

For V-JEPA + AC, the internal state head supplies autoregressive feedback. The calibrated head is applied afterwards to the same saved features. For LingBot, reference and generated RGB use the same independent measurement pipeline. Raw coordinates and CAD/IK reconstructions are retained as distinct measurement branches with a reported coordinate correction cost.

The controls include exact states, sampled-state derivatives, correct-video replay, fixed-feature head replacement, and geometric reconstruction. Current per-case records and aggregation checks are included in `results/`; run `./run.sh verify` to check them.
