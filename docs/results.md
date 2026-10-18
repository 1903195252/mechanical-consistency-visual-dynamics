# Current results

All reported model results use seeds 42, 43 and 44. The numerical records are included alongside the current manuscript tables.

## Planar resolution and integration

KeyCLD and CON use the same 343-assembly corpus at 64 and 256 pixels. Each setting has 324 test trajectories. CON's image MSE improvement over persistence is 37.6% / 37.2%; native mechanical coverage is zero for both video pathways at both resolutions. KeyCLD exact-state Euler / RK4 scores are 52.65 / 64.30.

The 256-pixel native Scomp values are 22.30 for KeyCLD and 10.96 for CON. Geometric reconstruction raises them to 53.10 / 48.26 while leaving force, energy and point errors above target. CON retains native dense image layers, expanding total parameters from 67.3M to 1.08B at a fixed 2,550-update budget.

Per-case metrics and aggregation are in [resolution_comparison.json](../manuscript/keycld_cvpr/sources/resolution_comparison.json). Resolution-independent controls are in [unified_scoring.json](../manuscript/keycld_cvpr/sources/unified_scoring.json).

## Shared spatial test cohort

The two spatial models share 50 Stewart and 50 Delta test assemblies, each with three conditions. V-JEPA + AC has 900 forecasts; LingBot has 1,800 videos across 0 / 50 updates.

| Family | LingBot native Scomp, 0 → 50 | CAD Scomp, 0 → 50 | CAD coverage, 0 → 50 |
|---|---:|---:|---:|
| Stewart | 5.67 → 6.01 | 54.60 → 54.31 | 450/450 → 450/450 |
| Delta | 2.80 → 3.39 | 43.37 → 51.92 | 359/450 → 424/450 |

LingBot's full-image MSE falls by 12.6% / 2.1% after adaptation, while motion-mask MSE rises by 4.0% / 7.2%. At 50 updates, matched reference/generated effort residuals are 3.01 / 40.51 N for Stewart and 0.733 / 12.32 N·m for Delta. Energy defects are 0.605 / 1.640 J and 0.203 / 1.206 J.

V-JEPA's calibrated-readout native / CAD scores are 6.20 / 55.11 for Stewart and 10.40 / 50.05 for Delta. CAD coverage is 450/450 and 432/450. Matched reference/generated effort residuals are 3.11 / 41.97 N and 2.08 / 12.42 N·m.

Current records:
- [LingBot per-seed summary](../results/spatial/lingbot/summary.json)
- [V-JEPA summary and readout controls](../results/spatial/vjepa/summary.json)
- [Continuous score records](../results/continuous/scores.json)

Scomp is the mean of six scores `100 / (1 + (error / tolerance)^2)`. Missing measurements contribute zero and keep their denominator. A score of 50 is the value when every error equals its tolerance; an aggregate score above 50 does not imply that every criterion passes.
