# Continuous SI scoring

For each registered trajectory and criterion, let `rho = error / tolerance`. The criterion score is `100 / (1 + rho**2)`. The primary `Scomp` is the arithmetic mean of the six criterion scores, balanced across conditions, mechanism families and seeds. Unavailable measurements contribute zero; registered failures remain in the denominator. A composite score of 50 is not a claim that all six criteria pass. Strict joint qualification requires every criterion to be measurable and within its tolerance.

Engineering tolerances specify the physical target. Measurement configuration registers the camera, resolution, coordinate readout, derivative estimation and bandwidth separately. Pixel spacing informs localization calibration; it does not replace the mechanism closure tolerance. Spatial effort and energy use the registered actuator and command scales.

The released planar experiments use 64- and 256-pixel inputs with matched field of view, splits, frames and native model objectives. Spatial models share 50 Stewart and 50 Delta test identities and three operating conditions. Reference states, sampled-state derivatives and correct-video replay quantify measurement errors. Fixed-feature readout controls preserve the forecast trajectory; CAD/IK interventions report the correction needed to recover physical geometry.

`continuous.py` provides the scoring implementation. `rescore_continuous.py` aggregates registered SI measurements. See [reproduction](../../docs/reproduction.md) and [numerical evidence](../../docs/results.md).
