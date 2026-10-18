# Mechanical consistency evaluation

The six SI criteria measure closure, tangent velocity, curvature acceleration, reaction-free actuator effort, work–energy balance, and point tracking. `protocol.json` defines the fixed engineering tolerances. The primary score is `continuous-si-1`: each measured tolerance ratio receives `100 / (1 + ratio**2)`, followed by balanced aggregation. Unavailable criteria contribute zero with all registered cases retained. Physical coverage and strict joint qualification are reported alongside the score.

Reference states, sampled-state derivatives and reference RGB replay quantify different parts of the measurement chain. Generated and reference videos share frozen readers and differentiators. CAD/IK reconstruction is an explicit intervention with a reported coordinate correction cost.

See [reproduction](../../docs/reproduction.md), [architecture](../../docs/architecture.md) and [results](../../docs/results.md). Run `./run.sh verify` for numerical checks. `evaluate.py` accepts a registered physical prediction/reference pair; the planar and spatial runners use the same continuous scoring layer.
