# Geometry and runtime package

Task controller sources live in `policies/controllers/`. Their standalone policies
are generated on demand; do not copy them or their bundled helpers back into `src`.

Keep these reusable capabilities here:

- RGB-D calibration, robot masking and cube/cuboid/column/beam estimation.
- Metric state, conditional prediction/fusion, symbolic relations and rigid goals.
- Gripper/robot geometry, shared execution and simulation/task infrastructure.

Compact target records and the shared PnP/session adapters live in `policies/`.
`scripts/build_geometry_session.py` constructs the public geometry interface
directly, independently of experiment-driver configuration. It retains the
relation/goal operators and uses the same actuator for both comparison arms.

A capability's presence does not prove its learning benefit. Prediction and
relation APIs remain available for research; the controlled results measure the
whole supplied geometry package. Old experiment outputs are evidence, not a reason
to retain duplicate source or generated policy snapshots in the active tree.
