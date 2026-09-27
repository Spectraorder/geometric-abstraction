# Package boundaries

Current experiments are launched by `scripts/run_constraint_geometry_experiment.py`
and `scripts/run_constraint_geometry_replications.py`, not by an in-package actor loop.
The obsolete actor/context/skill-promotion modules are in `archive/legacy_training`.

The remaining modules include shared dependencies, not 51 independent skills:

- **Visual evidence:** RGB-D calibration, robot masks, cube/cuboid/column/beam fitting,
  cuboid correspondence and the mixed PnP observer.
- **Geometric interface:** metric/symbolic scene state, rigid goals and support geometry.
  The compact per-target recorder and public sessions are in `policies/`.
- **Execution and evaluation:** robot geometry, playback, simulator environments,
  task definitions and scene generation. Evaluators remain private to the host.
- **Historical dependencies:** older controllers, perception diagnostics and demos
  still referenced by builders, tests or replay tools. Their names do not imply
  that the current actor receives their policies or state.

Builders also read Python files as text and select AST nodes, so an import-only
unused-code scan is insufficient. Remove a helper only after checking these paths.

Historical manifests hash all top-level Python modules. Source cleanup intentionally
changes that inventory; audit old experiments using their pre-cleanup checkout.
Never update a frozen manifest to disguise a source change. The archive README
records the recovery commit. New experiments require a fresh registration.
