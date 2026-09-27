> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Explicit initial observation context

The development prototype in `scripts/observation_context_prototype.py` separates
public initial-scene assumptions from desired task poses. It addresses the
[descriptor ambiguity](MIXED_TRAINING_FAILURE_DIAGNOSTIC.md): projecting task nodes
to shape/color used to remove an optional orientation hint and change the estimator.
The prototype is not installed in the running frozen replication.

A trusted host declares the initial inventory separately:

```json
{
  "schema": "initial_observation_context_v1",
  "source": "public_initial_scene_description",
  "objects": [
    {"color": "red", "shape": "thin_prism",
     "initial_orientation_assumption": "flat_x"}
  ]
}
```

This field is an explicit assumption, not a measured pose or a goal. The current
bridge scene already declares that the beam initially lies flat along world X.
A future task with different initial conditions must provide its own public
context or use an observer that resolves the initial geometric hypotheses visually.
Never generate this field by copying desired goal orientations or simulator poses.

`bind_session(factory, context)` gives the actor the existing
`Session(api, objects)` call while fixing the observation setup in the trusted
adapter. Requests must describe the same complete public shape/color inventory.
Goal grid, supports, desired orientation and descriptor ordering cannot change
initial model acquisition. The adapter creates fresh canonical descriptors in
context order. Neither caller mutation nor mutation of returned descriptors
changes the bound context. Unsupported shapes or missing assumptions are rejected
explicitly; this prototype has the same four-shape observer coverage as the study.

The initial context schema accepts no object positions, dimensions, contacts or
evaluator data. Dimensions and current poses still come from RGB-D. There is no
new visual estimator, history fusion, action selection, or predicted state in this
repair. The low-level observer still calls its legacy initial-hint field
`orientation`; that field is populated only at the adapter boundary.

## Validation

Three tests cover:

- Invariance to stripped/reordered task descriptors and changed desired poses.
- Copied context, inventory validation, and rejection of metric/goal fields in
  the initial assumption packet.
- Exact center, rotation, dimensions, observability and vertex equality using a
  recorded RGB-D fixture with two cubes, a flat thin prism and a cylinder. Both
  descriptor forms are checked under both numeric and symbolic sessions; their
  numerical fits also match across arms.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests \
  -p test_observation_context_prototype.py -v
```

The recorded-frame test performs no robot action and supplies no object truth.
Its robot-mask interface uses an empty sphere list, as in the existing observation
contract fixture. It checks constructor and observation parity, not closed-loop
success, broad viewpoint coverage, or sensor-only inference of the initial hint.

## Before the next policy comparison

After the current replication finishes, integrate one identical trusted context
binding into both arms and expose the public assumption packet in both actor
contracts. Keep the goal separate. Check all supported scene descriptors and a
physical starter parity rollout before freezing fresh training/reserved splits.
Do not replay just the successful old candidates or reinterpret this unit test as
improved policy learning. A new matched experiment is required to measure that.
