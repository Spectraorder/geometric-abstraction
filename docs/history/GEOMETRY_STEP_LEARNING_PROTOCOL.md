> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Target-only geometry abstraction learning pilot

Status: completed in the corrected root: 200 valid rollouts and six audited completed
actor contexts. [Full results](GEOMETRY_STEP_LEARNING_RESULTS.md): selected Base 80%,
Full geometry 70%; no iteration advantage established.


Compare **Base** with **Full geometry**, using the same mixed RGB-D observer, robot
masking, initial perception assumptions, CPU policy execution and PnP motion loop.
Only the supplied geometry interface/feedback differs. No numerical-only arm.
The base already has surface geometry; this tests the added abstraction package.

Task: the existing low-rise bridge with two cube foundations, a flat thin prism and
a cylinder cap. Four unique colors are cyclically permuted; geometry comes from
calibrated side/wrist RGB-D, never simulator object poses, dimensions or contacts.
Public task labels and robot CAD/encoders are allowed. Fixed public initial orientation
assumptions are separate from desired orientations and sensor evidence.

Register five fresh training seeds 930001–930005 and twenty fresh reserved seeds
940001–940020. The root manifest is authoritative. One isolated `gpt-6-astra` actor
chain per arm, common R0 starter, three revisions, equal model settings and rollout
budgets: six model calls and 200 rollout slots. All eight candidate versions freeze
before any reserved rollout. Select by highest training SR, earliest tie. Never
choose or repair a candidate using reserved results. Invalid candidates consume their
revision and slots. No prior evolved policy or inspected failure analysis enters
actor contexts. Tokens are recorded; rollout duration is not an objective.

The starter preserves the original anchor/support planning and the same pick/place
commands. It replaces global snapshots with current-object checkpoints before pick,
after pick/place, and after settling. The checkpoint reuses the current observation;
only explicit `refresh=True` senses that object. It never refreshes the whole scene.
Dimension initialization still observes the public inventory once. Support queries
remain explicit. The full checkpoint includes current geometry, optional desired
center/error, requested active-object points and one-step estimated change. It does
not certify contact or stability or carry hidden poses through missing evidence.
Desired centers use the existing target store rather than an independent goal model.

Both arms receive their own outcome/error/note histories, common executor source,
latest candidate and the same deterministic example/image selection rule. Per selected
example, attach only its last eight `geometry_step` records, capped at 16000 serialized
characters by removing whole oldest records. No global scene packets. The prompt has
the existing 120000-character hard cap. Global snapshot/state/text/history calls are
excluded by the planner validator in this version; explicit single-object observation
and geometry queries remain available. The restriction applies equally to both arms.

Registered primary metrics: reserved SR at every fixed revision, first revision
reaching 80%, mean reserved SR across R0–R3, and training-selected reserved SR.
Secondary: at each revision budget retain the highest training SR available, earliest
tie; evaluate that revision without reserved selection. This is not counterfactual
parent-retention evolution. Report every revision, including regressions and failures.
A single chain and twenty layouts cannot prove a repeatable learning benefit; any
promising advantage needs independent actor replication.

Before model calls, paired R0 trials must have byte-identical physical trajectories,
matching camera counts and matching focus estimates (display precision 1 micrometer).
Audit exact prompt/image reconstruction, runtime/library/source hashes and global
freeze timing. Preserve the preceding completed base/full pilot, including its R3
regression. This experiment tests the compact design; its character reduction alone
does not imply improved SR or lower reasoning latency.

Implementation: `scripts/run_geometry_step_experiment.py`; independent result audit:
`scripts/summarize_geometry_step_experiment.py`. The runner reuses the existing
evolution/rollout harness under scoped hooks and does not modify historical studies.


## Validator correction before reserved evaluation

The original `outputs/geometry_step_learning_pilot_01` is withdrawn for a host
validator defect, not for an unfavorable physical result. Base R2 named the cap
object `load`. The generic AST guard rejected every use of that ordinary local
variable as if it were a file-loading operation, producing five invalid slots
without executing physics. Both R3 model calls had begun from the resulting feedback;
they were interrupted before producing complete outputs. No reserved trial existed.

The correction is `outputs/geometry_step_learning_pilot_01_corrected`. Only
`run_vision_trial.validate_source` changed. Numerical I/O operation names may be value
variables, while prohibited calls, attributes and alias imports remain rejected.
The exact submitted base R2 source passes unchanged. Two focused regression tests
and eleven geometry/checkpoint/protocol tests pass. Controller, observer, library
bytes, starter, task/seeds and feedback contract remain identical.

The correction retains 25 valid trials, four complete actor generations and 2703
artifact files byte-for-byte. Five old invalid records remain solely in the withdrawn
parent. Only the previously unexecuted base R2 trials and subsequent R3 work run in
the corrected root. The independent auditor checks the exact source diff, retained
artifact hashes, original model contexts, unchanged experimental settings and the
absence of parent reserved trials. The earlier completed shared-observer study also
still audits from its archived runtime.

Accounting includes the two interrupted R3 invocations; they are not hidden free
retries. A completed corrected run will contain six complete retained/new model
generations plus these two interrupted attempts, and 200 valid-or-invalid registered
slots. Preserve all failures and this correction in any workshop report. The old
R2 invalid slots must not be presented as a geometry success-rate advantage.


## Completed training and pre-reserved diagnosis

Corrected training completed with 40 valid rollouts, zero invalid slots and all six
exact actor contexts verified. Successes over five layouts are Base [3,4,4,1] and
Full geometry [3,4,4,0]; the training-only selection rule chooses R1 for both.
All eight candidate versions were frozen before reserved execution. Reserved
outcomes were left unread until the complete evaluation became available.

The R3 training records show both policies first picking the cap in every layout.
Their proposed source-clearing intervention added a cap relocation before building.
New failures included cylinder observation loss, unconfirmed placement and one
unreachable tool request; these are actual valid-policy rollouts. This differs from
the preceding large-state pilot's no-motion R3 rejection. It does not establish that
the compact abstraction caused the regression or that source interference was the
original beam failure's cause. Preserve all revisions and select by training only.
The action/observation evidence is `training_r3_regression_diagnostic.json` in the
corrected experiment root. Completed payload measurements are recorded separately
in `training_context_footprint.json`; reduced text volume is not proof of SR benefit.

A possible future state extension would be one explicitly conditional held-object
transform relative to robot geometry, for reasoning through temporary occlusion.
It would require independent visual validation and must remain separate from current
sensor observations. No such prediction is enabled in this frozen experiment, and
no global pose history is justified by these failures alone.


## Post-study development: placement intent at checkpoints

The current development interface additionally records the last attempted placement
for the active object. `place(surface_xyz)` and `pick_and_place(color, surface_xyz)`
retain one target point; a new pick clears it. This does not alter controller commands
or automatically sense, retry or correct a placement. Explicit center goals remain
independent and are never overwritten by placement intent.

The full checkpoint's `placement_goal` has reference `center_xy_min_z`: center X/Y
and the estimated lowest world Z. This matches the executor's actual surface-point
contract, and is not the center of a material bottom face when tilted. Its signed
`remaining_translation_m` is target minus the current estimated extent point. It
expresses the desired settled placement, not a safe next gripper command: the common
executor intentionally uses release clearance. Missing or stale geometry leaves the
residual unknown. Geometry observability flags still apply; zero residual does not
certify contact, attachment, stability or success. Base retains its original surface
measurements and common executor status/error fields.

This addition was made **after** the frozen 200-rollout learning study. It was not
available to those actors and must not be credited with their results. Use archived
source snapshots to reproduce that study. Development evidence is recorded in
[evaluations/placement_feedback_development.json](../evaluations/placement_feedback_development.json).
Thirteen checkpoint/interface tests pass, including missing evidence, failed place,
intent reset, no implicit sensing, and preservation of explicit center goals.

On development seed 75103, Base and Full both complete the bridge with identical
physical trajectories and 54 camera calls. The full arm records six placement goals
within twelve checkpoints; its maximum compact record is 608 characters. A third
rollout using the frozen pre-change full library, identical starter and identical
scene/color mapping also has byte-identical physical arrays. The older development
example used different colors and is not a valid before/after control. No actor calls
or reserved evaluation were run for this development check.
