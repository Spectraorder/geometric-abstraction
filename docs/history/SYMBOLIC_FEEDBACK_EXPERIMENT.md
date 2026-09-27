> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Symbolic feedback with a frozen executor

## Completed pilot: no demonstrated learning-efficiency benefit

Both conditions reached 4/5 training success on revision 1 from a common 2/5
baseline. Selected policies succeeded on 5/10 reserved layouts for raw and 6/10
for symbolic feedback. There was one symbolic-only success, five shared successes
and four shared failures; exact paired McNemar p=1.0. This small amended pilot does
not establish a meaningful success-rate or iteration-efficiency advantage.

| Stage | Raw | Additional symbolic/metric feedback |
| --- | --- | --- |
| Shared baseline | 2/5 | 2/5 |
| Revision 1 | 4/5 | 4/5 |
| Revision 2 | 3/5 | Rejected: source over budget |
| Revision 3 | Rejected: source over budget | Rejected: source over budget |
| Selected revision | 1 | 1 |
| Reserved layouts | 5/10 | 6/10 |

The evidence includes 40 actual rollout records: five shared baseline trials,
15 executed candidate trials and 20 reserved trials. Fifteen additional assigned
slots were consumed by code-validation failures, not physical rollouts. Baseline
records and both unedited first proposals were reused from the original run; its
extra executed symbolic trials and discarded second requests are retained outside
this comparison. See the protocol amendment below.

Actor output tokens totaled 18,235 for raw and 23,611 for symbolic across the three
included proposals. These data provide no token-efficiency benefit either. Source
budget failures and the feedback-only treatment limit interpretation. Neither the
small reserved difference nor the earlier manual 5/5 development probe proves the
research objective.

[Measured report](../evaluations/symbolic_feedback_pilot_02.json),
[provenance audit](../evaluations/symbolic_feedback_pilot_02_audit.json), and
[learning-curve figure](../media/symbolic_feedback_pilot_02.png) retain the outcomes.


The core representation is typed symbolic entities and references plus metric
geometry. Text and JSON serialize that representation for the policy-writing
agent. Geometry predicates retain unknown states and residual distances rather
than reducing uncertain perception to unconditional Boolean facts.

This pilot asks whether supplying sensor-derived symbolic/metric feedback helps
an agent improve a high-level planner in fewer revisions. It does not test the
entire proposed persistent-state system at once.

## Frozen protocol

Runner: `scripts/run_symbolic_policy_experiment.py`.
Campaign: `outputs/symbolic_feedback_pilot_02` (corrected pilot).

- Both arms start from the same task adapter and shared RGB-D PnP implementation.
  Only high-level planner code changes. The common executor source, simulation,
  perception and protocol sources are hashed before running.
- Each arm receives three independent policy-writing turns with its own history,
  five training layouts (52001–52005) per revision, the same model/settings, and a
  14,000-character planner limit. Invalid candidates consume revision slots.
- Both arms see final success, execution errors, their own diagnostic notes, and
  the first/last side RGB-D observations for the first two training layouts.
  RGB and linearly encoded depth images plus calibration are supplied to both.
  Both can access full numerical RGB-D at runtime.
- The symbolic arm additionally receives compact geometric state and estimated
  support relations computed from those exact RGB-D frames. Initial on-table
  dimensions are reused for later instantaneous fits. No temporal pose memory,
  attachment prediction, private contacts or object-truth state is supplied.
- The raw arm can implement its own numerical perception and high-level reasoning.
  Neither arm can replace or tune the common actuator primitive. Validation
  rejects direct actuation, internal primitive access, attribute assignment,
  classes and protected-name replacement. Generated code is also archived for
  audit; this experimental boundary is not a hostile-code security guarantee.
- Select highest training success with earliest-revision tie breaking. Freeze
  both candidates before evaluating either on reserved layouts 53001–53010.
- Preserve raw prompts, exact model responses, token usage, frame hashes, every
  rollout and unsuccessful proposals. No human repair or solution insertion.

The shared primitive was developed on different scenes. A manual workspace
planner achieved 5/5 development towers, showing the common primitive can solve
this task under suitable planning. That planner and its evidence are withheld
from both actors. This is not agent-learning evidence.

## Interpretation

Compare success by revision, revisions/rollout slots to a prespecified 80% training
threshold, and selected-policy success on the paired reserved layouts. Report
model tokens separately; equal revision counts do not imply equal token counts.
A one-campaign pilot cannot establish a general learning-efficiency benefit.
Replicate promising differences with independent model contexts and new seeds.

The treatment supplies metric text in addition to matched RGB-D images. This
measures the practical value of preprocessing raw visual information into an
agent-readable representation. It does not isolate symbolic relation names from
numerical pose estimates, or prove memory/prediction is useful. Follow-up ablations
should compare metric-only text, metric-plus-relations, and timestamped memory
under equal observation/feedback budgets. Runtime reasoning speed is not a primary
outcome of this experiment.

## Separate offline perception diagnostic

The five shared baseline rollouts provide 30 initial/final object-center estimates.
Scoring sensor-derived estimates against time-aligned recorded simulator states
(after fitting, without feeding truth back to the actor) gives median center
error 0.098 mm and 95th percentile 2.531 mm. The 20 fully observed estimates have
median 0.096 mm and maximum 0.151 mm; the 10 partial estimates have median 1.716 mm
and maximum 2.762 mm. These are ideal simulated depth observations, not a claim
about real sensors, moving-frame accuracy, vertices, memory or policy learning.

The [private offline diagnostic](../evaluations/symbolic_feedback_initial_accuracy.json)
is explicitly outside all actor inputs. Its script reconstructs each scene only
after computing the complete sensor-derived feedback packet. It aligns stored
physics states by sensor timestamps and does not tune the geometry estimator.

## Validator correction and exact-candidate reuse

Pilot 01 had an overly broad AST check rejecting `nonlocal`, including an ordinary
numeric counter inside the raw actor's planner closure. That restriction was not
part of the actor contract and did not protect the frozen executor. The process
had already recorded invalid raw slots, executed the symbolic candidate and begun
both second proposals before the validator patch landed. Buffered output initially
obscured that ordering; file timestamps and the corrected amendment record retain
the actual sequence. Pilot 01 was stopped without reserved evaluation.

Pilot 02 uniformly permits numeric closure state. It retains both exact initial
proposals and the identical shared baseline evidence, then executes both candidates
under corrected validation. Neither candidate is patched. Both old second proposals
are excluded because they were conditioned on incompatible feedback. All original
files are retained, including the initial and corrected timing notes. Pilot 02 is
exploratory, with disclosed protocol amendment, not a preregistered confirmation.


## Interim corrected pilot result

The shared baseline is 2/5. Revision 1 is 4/5 in both conditions, with the same
successful layouts and the same failing layout. Both generated planners implement
their own RGB-D face-based center estimator and placement recovery. Thus the first
revision shows policy improvement but no supplied-feedback learning advantage.
Do not use the invalidated pilot 01 raw validation failures as a control score.

A limitation of this feedback-only treatment is that it supplies compact state
between policy revisions, while both actors still need to implement any improved
perception they want during runtime. A later independently frozen experiment can
supply the action-free symbolic/metric API online, with the same PnP executor and
control freedom, to test the full intended architecture. That would be a different
treatment and must not be merged with this pilot's learning curve. Persistent
memory and conditional prediction then require their own component ablations.


Revision 2 produced 3/5 executed successes for raw. The symbolic actor submitted
15,267 characters against the common explicit 14,000-character planner limit;
its five slots are recorded as candidate-validation failures, with zero executed
rollouts. This is an agent-budget failure, not a measured 0/5 physical-controller
result. The plot therefore labels allotted rollout slots and marks rejected code.
Unlike the earlier undocumented `nonlocal` defect, this limit was stated equally
in both actors' contracts and is retained without a free rewrite.
