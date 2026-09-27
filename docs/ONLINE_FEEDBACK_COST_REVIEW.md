# Online feedback cost review

This development-only review uses existing observations and local sensor readouts.
It launches no model requests and executes no manipulation actions. Historical
reserved outcomes and frozen runtime code remain unchanged.

## Repeated sensing is measurable but small

The current runner collects retry evidence for every object before preparing a
model request. Full then calls the geometry packet builder, which takes another
snapshot. With initial side and wrist images, one examined decision preparation
made 12 camera acquisitions. A scoped cache reused same-view frames while sensor
time was unchanged, reducing the count to two acquisitions and ten reuse hits.
The prototype is `scripts/online_decision_frames.py`; it is not connected to the
registered runner. Four unit tests verify view separation, time invalidation,
copy isolation, exception cleanup and readouts that advance time.

Three action-free checks on the existing development layouts failed strict output
equality. A diagnostic on development layout 2028100101 saved the compared states:
its geometric feedback and retry evidence matched, but the second repetition's
side RGB image did not. The difference affected 5,975 of 76,800 pixels, with a
maximum channel difference of three on the 8-bit scale. Wrist images matched.
The cause of this render difference is not established. It must not be silently
dismissed as numerical noise or called exact observation preservation.

In that diagnostic, uncached decision preparation took 0.119 and 0.115 seconds;
cached preparation took 0.108 and 0.109 seconds. These are two local measurements,
not an estimate of agent latency improvement. The saving is small compared with
multi-second model requests and approximately 131-second successful episodes.
The cache is therefore **not adopted**. All failed checks and diagnostic states
remain in `outputs/online_feedback_cache_dev`; `assessment.json` records the
conclusion and source hashes. No source frozen by earlier studies was changed.

## The current ordinary task often requires only an initial plan

The shared executor observes objects, binds supports, computes placement targets
and performs physical corrections. In the completed ordinary pair, each agent
issued one initial plan, after which local execution completed the task. There
were no repeated online model decisions in which Full could substitute text for
new image interpretation. Both arms received initial images, and Full additionally
received geometry. This explains why that pair cannot isolate the hypothesized
benefit of repeated text-based reasoning. It does not prove that the benefit is
absent in other online settings.

Shrinking JSON or caching a few frames cannot create a missing reasoning demand.
Do not add failures, remove successful Base episodes, or force extra calls only in
Base to obtain a positive result. The current system-level comparison should keep
its measured outcome.

## A separate, simple test would need an explicit spatial decision

A future representation-focused comparison could retain ordinary blocks and the
same trained skills, but make the LLM bind a current placement target before each
placement. Both arms would receive the same action schema and decision boundaries:
Base would derive its target from calibrated visual evidence, while Full could
query current geometric references. The actuator would execute the submitted
binding and apply the same validity checks, without silently substituting the
correct goal. This would test a different interface from the current node-only
skill calls and must be registered as a separate experiment.

Such a change should be evaluated for correctness on development observations
before any new reserved rollout. It would still have a single continuing scene,
no injected collapse, and natural failures within a fixed budget. It should not be
launched merely because the existing result was negative. An initial-plan-only
system comparison and a per-placement spatial reasoning comparison answer
separate questions; the latter is not a re-interpretation of earlier results.
