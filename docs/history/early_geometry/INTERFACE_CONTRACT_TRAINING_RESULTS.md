# Five-round training with the shared interface contract

Training completed in approximately 37 minutes 34 seconds, selecting **Base R4
at 3/5 (60%)** and **Full R4 at 4/5 (80%)**. The durable coordinator finished while
the interactive conversation was interrupted. No actor attempt failed because of
usage exhaustion, and no additional call or training rerun was needed on resumption.
Reserved evaluation has not started.

| Revision | Base training | Full training |
| --- | --- | --- |
| R0 | 0/5 | 0/5 |
| R1 | 2/5 | 3/5 |
| R2 | 1/5 | 3/5 |
| R3 | 2/5 | 3/5 |
| R4 | 3/5 | 4/5 |
| R5 | 0/5, invalid source length | 0/5, invalid source length |

The guard retained R1 through R3 in both arms, rejecting Base's R2 regression and
all ties. R4 repaired seed 2026092603 without losing existing successes in either
arm. Base R4 succeeds on 2603, 2604 and 2605; Full R4 additionally succeeds on
2601. Both still fail 2602. The R4 policies remain selected despite invalid R5
proposals. R5's zero scores describe invalid proposal slots, not a measured loss
of physical performance by the retained policies.

## Interface and source-budget outcomes

The prior module-level-assignment and checkpoint-status-type errors did not recur
in any recorded training error or note. Both arms produced executable R1 policies.
This observation is consistent with the intended benefit of the shared contract,
but one new chain per arm cannot isolate the prompt change from generation variance.

R5 exceeded the unchanged 30000-character source limit: Base generated 32352
characters and Full generated 33273. Their R4 sources were already 28574 and 27580
characters respectively. All ten R5 training slots were recorded as invalid
candidate failures without simulation; no manual edit, free rewrite or truncation
was inserted. Their generation costs remain included.

For a future registration, explicit source-length headroom and a bounded,
equally budgeted validation/repair mechanism could address this remaining failure
mode. Those changes have not been applied to this completed study. More rounds
alone do not prevent source growth or guarantee an improved executable policy.

## Resource accounting

| Model resource across R1–R5 | Base | Full |
| --- | ---: | ---: |
| Input tokens, including cached input | 295,536 | 298,739 |
| Cached input, subset of input | 206,720 | 198,528 |
| Output tokens, including reasoning | 60,657 | 58,347 |
| Input + output tokens | 356,193 | 357,086 |
| Sum of generation wall time | 1,851.3 s | 1,573.8 s |
| New images by revision | 4, 2, 2, 2, 2 | 4, 2, 2, 2, 2 |
| Tokens through R3 | 149,738 | 150,650 |
| Added R4–R5 tokens | 206,455 | 206,436 |
| R5 tokens, invalid candidate included | 112,700 | 112,597 |

Usage is computed from verified differences between cumulative session counters.
Cached input and reasoning are already included in their parent categories. Calls
ran concurrently by revision, so summed generation times are not experiment elapsed
time. Full used approximately the same total tokens as Base; this run does not
show a total-token saving. R4 added one training success per arm; R5 added cost
without an accepted policy. These are training-only observations, not held-out
learning curves or a causal comparison with the previous three-round study.

## Validation and evidence

A fresh audit on resumption passed: 60 registered training slots comprise 50
executed rollouts and ten invalid candidate slots. All ten completed model turns,
exact actor contexts, hypothesis reviews, source hashes, baseline parity, public
goal contracts, shared interface contract and training-only policy selection were
verified. Recomputed resource accounting exactly matches the coordinator's saved
report. No reserved results exist and no new model call was made during review.

Evidence is under `outputs/geometry_interface_contract_01/`:

- `training_lifecycle.json`, `training.log`: completed execution and elapsed time.
- `training_summary.json`, `training_review.json`: complete curves, retention and failure accounting.
- `audit_training.json`: verified training provenance and budget.
- `resource_accounting.json`: per-turn and cumulative verified model usage.
- `basic/frozen.py`, `symbolic/frozen.py`: selected R4 policies.

The 20 reserved seeds 2027012701–2027012720 remain unused. The existing standard
full-curve evaluation registers 240 test slots (R0–R5 for each arm); R5 would
contribute 40 invalid slots. Testing just the two selected policies requires a
separate 40-slot protocol. See the
[experiment guide](../../INTERFACE_CONTRACT_EXPERIMENT.md) for configuration details.
