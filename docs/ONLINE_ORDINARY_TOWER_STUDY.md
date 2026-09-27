# Ordinary online bridge-tower comparison

This small follow-up studies composition of the shared bridge and stacking skills
from ordinary unassembled scenes. It excludes externally injected disruption.
Natural failures remain in the same scene until success or timeout. The purpose
is to test whether geometric text improves reasoning efficiency; a positive effect
is a hypothesis, not a condition for retaining the results.

## Fixed scope

- Five new shared seeds, 2028103001–2028103005; one Base and one Full episode each.
- Ten episodes total, with a 180-second wall-clock budget per episode.
- Balanced batches of two complete pairs, two complete pairs, then one pair;
  at most four workers, with one BLAS/OpenMP thread per worker.
- Same bridge/stack skill experience, sensor-based executor, jaw-clearance workspace
  screen, action segmentation, and Astra medium planner in both arms. Ordinary
  successful queued operations continue locally without further model decisions.
- Base receives images and qualitative status. Full receives initial images and
  geometric text, with further images available on request. Both have the same
  public goal and can request observations.
- One reset at initialization. The clock starts before workspace initialization
  and includes perception, model requests, physical execution and settling.
  Simulation continues during model waits at one-to-one pacing.
- No artificial collapse, loaded-stack displacement, manual rescue, seed
  replacement, repeated attempts or automatic budget extension.

Ordinary does not mean preselected for success. Layouts are fixed without screening
their outcomes. If a layout causes a natural collision or recovery failure, it
remains in the comparison. Development layouts 2028100101–2028100103 are excluded.
The three accelerated physical checks that passed with the shared clearance screen
are readiness evidence only; they did not exercise online model latency.

## Evidence and interpretation

The corrected private evaluator checks the complete support graph and stability
for five simulated seconds, with at most 3 mm drift. Robot contact exclusion uses
geometry names and owning-body ancestry. Initial physics hashes verify pairing;
terminal model/state archives support later private contact reconstruction. None
of these evaluator records are available to the model or skills.

Report success out of five and mean capped completion time for each arm, counting
all unsuccessful episodes as 180 seconds. Show every paired outcome and time
change. Also report input/output/cached tokens, model requests, incomplete requests,
image attachments, first-request latency, cumulative completed-request time,
camera acquisitions and action time. Cached input is already part of input tokens.
Missing request usage remains unknown and is not replaced by zero. Campaign wall
time is separate from episode latency. A simulation lag above two seconds prevents
an unqualified real-time speed claim; retain and flag the affected episode.

Five pairs give an exploratory result. Report equal or worse outcomes as measured;
do not expand the study until a favorable result appears. Shared improvements to
workspace planning do not establish an advantage of the Full text interface.

## Running and preservation

The dedicated runner is `scripts/run_online_ordinary_study.py`. Preparation copies
runtime sources/assets, hashes the generated geometry and development evidence,
and records packages and the model client version. It makes no model requests.
Use a new output directory for preparation. The runner refuses existing attempts
and stops new batches after an infrastructure failure; in-flight workers finish.
Source checks run before dispatch and after execution. A source/resource audit and
paired summary run after all ten episodes finish.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/run_online_ordinary_study.py \
  --root outputs/online_ordinary_tower_01 --phase prepare

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/run_online_ordinary_study.py \
  --root outputs/online_ordinary_tower_01 --phase check
```

`--phase run` launches the ten registered episodes. Preparation or a source check
does not launch them. The original online pilot and controlled-recovery protocols
remain unchanged, including their recorded failures and evaluator limitations.

## Recorded outcome: stopped after the first batch

`outputs/online_ordinary_tower_01` stopped after four attempted slots. The scheduler
preserved both initialization errors and allowed the other in-flight pair to finish;
the remaining six slots were not dispatched. The protocol and frozen-source check
still pass. Do not resume this registration or replace its failing layout.

| Layout | Base | Full | Base / Full completed tokens |
| --- | --- | --- | ---: |
| 2028103001 | Success, 130.98 s | Success, 131.43 s | 11,614 / 11,918 |
| 2028103002 | Initialization error | Initialization error | 0 / 0 |
| 2028103003–2028103005 | Not dispatched | Not dispatched | Not incurred |

Each successful episode used one model request. First-request latency was 5.04 s
for Base and 5.89 s for Full. Both initialization failures occurred before any
model request, with `Visible samples contradict initialized shape bounds` in the
shared conditional planning-bound constructor. The recorded generic status is
`infrastructure_error`; the underlying failure is geometric initialization, not
model reasoning. This observation alone does not establish which bound assumption
or surface samples caused the contradiction. No failure layout was rerun or tuned.

Both attempted layout pairs have matching initial physics hashes and one reset per
episode. The maximum recorded lag in the successful pair was below 0.94 seconds.
The source/resource audit passed for all four attempts, with no incomplete model
requests. Independent reconstruction of all four terminal physics archives found
no robot/object contacts. This independently checks terminal contacts; continuous
stability still relies on the recorded private monitor.

There is no demonstrated Full advantage in this partial result. The successful
pair was mostly shared physical execution after one initial plan, so it did not
exercise repeated text-based online reasoning. This limits what the simplified
experiment can establish about the hypothesized benefit. It is not a reason to
introduce artificial difficult scenes or to discard the result.

The total completed model usage was 23,532 tokens across two requests. Unstarted
slots have no observed outcomes and must not be counted as failed or successful
episodes. The attempted-only capped means are 155.49 s for Base and 155.72 s for
Full; these include the initialization errors at the fixed 180-second cap and are
not five-layout estimates. Full artifacts are `audit.json`, `partial_summary.json`,
`STOP.json`, and `analysis/summarize_partial.py` under the study output.
