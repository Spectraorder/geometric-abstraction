# Online evaluation: extension to 50 paired layouts per task

The user authorized 50 paired layouts **including the original five** for each
of Bridge-tower, Ring-on-peg and Tower relocation. This adds 45 paired layouts
(90 episodes) per task: **270 new episodes, 300 in the combined dataset**.
The original five-pair studies and their exports remain unchanged. The expanded
[result report and figures](evaluations/online_50/README.md) are stored separately;
the current paper fragments now reference that expanded export.

This is a post-hoc extension prompted by reviewer feedback after the pilot results
were known. New layouts measure robustness to layout variation. One run per
condition/layout does not measure repeated-run stochastic variability, and the
Base/Full comparison still evaluates the whole geometry-and-feedback package.
Separating geometry estimation from representation requires another experiment.

## Frozen conditions

| Task | Original study | New layout seeds (inclusive) |
| --- | --- | --- |
| Bridge-tower | `online_compact_tower_200_01` | 2035101001–2035101045 |
| Ring-on-peg | `online_ring_200_01` | 2035102001–2035102045 |
| Tower relocation | `online_relocation_200_01` | 2035103001–2035103045 |

The runner reuses the original episode functions, prompts, feedback, sensor
interfaces, skill libraries, scene generators and physical success checks. Ring
copies its already selected grasp/lift prior; no policy is reselected. Both arms
use Astra with medium reasoning and a fresh persistent context per episode.
Tower retains its original default-provider routing; Ring and Relocation retain
HTTP-only routing. Runtime perception excludes simulator object truth.

Each episode has a 200-second wall-clock budget and one initial reset. The clock
includes initialization after reset, sensing, model waits, skill execution and
settling, including Ring's shared grasp/lift. Physics continues during model waits.
There are no injected disturbances. Successful actions do not reset the scene;
failed actions retain their physical consequences. The original hold criteria
and a maximum two-second simulation lag remain enforced.

The installed model client has changed from `0.155.0-alpha.16.3` to `0.159.2`.
The extension pins a copy of the latter executable and records its hash. All five
simulation package versions match the pilots (including MuJoCo 3.11.0). Runtime
source parity is checked against each pilot; the only permitted changed original
file is the unrelated method-figure renderer. Client version and service date
are cohort differences, even though the requested model and reasoning setting
are unchanged. Report the new cohort separately alongside pooled results.

## Execution and evidence

Use `scripts/run_online_extension_study.py`, with the new root
`outputs/online_extension_50_01` on the SSD. `prepare` freezes source hashes, pilot
evidence, the client binary, priors, seeds and all 270 slots before any dispatch.
It checks new seed ranges against prior registration and manifest files.

The coordinator runs **at most four episodes globally**, with complete Base/Full
pairs in each batch and tasks interleaved. Arm launch order alternates by pair.
Each worker uses one BLAS/OpenMP thread. Interface, token, prompt, initial-state
pairing and timing audits gate subsequent batches. Each batch has archived audit
reports; the final audit covers every new episode. Infrastructure failures or
audit violations stop dispatch. There are no automatic retries, replacements,
resumes or additional rounds; every attempted slot is retained.

`progress.json`, `lifecycle.json`, per-worker logs and hardware snapshots expose
execution status. Existing primary exports are not overwritten by partial results.
Based on pilot means, the added cohort is estimated to require about 39 million
recorded input-plus-output tokens (cached input included), and roughly 3–4 hours
at four workers, plus audit and startup overhead. This is an estimate, not a
token cap or a monetary-cost estimate; interrupted usage can remain unknown.

## Analysis fixed before new outcomes

Report three cohorts: original five pairs, new 45 pairs, and pooled 50 pairs.
Primary outcomes are success within 200 seconds and mean capped completion time,
with timeouts assigned 200 seconds. Retain every outcome; do not replace difficult
layouts or tune policies, observers or prompts on them.

Report paired Full-minus-Base differences, with 95% paired-layout percentile
bootstrap intervals (10,000 resamples, fixed seed 1050), separately for success
and capped time. The pair is the resampling unit. These intervals describe the
sampled layout population under this protocol, not generality to other tasks or
objects. Report model-request time, completed requests, image attachments, local
camera acquisitions and recorded tokens. Include interrupted waits in model time;
leave unknown interrupted token usage explicitly unavailable.

The final `summary.json` contains all three cohort summaries and paired intervals;
`episodes.csv` retains episode-level outcomes and resource measurements. A larger
sample can change either the direction or magnitude of pilot effects. No claim
of improved performance follows from registration or an incomplete run.

## Client packaging correction

The first four startup attempts stopped at the actor audit before any submitted
policy action: the copied 0.159.2 client tried to start its default-enabled Code
Mode helper, which was not included in the binary copy. The model responses were
complete, but the strict event audit rejected the initialization error. All four
attempts, transcripts and the STOP receipt are preserved. Their 48,972 recorded
tokens are recoverable from agreeing CLI and session counters and are reported
as infrastructure overhead rather than successful episode requests.

The first correction, `client_repair_01`, disabled the optional features. Its
configuration-list check passed, but actual `exec` initialization still emitted
the missing-host error. Its four startup attempts are also preserved. In total,
these eight attempts used **98,036 recorded tokens**, recovered from agreeing CLI
and session counters. No submitted policy action ran in any of them.

The active correction is **`online_extension_50_01/client_bundle_02/`**, prepared by
`prepare_online_extension_bundle_repair.py`. It freezes the complete client bundle,
including the required host component, while disabling the Code Mode tool at the
`exec` argument level. An actual startup check against an unreachable localhost
provider passes without initialization errors, credentials or completed model
requests. The strict event audit is unchanged; no errors are filtered out.

The original 270 evaluation slots, seeds and timing remain fixed; the four affected
slots use fresh contexts and directories. There are eight additional infrastructure
attempts (278 total attempts for 270 episode slots), not additional layouts or
selected policy outcomes. All startup costs remain separate campaign overhead in
`infrastructure_attempts.json`. Any further failure stops dispatch. The top-level
`continuation.json` points to the active registration and progress file; earlier
STOP receipts and progress files retain their historical meanings.

## GPU runtime repair and continuation

The client-bundle run stopped after 48 completed episodes when the eighth Ring
pair exceeded the registered two-second simulation-lag limit (3.30 and 3.25
seconds). The first 44 episodes had passed the full batch audit; the final
Relocation pair subsequently passed as well. The Ring pair's interface/resource
audit passed, but its timing failure remains recorded. None of these 48 slots
is replayed or replaced.

During execution, an unattended system upgrade replaced NVIDIA userspace libraries
with 580.178.04 while the loaded kernel module remained 580.173.02. Hardware
snapshots beginning at slot 32 report NVML version mismatch. Sixteen episodes
therefore have a failed GPU query at dispatch, including the two lag violations.
This is evidence of a runtime fault; it does not establish the renderer actually
used by every historical episode. Preserve the hardware snapshots and flag these
records independently of outcomes.

The user explicitly authorized repair. `online_gpu_preflight.py` checks an isolated
580.173.02 userspace runtime extracted from NVIDIA's official installer, matching
the loaded module. No system installation, module unload, desktop interruption,
or reboot is performed. NVML and actual MuJoCo EGL rendering identify the RTX 3090.
A four-process, development-only sensing/pacing check exercised all three scenes,
both cameras and local geometry, with maximum lags of 0.41–0.77 seconds and no
model requests. This checks infrastructure, not task success or every possible
runtime workload. System-wide driver mismatch still requires a later maintenance
restart; the experiment uses its separately pinned libraries.

`resume_online_extension_gpu.py` registers **`gpu_resume_01/`** and dispatches only
the **222 unstarted slots** through the unchanged episode functions. The original
STOP, progress, protocol, completed outcomes and archived audits are hash-checked
and retained. Four global workers and the 200-second budget remain unchanged.
Each batch must pass a fresh NVIDIA NVML/EGL check and the original strict episode
audits, including the two-second lag threshold. New failure stops dispatch without
automatic retry. Follow the top-level `continuation.json` for active progress;
episode evidence continues under `client_bundle_02/tasks/`.

Exports retain all-slot descriptive results with explicit timing and dispatch-GPU
quality flags, alongside a separately labeled sensitivity analysis excluding
complete pairs with either flag. Original-five, new-45 and pooled-50 cohorts stay
separate. A timing-flagged success is not presented as timing-valid, and no
replacement layout restores an omitted pair. This post-hoc quality analysis is
an amendment after the fault was discovered, not an originally registered rule.

## Ring camera pacing amendment after 160 slots

`gpu_resume_01` stopped after slot 159: both Ring episodes on seed 2035102027
recorded about 3.28 seconds of maximum simulation lag, while NVIDIA EGL/NVML
checks remained healthy. The preceding 108 resumed episodes passed the strict
batch audits. The stopped batch's Tower pair also passed independently. All 160
outcomes are retained, including four timing violations across the two stops;
the original sixteen failed GPU-query flags remain separate.

Archived camera times show repeated unavailable Ring observations during the
shared grasp/lift prefix. Each side/wrist fitting attempt was shorter than the
lag limit, but the retry sequence accumulated more than three seconds of CPU
work with only short simulation waits. The enclosing agent loop normally advances
elapsed physics; local sensor retries did not return to that loop. A working GPU
therefore does not by itself prevent this pacing fault. The earlier GPU mismatch
was real, but it cannot explain every observed timing violation.

The user authorized continuation. `online_ring_camera_pacing.py` adds one
process-local hook: advance elapsed physics through the existing wait loop before
each `OnlineRingSensors.camera` call. Both conditions use it, including their
shared grasp skill. The original 200-second deadline and two-second lag gate
remain active. Geometry fitting, prior policy, prompts, submitted action targets,
success checks and the Tower/Relocation camera paths are unchanged. New images
are sampled after elapsed physics has advanced, so this is an explicit runtime
timing amendment, not byte-identical execution of the earlier cohort.

Development validation used seeds 1 and 2 for the shared Ring grasp, followed by
synthetic delayed-sensing retries, alongside Tower/Relocation probes in four
processes. Both lifts were visually confirmed; Ring maximum lag was about 0.69
seconds and all four probes passed. No reserved layout was rerun and no model
request was made. Regression tests cover cumulative lag, unchanged wall costs and
deadlines, timestamp ordering, timeout propagation, and continuation dispatch.

`resume_online_extension_camera.py` registers **`camera_resume_01/`** for only the
**110 unstarted slots (indices 160–269)**. It reuses the prior coordinator and
strict audits, with four workers and no automatic retries. Episode evidence
stays under `client_bundle_02/tasks/`; each new episode records its runtime
amendment. The final export additionally separates runtime epochs (before GPU
repair, private GPU runtime, and camera pacing) rather than hiding the change
inside the pooled estimate. Historical outcomes and flags are never overwritten.

## Terminal-image repair after 168 slots

The camera continuation stopped at 168 outcomes. Ring slots 164 and 165 had
already reached success and saved their terminal model/state and two-second
stability traces. The final illustration capture then invoked the camera pacing
hook, which re-entered physics and raised the existing terminal-success signal.
The resulting `archive_error` stopped dispatch. This was an archival lifecycle
error; the two final illustration images were not saved.

`online_ring_camera_lifecycle.py` retains live camera pacing and marks an
environment terminal when its physical state is archived. Subsequent final-image
capture does not advance physics. Live deadlines and success signals still
propagate. Development checks on seeds 1 and 2 exercise actual successful and
timed-out episode finalization with no model calls or reserved layouts. Both
paths passed; the earlier failed timeout-probe attempt remains archived.

`terminal_resume_01/archive_recovery/` contains a derived audit view for the two
affected episodes. Its result copies remove only the known image error, and all
other evidence links to the originals. Independent prompt, image-input, token,
source, terminal-physics and stability audits passed before the outcomes entered
the continuation baseline. Receipts retain both result hashes, the original
error, and the missing-image flag. No model request, physics step or episode
replay was added. The original result files remain byte-for-byte unchanged.

The user authorized continuous monitoring/debugging through completion.
`resume_online_extension_terminal.py` registers **`terminal_resume_01/`** for
the **102 unstarted slots (168–269)**, with the same four workers, 200-second
budget and strict timing gates. All 168 existing outcomes are retained, with
four timing flags, sixteen dispatch-GPU flags and two independently verified
archive recoveries. The final summary separates runtime epochs, original/new
cohorts and paired quality sensitivities, including a sensitivity without the
recovered pair. Further faults stop dispatch for diagnosis; existing episodes
are never silently retried or replaced.

## Completed expansion and results

The following completion figures describe the **original attempts**, before the
subsequent paired-repeat reporting amendment below.

The expansion completed on 2026-10-21: all 270 registered new episodes are
finished, giving 300 episodes and 50 paired layouts per task with the original
30 episodes. The coordinator exited normally. The final strict audit covers all
102 episodes dispatched by `terminal_resume_01`; all passed. A separate completion
check verified all 300 original result hashes, exact registered slot coverage,
paired initial-state identities, 200-second budgets, single-reset execution,
token arithmetic and CSV consistency. It also checked the two archive-recovery
views against their unchanged original results.

| Task | New 45: Base | New 45: Full | Pooled 50: Base | Pooled 50: Full | Pooled capped time, Base / Full (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bridge-tower | 10/45 | 23/45 | 12/50 | 27/50 | 198.0 / 185.4 |
| Ring-on-peg | 44/45 | 45/45 | 49/50 | 50/50 | 115.6 / 77.3 |
| Tower relocation | 13/45 | 39/45 | 15/50 | 42/50 | 187.7 / 139.8 |

These are all-slot descriptive results and include the preserved quality flags:
four timing violations and sixteen failed dispatch GPU queries (two episodes
have both). The two independently verified terminal-image archive recoveries
retain separate labels. Excluding complete pairs with timing or GPU-query flags
leaves 48 Tower, 46 Ring and 47 Relocation pairs in the pooled sensitivity
analysis. Base/Full successes are respectively 12/48 versus 27/48, 45/46 versus
46/46, and 14/47 versus 39/47. All three tasks retain lower Full mean capped time
in that subset. A further new-cohort sensitivity excludes the recovered Ring
pair. Original-five, new-45, pooled-50 and runtime-epoch summaries remain separate.

The frozen completed-receipt token total is 41,585,499, including 37,242,497 from
the new episodes. A subsequent read-only audit recovers 90,796 tokens from four
terminal requests whose CLI totals agree with append-only session counter deltas.
The corrected export records 41,676,295 tokens (37,311,743 new); the frozen
summary and all episode outcomes remain unchanged. Of 76 interrupted requests
(48 deadline cancellations and 28 after verified success), 72 still have unknown
usage. Recovery does not turn a cancelled decision into an executed one.
The eight earlier startup attempts contribute a separate 98,036
recorded tokens of infrastructure overhead; their artifacts remain preserved.

The expanded results support higher completion rates on the two construction
tasks and shorter Ring completion time under the fixed budget. They remain a
comparison of the complete supplied geometry-and-feedback package, with one
attempt per arm/layout and the disclosed runtime amendments; this expansion
does not add representation ablations or repeated attempts on each layout.

Authoritative artifacts:

- [Final progress](../outputs/online_extension_50_01/terminal_resume_01/progress.json)
- [Completion verification](../outputs/online_extension_50_01/terminal_resume_01/completion_audit.json)
- [Summary, paired intervals and quality sensitivities](../outputs/online_extension_50_01/terminal_resume_01/summary.json)
- [All 300 episode records](../outputs/online_extension_50_01/terminal_resume_01/episodes.csv)
- [Strict final audit of the 102 continuation episodes](../outputs/online_extension_50_01/client_bundle_02/audits/gpu_resume_final/records.json)

The expanded reporting step exports the 300 episodes, request-level resources,
cohort summaries, paired intervals and five figures under
[evaluations/online_50](evaluations/online_50/README.md). The paper figures and
numerical text now reference that export. Original pilot exports under
`evaluations/online/` remain unchanged as historical evidence.

## Primary-report update after targeted repeats

After the expansion, all nine timing/GPU-flagged pairs were repeated once under
the separate [targeted-repeat registration](ONLINE_TARGETED_REPEAT.md). All 18
passed audit. The user then requested incorporating them into the primary report;
this is an explicit post-hoc reporting amendment. Both repeated arms replace the
original on every selected pair, including regressions. Raw original outcomes,
source registrations and the previous exported figures remain preserved.

The current 300-row analysis therefore comprises 282 original episodes plus 18
repeats. Bridge-tower Base/Full success becomes **12/50 versus 28/50**; Ring stays
**49/50 versus 50/50**, and Relocation **15/50 versus 42/50**. Revised capped times
are **198.0/185.3**, **115.8/77.2**, and **186.4/139.2** seconds, respectively.
Selected recorded tokens total **41,603,662**, including 90,796 verified terminal
tokens; 73 interrupted usages remain unknown. The 18 superseded originals used
another **2,745,300** recorded tokens. Including them and 98,036 startup tokens,
recorded campaign cost is **44,446,998**, with unknown usage additional.

The original-attempt aggregate and sensitivity excluding all nine affected pairs
remain in the current export. See `replacements.csv`, `superseded_attempts.csv`,
`original_attempts.csv`, and `replacement_manifest.json` under
[evaluations/online_50](evaluations/online_50/README.md). No new independent layouts
or component ablations are introduced by this amendment.
