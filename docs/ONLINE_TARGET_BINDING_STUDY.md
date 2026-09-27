# Five-layout online target-binding comparison

This separate study evaluates the explicit-target interface after component,
physical and paired live development checks. It retains the ordinary bridge plus
two-cube task, no scene reset after failure, and the user's small evaluation scope.
It does not extend the stopped node-only study or reuse its reserved layouts.

Status: all ten episodes completed and passed both audits. Full succeeded on 3/5
layouts and Base on 0/5 under the original 180-second budget. Full used more
recorded tokens; see the [complete results and limits](ONLINE_TARGET_BINDING_RESULTS.md).

## Budget amendment for subsequent runs

The user requested 200 seconds per episode after all ten slots of
`outputs/online_bound_tower_01` had started. Their deadlines were already fixed;
that registered study retains its 180-second protocol and recorded outcomes.
The next registration and explicit-target development default to 200 seconds in
both arms. This amendment does not extend active deadlines, relabel old timeouts,
authorize automatic retries, or launch another campaign. A new reserved campaign
requires a separate registration and fresh paired layouts; the five-layout and
four-worker scope remains unchanged.

Archived sources remain the implementation record for the 180-second study.
After the working runner is updated, its strict source check against that old
registration is expected to reject the changed working files. Use the archived
source tree to reproduce the historical implementation.

## Fixed protocol (historical 180-second study)

- Five fresh shared seeds: 2028104001–2028104005, with no outcome screening.
- One Base and one Full episode per seed: ten slots, no repetitions or replacements.
- A 180-second wall-clock budget per episode, beginning after the sole initial
  reset and including initialization, model requests, queries, actions and settling.
- At most four workers in balanced 4/4/2 batches; one BLAS/OpenMP thread per worker.
- One persistent Astra medium context per episode; one placement or query per
  decision for both arms. Same skill experience, controller, action schema and
  support-change interruption rule.
- Both arms can request images and metric pixel samples from the displayed frame.
  Full additionally receives typed object references with current evidence. No
  automatically computed task destination is given to either actor.
- Explicit relocations provide recovery. No injected collapse, displacement,
  manual rescue, reset after failure or automatic target replacement.
- Infrastructure errors stop new batches after in-flight workers finish. Retain
  STOP receipts and every attempt; no automatic retry or budget extension.

The actor selects worksite, order and numeric support-surface targets. The shared
actuator executes the submitted point. This differs from earlier node-only skill
calls, which selected destinations inside the executor. Report the interface
change and do not pool those earlier results with this study.

## Outcomes and resource reporting

Primary outcomes are success out of five and mean capped completion time, with
all unsuccessful episodes counted at 180 seconds. Show each paired result and
its signed time difference; unstarted slots have no outcome and cannot be silently
counted as failures or successes. Success uses the same private complete support
graph, five-second stability, 3 mm drift bound and robot-contact exclusion in both
arms. Contact grouping includes unnamed collision meshes through body ancestry.

Also report input tokens, cached input, output tokens, completed requests,
incomplete requests, model-request wall time, attached images, metric queries,
camera acquisitions, feedback time and action time. Cached input is included in
input tokens. Missing request usage remains unknown. Campaign wall time is
separate from episode latency. Flag simulation lag above two seconds and qualify
any real-time claim rather than discarding the episode.

The independent audit compares actual model responses with submitted target
bindings, checks depth samples against displayed RGB-D archives, excludes object
reference feedback from Base, verifies paired initial physics, and reconstructs
terminal robot contacts. It also recomputes token deltas from cumulative session
counters. Terminal reconstruction is not independent replay of the complete
five-second stability interval.

Five pairs provide exploratory evidence, not a general guarantee. Report all
unfavorable results and resource tradeoffs. Do not grow or modify the study until
a desired result appears. The positive development example is excluded from this
evaluation and does not justify a presumption that Full will win.

## Entry point and artifacts

Use `scripts/run_online_bound_study.py` with `--root
outputs/online_bound_tower_01`. `--phase prepare` freezes runtime sources/assets,
geometry bundle, package/client versions, seeds, slots and development evidence;
`--phase check` verifies those records. `--phase run` dispatches the ten slots and
produces `progress.json`, source/resource and interface audits, and `summary.json`.
`--phase summarize` can recompute the analysis without rerunning episodes.

The separate development CLI still rejects reserved seeds. Registered workers use
the same episode body only after verifying the frozen study. Existing attempts
cannot be automatically resumed or overwritten. Interpretation and adaptation
details are in [the development record](ONLINE_TARGET_BINDING_DEVELOPMENT.md).
