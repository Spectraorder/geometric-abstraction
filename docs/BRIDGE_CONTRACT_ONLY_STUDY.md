# Bridge policy acquisition with actuator contracts only

This study follows `geometry_parallel_01` with fresh paired training chains and
fresh layouts. Both arms receive the public actuator contract without its source.
The original bridge scene, geometry bundle, shared visual pick/place actuator,
0.70 grasp fraction, noslip-zero physics, observation-only starter, and
failure-triggered image feedback remain unchanged. The historical source-disclosed
study is preserved as a separate comparison.

## Registered design

- Output: `outputs/geometry_bridge_contract_01`, backed by SSD storage.
- Entry point: `scripts/run_bridge_contract_study.py`.
- Five independent Base/Full chain pairs, with separate persistent actor contexts.
- Training seeds: 2032100101–2032100125; five disjoint layouts per pair, shared by
  its two arms.
- Reserved seeds: 2032101001–2032101100; 100 fresh layouts shared by all policies.
- R0 observes without acting. R1–R3 are generated Python programs.
- Model: `gpt-6-astra`, high reasoning; 900-second request timeout and a
  50,000-character source limit.
- Budget: 30 completed model generations, 200 training slots, and 4,000 reserved
  evaluation slots. Every R0–R3 candidate is evaluated after all training freezes.
- Concurrency: at most four model requests and 24 simulation workers globally.
  Training reduces new rollout leases under memory pressure; evaluation starts
  with 48 slots and falls back to eight workers if the resource check fails.
- Policy runtime: CPU perception and control, 120-second execution limit and
  1,200 API calls. GPU simulation rendering is permitted.
- Retention: accept only a strict superset of the incumbent's training successes;
  ties and regressions retain the incumbent. No test-based policy selection.

The complete revision-by-layout matrix is specified before training. Materializing
its evaluation records later requires completion and audit of all five pairs.
Invalid programs consume failed slots without a physical rollout. No reserved
outcomes enter actor prompts, revisions, or retention.

## Information boundary

Both arms retain calibrated side/wrist RGB-D, public robot information, a
color-addressed declarative support goal, qualitative action feedback, and the
same actuator. Base can write its own numerical perception. Full additionally
uses the existing geometric queries, operators, and checkpoints. Neither actor
receives the executor implementation, oracle coordinates, private contacts, or
private evaluator details. The shared actuator still performs visual grasp
localization in both arms.

The bridge comprises two cube foundations, a spanning beam, and a cylinder cap.
The frozen support-graph evaluator and five-second stability check are retained,
including the historical contact helper's limitation for unnamed robot geometry.
This study changes prompt disclosure; it does not silently change the historical
physical success criterion. Fresh chains and layouts mean that differences from
the previous study are descriptive rather than a paired causal estimate of
source disclosure alone.

## Transport, accounting, and preservation

All actors use the same HTTP-only OpenAI provider alias from their first turn.
The alias retains existing authentication, model, reasoning setting and timeouts;
it changes no task information. This avoids mixing WebSocket and HTTP requests
within the new study. The specialized actor implementation is archived and hashed
before dispatch; the historical actor file is unchanged.

Per-turn token accounting verifies cumulative-counter transitions and excludes
unchanged broadcasts from new usage. Cached input is included in input tokens;
reasoning output is included in output tokens. Record image attachments, camera
calls, request time, rollout cost, and worker concurrency separately. Preserve
incomplete requests with unknown usage rather than treating them as free.

Stop dispatch on infrastructure failures and preserve completed and incomplete
artifacts. No automatic retries, replacement seeds, extra rounds, or extension
of the test split are registered. Report raw and training-retained test curves,
paired chain variation, and cumulative acquisition resources. The 100 test
layouts are shared across five policies per arm; they are not 500 independent
layouts. These are single-policy experiments, with no model calls inside a
rollout.

## Validation and entry points

Focused tests check both-arm source exclusion, unchanged persistent follow-up
feedback, scoped HTTP configuration, the complete 4,000-slot evaluation matrix,
and rejection of changes to frozen policies. Observation-only preflight uses
development seeds 1 and 2, with no model calls or reserved execution. It checks
paired initial observations and reconstructs the two initial actor prompts.
Runtime bundle hashes are compared with the historical bridge study.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python \
  scripts/run_bridge_contract_study.py \
  --output outputs/geometry_bridge_contract_01 --phase prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -u \
  scripts/run_bridge_contract_study.py \
  --output outputs/geometry_bridge_contract_01 --phase all
```

The coordinator proceeds from training to evaluation automatically after the
training audits and global freeze. Training progress appears at the study root;
evaluation progress and curves appear under `learning_curves/`. The root
`lifecycle.json` records completion or failure.

## Launch verification

The seven focused tests passed. All four development observation-only rollouts
completed without policy errors; both arms attached four initial images. The
executor, both runtime libraries, and starter matched the historical bridge
bundle byte for byte. Fresh-seed verification inspected 254 registration files,
including SSD-backed study directories, and found no overlap. The campaign is
registered and launched, with 24 rollout leases active during initial R0 checks.
This validates startup and the interface boundary, not task success.
