# Paired ring-over-peg policy acquisition study

The user authorized five independent training chains per arm, R0–R3, five training
layouts per pair, and 100 shared new reserved layouts evaluated at every revision.
This is a new Base/Full study; the preceding Full pilots are development evidence.
Their learned policies, outcomes and privately evaluated coordinates do not enter
the actors in this study.

## Registered design

- Output: `outputs/ring_peg_comparison_01`, backed by the SSD study directory.
- Entry point: `scripts/run_ring_peg_comparison_study.py`.
- Five replicas, each with separate Base (`basic`) and Full (`symbolic`) contexts.
- Training seeds: 2031101001–2031101025, five consecutive seeds per paired replica.
- Shared reserved seeds: 2031109001–2031109100. No screening or replacement.
- R0 is the same non-acting API scaffold source for both arms. R1–R3 are generated
  programs in fresh independent persistent Astra-high contexts for each chain.
- Thirty model requests and 200 training rollout slots; 4,000 reserved slots
  cover two arms, five replicas, four versions and 100 layouts.
- Preserve every incumbent training success and add at least one to replace it;
  ties retain the incumbent. No reserved outcome influences policy retention.
- Complete all ten training chains, audit actor lineage, reconstruct prompts,
  verify paired initial images and resource counters, then freeze all candidates
  before any reserved rollout. Every candidate is evaluated, including rejected
  training proposals and non-acting R0.

## Information and control boundary

Both arms receive the same task, object identities/color labels, declared shape
assumptions, qualitative starter guidance, robot kinematic previews and explicit
pad/jaw motion API. Neither actor receives executor, perception or oracle source.
The same source scaffold is supplied to both arms. It includes explicit command
bookkeeping, error handling and observation examples, with no grasp search or
manipulation sequence. It uses the public `geometry_available` capability flag
to request Full geometry during R0; both arms also acquire an initial raw frame.
The capable 4/5 starter is not supplied.

Both isolated policy processes can call `api.camera('side'|'wrist')`. This returns
identical calibrated 640x480 RGB-D access: RGB, optical-axis metric depth, valid
pixels, intrinsics, camera-to-world transform and time. Arrays are transferred
losslessly through a compressed NPZ transport and decoded inside the trusted RPC
client. No repository files or simulator objects are exposed to policy code.
The Base policy can implement CPU estimation with numpy/scipy/math.

Full additionally has the existing sensor-grounded `api.observe(object_id)`.
Base requests for this method are rejected before any fitting. Base runtime
records and actor feedback contain only its own notes and timestamps, qualitative
errors and binary outcomes; private evaluator details remain host-only. Both arms
receive the same qualitative geometry/robot guidance, including material versus
opening references, three-dimensional closing direction, and height-aware finger
clearance. Full field names explain its additional observation API; they do not
supply geometry values to Base.

Both arms use the unchanged repaired pad actuator, ring scene, noslip 0/libccd
contact settings and private insertion/release/two-second stability evaluator.
Runtime object positions and dimensions remain sensor-derived. This compares the
supplied geometry package, not an isolated effect of text serialization or a
real-time online LLM controller.

## Feedback, execution and resources

Each model revision receives its own preceding training feedback, the training-
retained policy, and the latest side image from each of the first two registered
training seeds. Feedback retains at most four latest permitted records. The actors
cannot use tools or read source files. Contexts do not cross arms or replicas.
Each rollout permits 120 seconds and 600 API calls, with a 50,000-character source
limit. Invalid programs are recorded as failed slots, not silently retried.

Up to four chain processes and four model requests run concurrently. Each chain
runs five training rollouts, with a global 24-rollout lease limit. Reserved testing
starts with a 48-slot pilot at up to 24 workers. Hardware samples during the pilot
trigger a fallback to eight workers if available RAM falls below 8 GiB, free GPU
memory falls below 2,000 MiB, or GPU sampling fails. Keep a 20 GiB disk reserve.
Sources, starter and both contracts are frozen before dispatch. Infrastructure
errors stop new work, retain in-flight evidence and produce `STOP.json`; no retries,
extra rounds or seed replacement are authorized.

Report raw and training-retained reserved curves, per-chain variability and
cumulative verified model tokens. Cached input is already included in input;
reasoning output is already included in output. Resource reports retain model
request time and image counts. Scheduler leases separate total work/queue time
from parallel campaign wall time. The same 100 layouts are shared across chains;
they are not 500 independent test layouts. Training-chain replication remains five.

## Development validation

`outputs/ring_peg_dev/comparison_preflight_01` contains six observation-only RPC
checks on the three existing development seeds, one per arm. All completed, and
initial RGB files matched exactly for every pair. Tests separately verify raw
array fidelity, field whitelisting, Base geometry rejection, shared qualitative
guidance, strict retention and feedback-leak rejection.

A separate Full motion check on development seed 1 through the paired runtime
completed the physical task, including the unchanged two-second stability check,
in `outputs/ring_peg_dev/comparison_motion_01`. It used the capable development
skill solely to exercise the runtime. Neither that skill nor the result enters
actor prompts. These checks establish interface viability, not study success rates.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python \
  scripts/run_ring_peg_comparison_study.py \
  --root outputs/ring_peg_comparison_01 --phase prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -u \
  scripts/run_ring_peg_comparison_study.py \
  --root outputs/ring_peg_comparison_01 --phase run
```

## Authorized recovery after a model timeout

The first campaign attempt stopped after 140/200 training slots and 22 completed
model generations. Replica 3 Base R2 encountered WebSocket idle timeouts and
reached the registered 900-second request limit. No candidate or final response
was saved. Reserved evaluation had not started.

The user explicitly authorized repair and continuation. The separate
`resume_ring_peg_comparison.py` adapter preserves the frozen runtime and protocol.
Its `resume_01` registration pins every completed candidate and trial artifact,
archives the original stop/lifecycle and interrupted request, and preserves the
interrupted actor home, including its local history databases. The active actor
home is reconstructed from its complete R1 transcript, maintaining the same
session identity and completed context without the interrupted turn. The exact
registered R2 prompt is then submitted once. Existing generations and rollouts
are audited and reused. There are eight remaining generations and sixty remaining
training slots, followed by the original 4,000 reserved slots. This amendment
permits at most 31 model attempts for 30 completed revisions, including the one
interrupted attempt; it does not authorize another automatic retry.

Model settings, the 900-second timeout, prompts, training retention, seeds,
actuators, evaluators and concurrency limits remain unchanged. The adapter keeps
the all-training-frozen gate and the 48-slot hardware pilot before full evaluation.
Its scheduler audit includes the registered failed lease in concurrency and cost
accounting, while rejecting any additional unregistered failure.

The interrupted request lasted approximately 900 seconds and attached two images.
Its token events only repeated the preceding turn's cumulative counters, so its
final token usage is unknown. Resource reports preserve this incomplete attempt
separately. Completed-turn token curves are therefore incomplete measures of total
consumption, not evidence that the interrupted request was free. Original elapsed
campaign time is retained separately from resumed execution time.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python \
  scripts/resume_ring_peg_comparison.py \
  --root outputs/ring_peg_comparison_01 --phase register
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -u \
  scripts/resume_ring_peg_comparison.py \
  --root outputs/ring_peg_comparison_01 --phase run
```

## HTTP transport continuation

After the authorized retry also reached the 900-second WebSocket timeout, the
user requested continuation. `resume_ring_peg_http.py` registers `resume_02_http`
without changing either earlier registration. It reuses 28 completed generations
and 185 training slots, preserves both interrupted requests, and dispatches only
the remaining two generations and fifteen training slots before the original
4,000-slot reserved evaluation. The total model-attempt ceiling is now 32,
including both incomplete requests; further failures stop dispatch.

This continuation changes the two remaining model requests to an HTTP-only custom
provider alias, `openai-http`, with `requires_openai_auth=true`,
`wire_api="responses"`, and `supports_websockets=false`. It supplies no custom
base URL or credentials and retains the existing OpenAI login. The installed
CLI rejects overrides of the built-in `openai` provider and has removed its old
WebSocket feature flags, so a provider alias is required. Offline configuration
validation passed. Request commands record the exact transport settings.

The model remains `gpt-6-astra` with high reasoning, unchanged prompts, image
selection, completed persistent context, 900-second request timeout, and policy
source budget. The runtime, source checks, training selection, seeds and
all-training-frozen evaluation gate remain fixed. Both interrupted requests have
unknown final token usage and are retained separately in resource accounting.
Request-time comparisons must disclose this mixed-transport continuation.
The prior elapsed time includes both preceding campaign attempts.
