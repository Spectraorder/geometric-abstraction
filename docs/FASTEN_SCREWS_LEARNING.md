# Fasten Screws learning campaign

Current learning physics is `fasten_robodojo_v2`: see the
[thread deflection/rebound correction](FASTEN_THREAD_CONSTRAINT.md).
All campaigns named below were recorded with earlier physics. Their saved
scores and prompts remain unchanged. For new learning, choose a fresh root,
such as `outputs/fasten_single_nut_v2_01`; do not resume an old V1 root.

Current objective: **full fastening**, restored after the
[tracking/grasp diagnosis and pickup checks](FASTEN_PICKUP_DIAGNOSIS.md).
The first RoboDojo-asset campaign never closed its gripper; the later pickup-only
LLM campaign achieved 2/3 development passes. Pickup success is not a 20/50/100
fastening score. It is a source of reusable code, not the final task.

## Single-nut starting curriculum

Start with `--pairs 1`: the scene contains only one red nut and one fixed red
bolt, with seed-dependent position jitter. Success requires the same three net
clockwise turns, depth, two-second released stability, open gripper and return
home. The score remains the original one-pair milestone (20); success here is
**single-nut curriculum success**, not three-pair benchmark completion.
The runtime policy still receives only camera data, robot proprioception and
symbolic goals. Defaults remain three pairs for historical runs and replay.
Pair count is saved in campaign/trial metadata and checked before resuming or
reusing trial results. Use a fresh root when changing it.

Campaign `outputs/fasten_single_nut_01` starts with the current-scene full-task
round-2 code and pickup round-2 code as unscored generation references. It uses
`gpt-6-astra`, `robot_gravity_ff_v1`, three rounds, three development seeds per
round, and the existing 60,000 reported-token stop threshold checked between
calls (not a hard billing cap). Once all development seeds pass, the policy is
frozen and evaluated on the separate held-out seeds.

```bash
uv run --no-sync python scripts/launch_fasten_learning.py \
  --jobid YOUR_ACTIVE_GPU_JOB_ID --root outputs/fasten_single_nut_01 \
  --pairs 1 --stage fasten --rounds 3 --max-reported-tokens 60000 \
  --max-prompt-chars 50000 \
  --reference-policy outputs/fasten_screws_robodojo_full_01/round_002/candidate.py \
  --reference-policy outputs/fasten_pickup_llm_01/round_002/candidate.py

tail -F outputs/fasten_single_nut_01/progress.log
```

While the model is generating code, its live events are in
`outputs/fasten_single_nut_01/round_001/model.jsonl`. After a round finishes:

```bash
uv run --no-sync python view_scene.py \
  --vision-trial "$(cat outputs/fasten_single_nut_01/best_trial.txt)"
```

## Full-task continuation with prior code

Campaign `outputs/fasten_screws_robodojo_full_01` uses the current RoboDojo scene
and `robot_gravity_ff_v1` controller, with the original complete 20/50/100 objective.
Its first generation receives two **LLM-generated** code references:

- `outputs/fasten_screws_01/round_003/candidate.py`: the earlier partial-fastening
  policy. Its replay on the drift-corrected procedural V3 scene reached about
  0.258 turns for the red pair on development seed 5102, not a completed fastening.
- `outputs/fasten_pickup_llm_01/round_002/candidate.py`: the newer pickup policy,
  which passed 2/3 current-scene development layouts. Its finish-holding behavior
  must be replaced by alignment, threading, release and return home.

Reference sources are copied into `reference_policies/` with SHA-256 provenance
in `campaign.json`. No previous scores, trial successes, or best-policy rankings
are imported. Only newly generated and evaluated full-task policies can become
this campaign's best policy. Later rounds receive the new best code and feedback,
not both full reference files repeatedly. The hand-written pickup diagnostic
is not supplied as a learned policy.

The first prompt needs a 40,000-character cap to include both complete references.
At the user's request for a longer run, it is limited to five rounds and a
100,000-reported-token threshold checked
between calls (one call can exceed it; this is not a hard billing cap).
The model is `gpt-6-astra`; development/frozen seeds remain separate.

Start on the allocated GPU node with a fresh root, or resume the same campaign:

```bash
uv run --no-sync python -u scripts/learn_fasten_screws.py \
  --root outputs/fasten_screws_robodojo_full_01 --stage fasten \
  --model gpt-6-astra --rounds 5 --max-reported-tokens 100000 \
  --max-prompt-chars 40000 \
  --reference-policy outputs/fasten_screws_01/round_003/candidate.py \
  --reference-policy outputs/fasten_pickup_llm_01/round_002/candidate.py
```

Monitor with `tail -f outputs/fasten_screws_robodojo_full_01/progress.log`.
The exact generation input is saved in each round's `prompt.txt`.

## Task and physics

The learning task extends the original two-pair inspection scene to three pairs
(red, blue, yellow), matching the supplied 20/50/100 scoring milestones. Bolts
are fixed to the tabletop so the single SO-100 can manipulate the nuts. Positions
are mildly jittered and the nut color order is shuffled by seed.

This is our **simplified compliant helical model**, inspired by the
[RoboDojo scoring description](https://robodojo-benchmark.com/doc/sim-tasks/fasten-screws/).
Current scene `fasten_robodojo_v2` uses the [original RoboDojo assets](../assets/robodojo_fasteners/README.md),
converted from USDZ with their original dimensions. It does not reproduce
RoboDojo's PhysX SDF collision simulation or claim validated thread mechanics.
These thread triangles do not generate individual thread-contact forces. After
the grasp/alignment engagement checks, the solver constraints
couple nut height to its unwrapped rotation about the bolt. Historical V1 uses
explicit compliant forces and torques. No object pose is
teleported and no nut is attached to the gripper by a weld. Further calibration
and grasp/reach evaluation will be needed before hardware use.

Entry requires both jaws to contact the nut, its axis within 6 degrees of
upright, XY error below 2 mm, and its bottom near 2 mm below the bolt tip.
The thread can engage any color; only matching colors earn credit. Clockwise
rotation viewed from above advances the nut with a nominal 3 mm pitch. Three net turns
and approximately 9 mm depth are required. This replaces the procedural models'
4 mm pitch / 12 mm depth; old trials are not rescored. Turning backward subtracts progress;
it cannot accumulate progress by oscillating. Large misalignment breaks engagement.
Unthreading counterclockwise past entry releases it. Solver-based dry friction
(introduced in **fasten_helical_v3**, retained with the imported models) lets the
robot release/regrasp between limited wrist-roll strokes.

In v2 and later, the coupling seats to the nearest modeled thread phase for the current
nut yaw (at most half a pitch from nominal entry), without teleporting it. Depth
is measured from that phase-specific entry. The 2 mm XY / 6 degree entry tolerance
is a coarse capture approximation, not the much tighter geometric clearance.
This does not simulate cross-threading, material deformation or tightening torque.

### Gravity-drift correction (v3)

V1/V2 limited friction using the current angular speed. At rest that limit became
zero, so the gravity-induced axial reaction drove slow, uncommanded rotation.
The reported turn count therefore included passive drift. The earlier short
torque/depth tests did not detect this; those results are not evidence of reliable
self-locking or robot-driven fastening.

V3 replaces that explicit spin-drag approximation with MuJoCo's solver-based
[dry friction](https://mujoco.readthedocs.io/en/stable/computation/index.html#friction-loss)
on the engaged nut's local spin axis: 0.002 N m friction loss and 0.006 N m s/rad
implicit damping. A high-impedance friction constraint limits numerical creep.
It is activated only while engaged with a fixed bolt and removed on disengagement
or reset. Translation is not locked; no poses or velocities are overwritten.
In the historical spring model, the compliant axial spring settles slightly under weight, which
is distinct from continued screw rotation. Parameters are demonstration values,
not measured hardware friction.

In the seed-5102 gravity-only fixture, after one second of settling, a 30-second
hands-off hold drifted **1.4064 mm / 0.35161 turns in V2**, versus **0.000118 mm /
0.0000302 turns in V3**. This is near-static within numerical tolerance, not a
claim of mathematically zero drift for unlimited simulation time.

Regression tests cover a 30-second hands-off hold after settling (less than
0.01 mm depth drift and 0.001 extra turns), a near-complete nut that cannot earn
credit by waiting, applied torque advancing and reversing the nut, sustained
holding after fastening, and friction removal on disengagement/reset.
V1/V2 recordings remain replayable with their original models; never silently
relabel or reuse their scores as V3 results. Learning is not restarted by this fix.

| Score | Measured condition |
| --- | --- |
| 0 | No qualifying pair, or gripper not open/released |
| 20 | One matched, upright, centered pair reaches rotation and depth requirements; gripper open |
| 50 | At least two pairs qualify and gripper open; also applies to three if robot has not returned home |
| 100 | All three qualify, gripper open, every arm joint within 0.12 rad of starting home |

Each qualifying pair must remain valid without jaw contact for 2 seconds.
Evaluation occurs after a 5-second post-policy hold. Trials with a policy error
cannot be successful even if the scene score happens to reach 100.

Separate stage counts (out of 3) are reported for approach, two-jaw contact,
lift, alignment, engagement, quarter-turn rotation, target depth, and sustained
fastening. These are **diagnostics**, not additional benchmark points. The best
policy is selected by full-success count, score sum, then stage-count sum and
fewer policy errors. A sensor-only infrastructure probe is never a best-policy
candidate. Thus an inert probe cannot silently replace an attempted learned policy.

The exact task/API prompt is [FASTEN_SCREWS_PROMPT.txt](FASTEN_SCREWS_PROMPT.txt).
The LLM receives RGB-D images, the best prior code, recent hypothesis outcomes
and post-trial feedback; the runtime policy receives only calibrated camera data,
robot proprioception and symbolic goals. Evaluator measurements do not enter
runtime observations. Policy generation uses the existing isolated Codex CLI
backend; no new API key is required if `codex login status` succeeds.

The first campaign is bounded to 3 LLM rounds, 3 development seeds per round,
32,000 prompt characters, 300 seconds per model call and 300 seconds per trial.
The reported-token stop threshold is 60,000: it is checked **between calls**, so
one call can exceed the threshold. Images/provider overhead and incomplete calls
can add usage; this is not a guaranteed billing cap. Progress and reported usage
are persisted. No learning job starts merely by opening the viewer.

Run directly on an allocated GPU node:

```bash
uv run --no-sync python -u scripts/learn_fasten_screws.py \
  --root outputs/fasten_screws_robodojo_01 --model gpt-6-astra --rounds 3
```

Or launch with a log from the login node, using your active allocation:

```bash
uv run --no-sync python scripts/launch_fasten_learning.py \
  --jobid YOUR_JOB_ID --root outputs/fasten_screws_robodojo_01 --rounds 3
tail -f outputs/fasten_screws_robodojo_01/progress.log
```

Resume using the same root and a higher total round cap (and raise the token cap
explicitly in the direct command if needed). Do not start a second process while
the first is active. Existing completed trials are reused only for the same
candidate hash, task, seed and scene version. Interrupted trial directories are
preserved. The old `outputs/fasten_screws_01` campaign used v1 smooth-bore nuts;
it remains historical evidence. V2 threaded-scene trials also used the drifting
friction model. V3 fixed friction but still used procedural geometry. None can be
resumed as a `fasten_robodojo_v2` campaign; V1 campaigns also require a fresh root. The runner rejects such reuse before
making a model call. Start a new root for the imported assets.

After the first completed round, `best_policy.py` and `best_trial.txt` are updated
automatically. Open the trial path stored in the text file:

```bash
uv run --no-sync python view_scene.py \
  --vision-trial "$(< outputs/fasten_screws_robodojo_01/best_trial.txt)"
```

This uses the existing 1x replay, causal trajectory and look-ahead phantom.
To inspect just the three-pair scene:

```bash
uv run --no-sync python view_scene.py --scene-only \
  --scene-task fasten_screws_learning --seed 5101
```

Development seeds are 5101–5103. Once all three score 100, the best policy is
frozen and tested on 5201–5203; held-out results are not sent back to the actor.
Repair-learning work remains in its previous stash.
