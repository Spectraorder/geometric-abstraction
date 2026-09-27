# Single hoop and pole: scene inspection

A simpler insertion scene with one freely movable red hoop and one matching red pole
fixed to the tabletop. The hoop starts flat on the table beside the pole.
The intended future task is to lift the hoop over the rounded pole tip, lower
it around the pole and release it onto the base. No screwing is required.

```bash
uv run --no-sync python view_scene.py --scene-only --scene-task hoop_pole
```

Open the printed Viser URL. The SO-100 arm, wrist camera and side RGB-D camera
are included. This is a passive inspection scene: no policy, evaluator or
learning campaign is started. The first layout is fixed; `--seed` does not
change it. There is exactly one pair; omit `--object-count`.

| Part | Dimensions |
| --- | --- |
| Hoop | 40 mm outer diameter, 24 mm hole, 12 mm height, 20 g |
| Pole | 12 mm diameter, 51 mm total height including base and rounded tip |
| Fixed base | 48 mm diameter, 6 mm height |
| Radial insertion clearance | Nominally 6 mm |

Physics uses ordinary MuJoCo rigid-body contact and gravity, with no thread
model, grasp attachment, engagement trigger or invisible entry stop. The hoop
uses 32 convex annular sectors for collision so the central hole stays open;
its smooth visual surface uses 96 segments. The cylinder/sphere pole collision
surfaces are also its visible surfaces. The polygonal hoop bore differs from
its circular visual surface by at most about 0.058 mm radially.
Scene identifier: `hoop_pole_v1`.


## Learning campaign

`outputs/hoop_pole_01` learns placement with the same matching red pair. Training
layouts add independent XY jitter of up to 5 mm to both parts. The passive
`hoop_pole` inspection layout remains fixed; `hoop_pole_learning` previews a
seeded training layout. Learning scene version is `hoop_pole_learning_v1`.

The runtime actor receives only side RGB-D, wrist RGB, robot proprioception,
and a symbolic task. Perception and control stay on CPU. Robot-only gravity
feedforward uses the existing actuators and force limits. There is no thread
model or attachment constraint.

Success requires base contact with the hoop bottom within 1 mm of the base top,
center error at most 5 mm, hoop tilt at most 10 degrees, no robot contact,
gripper open, tool at least 4 cm from the hoop center, and low linear/angular
speed sustained for 2 seconds. Evaluation follows a 5-second post-policy hold.
A policy error cannot pass. Returning home is optional. Success scores 100;
partial approach/grasp/lift/alignment/insertion/release milestones guide revision
but do not constitute success.

The first campaign uses `gpt-6-astra`, at most 3 rounds, and a 60,000 reported-token
threshold checked between model calls, not a hard billing cap. A call or provider
overhead may exceed it. The model gets an unscored snapshot of the prior learned
nut-pickup policy to adapt; no old task scores or hand-written manipulation
policies are imported. Development seeds are 6101–6103. Once all three pass,
the best policy is frozen and tested on 6201–6203; held-out results are never
sent back as development feedback. The exact prompt is saved per round.

Start under an allocated GPU job:

```bash
uv run --no-sync python scripts/launch_hoop_learning.py \
  --jobid YOUR_JOB_ID --root outputs/hoop_pole_01 --rounds 3 \
  --reference-policy outputs/fasten_pickup_llm_01/round_002/candidate.py
```

Monitor:

```bash
tail -F outputs/hoop_pole_01/progress.log
```

After the first round, replay its current best recorded trial:

```bash
uv run --no-sync python view_scene.py \
  --vision-trial "$(cat outputs/hoop_pole_01/best_trial.txt)"
```

The sensor-only smoke check in `outputs/hoop_pole_sensor_probe_01` is infrastructure
validation, not a learned candidate, and cannot become this campaign's best policy.


## Smaller opening evaluation

The `tight` variant reduces the hole diameter from 24 mm to 20 mm (about 17%).
The 12 mm pole leaves 4 mm nominal radial clearance instead of 6 mm, a one-third
reduction. Outer diameter, height, total mass, colors, pole, controller and layout
jitter remain the same. Both visual and collision meshes use the smaller bore.
The 32-sector collision bore differs from the visual circle by at most 0.048 mm.

Inspect the smaller pair:

```bash
uv run --no-sync python view_scene.py --scene-only --scene-task hoop_pole --hoop-variant tight
```

Evaluation uses scene identifier `hoop_pole_learning_tight_v1`. Its success
criterion retains the 1 mm clearance margin, reducing the allowed center error
from 5 mm to 3 mm; all other success requirements are unchanged. The runtime
policy still receives only images, robot state and the symbolic goal. Saved
trials restore their own variant automatically; original trials retain their
24 mm geometry and metadata. Omit the variant flag for the original scene.

`outputs/hoop_pole_tight_eval_01` tests an unchanged copy of the round-three best
policy on the same six layouts (6101–6103 development, 6201–6203 previously held
out). This is a geometry transfer evaluation with no retraining. The policy
hash, geometry and per-seed results are saved in `evaluation.json`.

Replay the best tighter-hoop result:

```bash
uv run --no-sync python view_scene.py \
  --vision-trial "$(cat outputs/hoop_pole_tight_eval_01/best_trial.txt)"
```

To run an additional trial under a GPU allocation (use a new output directory):

```bash
MUJOCO_GL=egl uv run --no-sync python scripts/run_vision_trial.py \
  --task hoop_pole --layout sampled --hoop-variant tight --seed 6301 --timeout 300 \
  --candidate outputs/hoop_pole_tight_eval_01/frozen_policy.py \
  --output outputs/hoop_pole_tight_seed6301
```

Completed comparison: **5/6** passed with the 20 mm bore, versus 6/6
with the 24 mm bore. See `outputs/hoop_pole_tight_eval_01/REPORT.md` for
per-seed center errors and failure details.


## Tighter-hoop continuation

`outputs/hoop_pole_tight_learn_01` continues from the original best policy on
`hoop_pole_learning_tight_v1`. It first measures that unchanged policy on six
development layouts (6101–6103 and 6201–6203), retains it as the starting best,
and requests at most three revisions using `gpt-6-astra`. The reported-token
threshold remains 60,000, checked between model calls; it is not a hard cap.
Earlier held-out layouts are now explicitly development data because their
outcomes were inspected. The scene, 3 mm center tolerance, and sensor-only
runtime contract remain unchanged.

After all six development layouts pass, or the revision/token limit is reached,
the best measured policy is frozen. Both it and the starting policy are tested
on the same ten fresh seeds 6301–6310. Those results never enter the revision
prompt. `campaign.json` records their comparison; a higher development score
alone does not establish a higher held-out success rate. A worse revision cannot
replace the starting policy's higher development score.

The saved `run.sbatch` uses one GPU, two CPUs, 16 GB RAM and a one-hour wall limit.
`launch.json` records the Slurm job ID and exact invocation. Campaign state and
trial outputs allow resumption with that script after an interruption.

Monitor from the repository:

```bash
tail -F outputs/hoop_pole_tight_learn_01/progress.log
```

After baseline evaluation, replay the campaign's current best recorded trial:

```bash
uv run --no-sync python view_scene.py \
  --vision-trial "$(cat outputs/hoop_pole_tight_learn_01/best_trial.txt)"
```


## Frozen best policy

The completed tighter-hoop continuation produced
[`policies/insert_hoop_tight_v1.py`](../policies/insert_hoop_tight_v1.py).
Round one passed **6/6 development layouts and 10/10 fresh held-out layouts**.
The original policy passed 5/10 on those same fresh layouts with the same smaller
hoop. See the [versioned evidence and replay](evaluations/hoop_pole_tight_v1/README.md)
for exact results, source hashes and commands that work from a fresh checkout.
