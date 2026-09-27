> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# RGB-D tower policy for 4 cm cubes

This historical experiment used a 4 cm scene. The current scene uses 3 cm cubes;
see the [current policy report](VISUAL_RETRY_POLICY.md).

On 2026-09-21 (America/New_York), five LLM proposal/evaluation rounds evolved
the saved randomized RGB-D policy for the new 4 × 4 × 4 cm blocks. The starting
policy passed 0/5 training layouts. The promoted policy is
[`stack_three_blocks_cubes_4cm.py`](../../policies/stack_three_blocks_cubes_4cm.py).

SHA-256: `4cc3b4ccd976a91ceecc98d016064468313e9c0c968614cdca77133109bd7c8b`.

## Results

| Evaluation | Seeds | Passes |
| --- | --- | --- |
| Training, random XY | 101–105 | 5/5 |
| Reserved, random XY; frozen policy | 201–205 | 4/5 |
| Nominal-layout validation; same frozen policy | 81–83 | 2/3 |

The policy was frozen after training and was not revised using either validation
set. Seeds 201 and 202 had previously been evaluated on the older block geometry;
these results are fresh evaluations of the frozen policy on 4 cm cubes, not a
claim that those seed numbers have never been used before.

## Evolution

| Round | Training passes | Main change |
| --- | --- | --- |
| Baseline | 0/5 | Existing randomized policy |
| 1 | 1/5 | Lower release clearance, slower/wider opening, staged withdrawal; retain the unobstructed blue-top measurement |
| 2 | 2/5 | Grasp higher, by 30% of the camera-measured block height above its center |
| 3 | 4/5 | Measure red's release-to-settled XY shift and compensate the green placement |
| 4 | 4/5 | Increase the compensation gain to 1.35 and cap it at 30% of measured block width |
| 5 | 5/5 | Accept camera-estimated contact gaps down to −5 mm in final alignment/release checks |

The final change affects the policy's perception-based release gate, not the
physical evaluator. Success criteria, simulator dynamics, and the 4 cm geometry
were unchanged throughout evolution. The policy obtains object positions and
dimensions from RGB-D, uses robot proprioception, and has no seed-dependent
actions or access to evaluator state. Proposals used prior code, sensor images,
and reduced post-trial feedback; the recorded model events passed the runner's
tool-use audit. The proposal model was `gpt-6-astra`.

## Remaining failures

- Randomized seed 205 failed the evaluator's two-block-prefix support margin on
  blue. Its full three-block structure passed the five-second stability check,
  but the evaluator also checks the red-on-blue prefix at the final instant.
  That stricter criterion was retained, and the result remains a failure.
- Nominal seed 82 failed support-margin checks and exceeded the 3 mm green-block
  drift limit during the final stability window.

These observations came from validation after freezing and were not fed back
into the learner. The small sample does not establish reliability across all
layouts.

## Artifacts and replay

Local evidence is under `outputs/cube_4cm_evolution_01/`:

- `campaign.json`: lineage, hypotheses through linked responses, policy hashes,
  training results, reserved results, and usage counters.
- `round_001/` through `round_005/`: prompts, model events, complete candidates,
  sensor frames, trial traces, API calls, and recorded physics states.
- `heldout/` and `nominal_validation/`: frozen-policy validation results.
- `heldout/seed_201/demo.mp4`: a successful reserved-layout replay at 2×.
- `initial_policy.py` and `best_policy.py`: unchanged baseline and frozen winner.

The original baseline evidence is under
`outputs/cube_4cm_eval_20260922T015205Z/` and was reused rather than rerun.
The five proposal calls reported 90,260 input and 20,254 output tokens; these
are CLI usage counters, not a billing estimate.

Replay saved physics without executing the policy again:

```bash
uv run python view_scene.py \
  --vision-trial outputs/cube_4cm_evolution_01/heldout/seed_201
```

Run the promoted policy on a selected layout and save its rollout:

```bash
uv run python view_scene.py --vision-policy policies/stack_three_blocks_cubes_4cm.py \
  --vision-layout random_xy --seed 201 --vision-output outputs/cube_4cm_view_201
```

The new output directory must not already exist.
