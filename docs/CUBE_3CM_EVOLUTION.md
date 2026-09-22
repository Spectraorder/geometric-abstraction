# Improving the 3 cm cube policy

This is a historical experiment. See the [current cube-and-gripper policy](VISUAL_RETRY_POLICY.md)
for the latest validated results.

The new policy is
[`stack_three_blocks_cubes_3cm.py`](../policies/stack_three_blocks_cubes_3cm.py).
Its SHA-256 is
`acbbb2796542f1ad55106adf4d5f0a9ca368089adf54256491ec21c75416f43c`.

| Policy | Training: 101–105, 1001–1005 | Reserved: 2001–2010 |
| --- | --- | --- |
| Original geometry policy | 4/10 | 4/10 |
| Improved 3 cm policy | **8/10** | **9/10** |

Both policies were evaluated on the same reserved layouts. The improved policy
preserved all four baseline successes and additionally passed 2002, 2004, 2005,
2007, and 2008. Only 2006 failed. The candidate was selected and frozen before
reserved evaluation, and no changes were made afterward.

Seeds 1001–1005 were validation seeds in the earlier experiment. Their outcomes
were already known, so they were explicitly included in training here; 2001–2010
were reserved before this search began. These small-sample results do not
establish reliability across every possible layout.

## Final changes

The selected candidate retains the original geometry fitter and sensor-only API.
The final repair changes motion parameters rather than perception:

- Grasp at 70% of the camera-measured cube height above its centroid, instead
  of 30%—approximately 21 mm versus 9 mm for these cubes. This increases finger
  clearance around supporting blocks.
- Use 0.65 times the measured red release drift to compensate green placement,
  instead of 1.35, and cap compensation at 20% of measured cube width rather
  than 30%.
- Target 3 mm final contact clearance instead of 1 mm, and stop correction
  within the existing 4 mm upper release-gap limit.

The geometry, mass, friction, robot, success criteria, and numerical visual fitter
were unchanged. Object sizes, centers, vertices, and release-drift compensation
continue to come from RGB-D. There are no actions keyed to seed numbers or
memorized object positions. The frozen source passed the trial runner's source
validator, and its geometry-helper prefix is identical to the baseline's.

## Search record

Eleven agent-authored candidates were evaluated on the fixed ten training seeds.
Each hypothesis, complete source, and trial evidence is retained locally.
No separate proposal-model calls were used in this search.

| Candidate | Training passes | Main experiment |
| --- | --- | --- |
| Baseline | 4/10 | Original geometry policy |
| 01 | 4/10 | Grasp height 42% |
| 02 | 4/10 | Re-grasp and re-center red using measured release drift |
| 03 | 4/10 | Refresh blue XY from visible side planes |
| 04 | 3/10 | Combine red re-centering with base refresh |
| 05 | 6/10 | Grasp height 55% |
| 06 | 7/10 | Grasp height 70% |
| 07 | 8/10 | Grasp height 85% |
| 08 | 8/10 | Grasp height 70%, smaller release compensation |
| 09 | 6/10 | Grasp height 70% plus red re-centering |
| **10** | **8/10** | Candidate 08 plus 3 mm final clearance |
| 11 | 7/10 | Grasp height 70%, intermediate compensation strength |

Selection maximized full training successes, then mean partial-credit score.
Candidate 10 won the tie with score 0.85 versus 0.80 for candidates 07 and 08.
The more complicated re-grasping/base-refresh branches were not promoted.

## Remaining failures

Training failures were 104 and 1004. On reserved seed 2006, red placement did not
converge within the motion budget: its final camera-estimated gap was about
9.9 mm, outside the release gate. The policy returned the payload to its source
instead of completing the tower. This failure was recorded, not used for tuning.

## Evidence and replay

The experiment is saved under `outputs/cube_3cm_evolution_01/`:

- `experiment.json`: declared splits, baseline, all candidate results,
  selection rule, frozen hash/time, and paired reserved results.
- `candidate_01/` through `candidate_11/`: hypotheses, policy sources,
  per-seed sensor frames, API logs, traces, and physics rollouts.
- `initial_policy.py` and `best_policy.py`: frozen comparison sources.
- `reserved/baseline/` and `reserved/improved/`: paired validation results.
- `reserved/improved/seed_2002/demo.mp4`: successful 2× replay on a seed
  where the baseline failed.

This search executed 110 training trials and 20 reserved trials. The ten original
baseline training results were reused from the earlier 3 cm transfer experiment.

Replay saved physics using the current 3 cm scene:

```bash
uv run python view_scene.py \
  --vision-trial outputs/cube_3cm_evolution_01/reserved/improved/seed_2002
```

Run the improved policy on a selected layout:

```bash
uv run python view_scene.py --vision-policy policies/stack_three_blocks_cubes_3cm.py \
  --vision-layout random_xy --seed 2002 --vision-output outputs/cube_3cm_view_2002
```

The new output directory must not already exist.
