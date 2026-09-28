# Sparse observation replays

Six selected successful Full examples from the primary studies. These videos
show archived observations, not continuous recordings. They illustrate task
behavior; the complete result tables include all successes and failures.

| Video | Seed | Policy | Archived side frames shown | Duration |
| --- | ---: | --- | ---: | ---: |
| [single_policy: Bridge + cylinder](single_policy_bridge.mp4) | 2032101001 | Chain 1, retained R2 | 32/37 | 37 s |
| [single_policy: Three-cube stacking](single_policy_stacking.mp4) | 2029091001 | Chain 1, retained R3 | 22/22 | 27 s |
| [single_policy: Ring-on-peg](single_policy_ring.mp4) | 2031109002 | Chain 1, retained R3 | 12/12 | 17 s |
| [online: Bridge-tower](online_tower.mp4) | 2028105002 | Online Full | 2/2 | 15 s |
| [online: Ring-on-peg](online_ring.mp4) | 2033101001 | Online Full | 12/12 | 18 s |
| [online: Tower relocation](online_relocation.mp4) | 2034101005 | Online Full | 2/2 | 15 s |

## Reading the replays

- Timestamps are original simulation time. Equal display holds do not preserve real-time duration; playback must not be used to compare speed.
- Repeated same-time observations are deduplicated. Up to 32 distinct external-camera frames are sampled evenly in frame order; no intermediate motion is generated.
- Each online clip ends with a labeled re-render of its archived terminal state. This is not an image delivered to the agent. Only forward kinematics/rendering is run, without advancing physics.
- Bridge-tower and Relocation have only two archived external observations in the selected episodes. Their intermediate actions are not reconstructable from these images.
- Single-policy clips use the final training-retained policy, which can be an earlier revision. Online Ring includes its shared grasp/lift observations.
- Success labels come from original result records, not a new assessment of these videos. The final camera observation may precede task verification.
- Video canvases are 1280 x 720; original camera images are 320 x 240 or 640 x 480. Enlarging them does not add sensor detail.

## Selection and provenance

Full successful examples only: first successful seed of the first chain with a successful final training-retained policy; online most archived side observations, tie lowest seed.

[index.csv](index.csv) identifies each example; [manifest.json](manifest.json)
records source and video hashes, frame timestamps, and selection rules.
Source paths identify archives in the research repository, not files included
in this submission bundle. No experiment was rerun to produce these clips.

[Preview contact sheet](preview.png)
