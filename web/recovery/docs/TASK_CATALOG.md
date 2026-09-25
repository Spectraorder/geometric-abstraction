# Task catalog

As of base commit 3a2a48a plus this branch. All current execution evidence is simulation evidence. Historical reported results are not rerun by this inventory.

## Three-cube tower (tower3)

Status: **validated**. Input: RGB-D + proprioception.

198/200 reserved random layouts; 10/10 nominal. Three upright 3 cm cubes; fixed colors; base relocation allowed.

Entry point: `policies/stack_three_blocks_visual_geometry.py`.

Evidence: [docs/VISUAL_RETRY_POLICY.md](https://github.com/Spectraorder/aspire_so100/blob/codex/dual-system-benchmark-english/docs/VISUAL_RETRY_POLICY.md).

## Upright prism with clutter removal (prism_clutter)

Status: **validated**. Input: RGB-D + proprioception.

47/50 reserved random layouts. Purple slender prism on blue cube; up to two blockers; initial upright, zero yaw.

Entry point: `policies/stack_prism_upright_visual.py`.

Evidence: [docs/PRISM_CLUTTER_POLICY.md](https://github.com/Spectraorder/aspire_so100/blob/codex/dual-system-benchmark-english/docs/PRISM_CLUTTER_POLICY.md).

## Select the correct shape among same-color objects (inventory_prism)

Status: **validated**. Input: RGB-D + proprioception.

78/80 reserved: 9/10 at 2 objects, 50/50 at 6, 19/20 at 10. Red slender prism on blue cube from 20-object catalog. Shape inventory is not arbitrary-goal competence.

Entry point: `policies/stack_inventory_cpu.py`.

Evidence: [docs/INVENTORY_CPU_POLICY.zh-CN.md](https://github.com/Spectraorder/aspire_so100/blob/codex/dual-system-benchmark-english/docs/INVENTORY_CPU_POLICY.zh-CN.md).

## Reobserve after transient target occlusion (transient_occlusion)

Status: **pilot**. Input: RGB-D + proprioception.

2/2 development seeds in the historical precise-mask experiment. Image-space fault; does not establish physical-occluder robustness.

Entry point: `research/recovery_lab_20260924/policies/reobserve_then_stack_v2.py`.

Evidence: [research/recovery_lab_20260924/docs/04_LOCAL_RESULTS.zh-CN.md](https://github.com/Spectraorder/aspire_so100/blob/codex/dual-system-benchmark-english/research/recovery_lab_20260924/docs/04_LOCAL_RESULTS.zh-CN.md).

## Stop when target evidence stays absent (persistent_occlusion)

Status: **pilot**. Input: RGB-D + proprioception.

Stopped without motion on one development seed. Correct abstention is separate from tower completion.

Entry point: `research/recovery_lab_20260924/policies/reobserve_then_stack_v2.py`.

Evidence: [research/recovery_lab_20260924/docs/04_LOCAL_RESULTS.zh-CN.md](https://github.com/Spectraorder/aspire_so100/blob/codex/dual-system-benchmark-english/research/recovery_lab_20260924/docs/04_LOCAL_RESULTS.zh-CN.md).

## Recover from the first failed gripper close (grasp_miss)

Status: **pilot**. Input: RGB-D + proprioception.

One recovery-lab seed passed. Existing retry skill; not a newly learned model policy.

Entry point: `policies/stack_three_blocks_visual_geometry.py`.

Evidence: [research/recovery_lab_20260924/docs/04_LOCAL_RESULTS.zh-CN.md](https://github.com/Spectraorder/aspire_so100/blob/codex/dual-system-benchmark-english/research/recovery_lab_20260924/docs/04_LOCAL_RESULTS.zh-CN.md).

## Recover from opening the gripper during lift (lift_loss)

Status: **pilot**. Input: RGB-D + proprioception.

One recovery-lab seed passed. 35 mm lift trigger; 450 ms opening; not friction randomization.

Entry point: `policies/stack_three_blocks_visual_geometry.py`.

Evidence: [research/recovery_lab_20260924/docs/04_LOCAL_RESULTS.zh-CN.md](https://github.com/Spectraorder/aspire_so100/blob/codex/dual-system-benchmark-english/research/recovery_lab_20260924/docs/04_LOCAL_RESULTS.zh-CN.md).

## Astra plan and Jev initial-observation decisions (dual_system_gate)

Status: **integration**. Input: RGB-D-derived summary + frozen visual skill.

See live integration report; development only. Initial observation only. Full online grasp/rebuild orchestration is not implemented.

Entry point: `scripts/run_dual_system_trial.py`.

Evidence: [docs/DUAL_SYSTEM.md](https://github.com/Spectraorder/aspire_so100/blob/codex/dual-system-benchmark-english/docs/DUAL_SYSTEM.md).

## Historical single-block and tower policies (legacy_privileged)

Status: **historical**. Input: Privileged object state.

Separate historical demonstrations. Never combine these scores with sensor-only results.

Entry point: `src/aspire_so100/agent_loop.py`.

Evidence: [docs/OBSERVATION_AUDIT.zh-CN.md](https://github.com/Spectraorder/aspire_so100/blob/codex/dual-system-benchmark-english/docs/OBSERVATION_AUDIT.zh-CN.md).

## Repair one tower without disturbing the other (dual_tower_repair)

Status: **design_only**. Input: Proposed RGB-D + proprioception.

Not executed. Needs support DAG, damage localization, and local dismantling.

## Two supports and a horizontal beam (bridge)

Status: **design_only**. Input: Proposed RGB-D + proprioception.

Not executed. Needs payload reorientation and two-contact support evaluation.

## Continue after a person moves a block (human_edit)

Status: **design_only**. Input: Proposed RGB-D + proprioception.

Not executed. Needs external-change detection, association recovery, and handover protocol.

## Reacquire identity after transport-time occlusion (severe_occlusion)

Status: **design_only**. Input: Proposed RGB-D + proprioception.

Not executed. Needs temporal identity and uncertainty-aware active viewpoints.

## Commands for supported tasks

Use a new output directory for every run. macOS uses its native OpenGL backend; set MUJOCO_GL=egl only on a configured headless Linux host.

```bash
uv run python scripts/run_vision_trial.py --task tower --layout random_xy --seed 4001 --candidate policies/stack_three_blocks_visual_geometry.py --output outputs/tower_new
uv run python scripts/run_vision_trial.py --task prism --layout random_xy --seed 405 --candidate policies/stack_prism_upright_visual.py --output outputs/prism_new
uv run python scripts/run_vision_trial.py --task inventory_prism --layout sampled --object-count 6 --seed 604 --candidate policies/stack_inventory_cpu.py --output outputs/inventory_new
uv run python research/recovery_lab_20260924/harness/run_fault_trial.py --fault miss --seed 4001 --candidate policies/stack_three_blocks_visual_geometry.py --output outputs/miss_new
```

Use `--fault slip`, `occlusion`, or `persistent_occlusion` with the recovery harness. The historical image masks are camera-call-based; use the new dual-system runner for time-based initial-occlusion model comparisons. None of these commands executes the proposed bridge or dual-tower repair.

The original single-block pipeline and historical agent loop use privileged object state. Their scores belong to a separate track; see the observation audit before choosing an interface.
