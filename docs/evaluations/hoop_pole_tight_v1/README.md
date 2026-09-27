# Frozen tighter-hoop policy

The exact round-one policy is `policies/insert_hoop_tight_v1.py`. It passed
6/6 development trials and 10/10 fresh held-out trials, compared with 5/6 and
5/10 for the original frozen policy. Both policies used the same 20 mm hoop
opening, 12 mm pole and 3 mm center-error threshold. Sixteen tested layouts
are evidence of this run, not a guarantee of general reliability.

Development seeds: 6101–6103 and 6201–6203. Fresh held-out seeds: 6301–6310.
The previously inspected 6201–6203 results were explicitly moved into development.
The best policy was frozen before either policy's fresh comparison; those
held-out outcomes never entered the revision prompt. The generated policy uses
live RGB-D and robot state; private evaluator metrics are post-trial feedback.

`manifest.json` records the exact policy hashes, geometry, results and checksums.
The result JSON files and generation prompt/response are byte-for-byte copies
of the original artifacts. The prompt retains its historical local paths.
One complete physics replay is included; full sensor captures and the remaining
rollouts are retained in the original local campaign under `outputs/`.

From the repository root, replay the saved held-out success without another model call:

```bash
uv run --no-sync python view_scene.py \
  --vision-trial docs/evaluations/hoop_pole_tight_v1/replay_seed_6301
```

Run the frozen policy in a new trial on a machine with EGL rendering support:

```bash
MUJOCO_GL=egl uv run --no-sync python scripts/run_vision_trial.py \
  --task hoop_pole --layout sampled --hoop-variant tight --seed 6401 \
  --timeout 300 --candidate policies/insert_hoop_tight_v1.py \
  --output outputs/hoop_tight_frozen_seed6401
```

The archived predecessor is `policies/archive/insert_hoop_standard_v1.py`.
Its name describes the original training geometry; its comparison here was
measured on the tighter scene. Neither source was edited when packaging.
