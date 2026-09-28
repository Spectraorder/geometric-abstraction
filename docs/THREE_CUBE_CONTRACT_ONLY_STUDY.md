# Three-cube policy learning with actuator contracts only

This new study follows the completed [source-disclosed comparison](THREE_CUBE_GEOMETRY_STUDY.md).
Inspection found that all five Base initial prompts contained the common executor
source, including color masks, RGB-D deprojection, table fitting, and top-surface
geometry estimation. The runtime Base interface withheld object geometry, but
its actor could use that implementation as a reference when writing its own policy.
The earlier results remain valid for that stronger, source-disclosed baseline.

## Intervention and unchanged components

Neither arm receives the executor implementation in its initial prompt. Both
retain its public call signatures, units, status semantics and failure behavior.
The runtime actuator is unchanged. Both arms retain calibrated side/wrist RGB-D,
public robot information, declarative support goals, qualitative action feedback,
and permission to write perception code using NumPy/SciPy within 50,000 characters.
Full additionally receives the existing geometry API and geometric checkpoints.

The initial observation-only policy, cube observer, 0.40 grasp fraction, libccd
physics, 120-second rollout timeout, Astra high model, persistent actor contexts,
image schedule, strict training-only retention, and success evaluator are unchanged.
The common actuator still localizes objects for picking; this is not a comparison
of complete end-to-end perception and control systems.

## Registered budget

- Output: `outputs/geometry_cube_stack_contract_01`.
- Five independent paired training chains; five training layouts per pair.
- Training seeds: 2029090101–2029090125, disjoint between pairs.
- R0 observation-only and three generated revisions R1–R3 per arm.
- Thirty model generations and 200 training rollout slots, including R0.
- Twenty fresh shared reserved layouts: 2029091001–2029091020.
- Evaluate every frozen revision after all training and context audits finish:
  5 pairs × 2 arms × 4 revisions × 20 layouts = 800 evaluations.
- Up to four simultaneous model requests and 24 simulator workers globally, with
  the existing memory-dependent fallbacks. One BLAS/OpenMP thread per worker.

No reserved results enter training or policy retention. Invalid policies consume
their registered revision slots. Infrastructure failures stop dispatch and retain
incomplete attempts; no automatic retries or replacement layouts are permitted.
All outcomes, including regressions, will be reported.

Report raw and training-retained success curves, cumulative tokens (with cached
input already included), image attachments, request times and rollout costs.
There are 20 distinct test layouts, not 100 independent layouts per arm/revision.
The historical comparison uses different chains and layouts; a change relative
to it cannot isolate the causal effect of removing source disclosure.

## Validation and execution

The new context adapter removes exactly one executor-source block and rejects a
missing or duplicated block at the first turn. Follow-up feedback remains unchanged.
Archived prompts and images are reconstructed during training audits. The known
object-count audit correction and verified cumulative token transitions are
installed in the new runner before training; historical runtime files are unchanged.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python \
  scripts/run_three_cube_contract_study.py \
  --output outputs/geometry_cube_stack_contract_01 --phase prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -u \
  scripts/run_three_cube_contract_study.py \
  --output outputs/geometry_cube_stack_contract_01 --phase all
```

The `all` phase trains all chains, freezes candidates, materializes the registered
800-test matrix, and runs evaluation automatically. Inspect `progress.json`,
`lifecycle.json`, each replica's `training.log`, and `learning_curves/progress.json`.

## Authorized 100-layout extension

After inspecting the completed 20-layout results, the user requested 100 shared
test layouts. The separate `evaluation_100/registration.json` preserves the old
protocol, policies, training retention choices and all 800 existing results.
It adds seeds 2029091021–2029091100 to the original 2029091001–2029091020 and
registers all five pairs and all R0–R3 revisions: 4,000 results total, 3,200 new
rollouts, no model calls or retraining. Up to 24 rollout workers are used, with
the existing hardware pilot and fallback. Additional color maps are stored in
extension-only manifests; the original manifests remain unchanged.

This is a post-hoc sample-size extension, not a new preregistered confirmation.
Report both the combined 100-layout curves and the additional 80-layout results.
Do not select policies using either test split. All outcomes and incomplete attempts
remain recorded; infrastructure errors stop dispatch without automatic retries.
Use `scripts/extend_three_cube_contract_evaluation.py --root
outputs/geometry_cube_stack_contract_01 --phase prepare` followed by `--phase run`.
Progress and results live under `evaluation_100/`.

The completed extension required a log-path recovery after 24 new trials had
already produced valid results. `resume_three_cube_contract_extension.py` preserves
those results and the original stop receipt while adding the expected path aliases.
`run_repaired_cube_extension.py` additionally audits the failed log-only leases
against the preserved result hashes. These are registered recovery adapters for
the existing run; do not restart its original dispatcher or overwrite completed
trials. Recovery receipts live in `evaluation_100/path_repair/`.
