# Single-policy acquisition experiments

This document collects the experimental setup for the workshop paper. The
[consolidated results report](SINGLE_POLICY_RESULTS.md) and
[data bundle](evaluations/single_policy/README.md) contain the completed outcomes,
curves, and resource accounting. The [LaTeX fragment](paper/single_policy_experiments.tex)
provides setup and results for the paper. Online reasoning is reported separately.

The primary studies are the new contract-only bridge study
`geometry_bridge_contract_01`, contract-only stacking study
`geometry_cube_stack_contract_01`, and paired ring study `ring_peg_comparison_01`.
All three completed 4,000 evaluation slots. Earlier source-disclosed comparisons,
capable-starter runs, and Full-only ring pilots remain separate historical or
development evidence; they are not additional independent comparisons here.

## Main-text draft

### Single-policy acquisition

We study whether supplying a geometric abstraction helps a language-model agent
acquire an executable manipulation policy from a small number of training
rollouts. The agent generates and revises a Python program, which obtains current
observations and executes through a robot interface. Learning changes the
program, while the supplied geometric estimators and operators remain fixed.
This setting tests policy acquisition and generalization to new layouts. Our
separate online experiments study decisions made by the language model during
continuous interaction.

**Tasks and environment.** We use a simulated SO-100 tabletop robot for three
tasks: constructing a bridge, stacking three cubes, and placing a ring over a
fixed peg. The bridge uses two cube foundations, a spanning rectangular beam,
and a cylinder placed on the beam. Stacking requires an ordered three-cube tower.
Ring placement requires grasping material around an opening and moving the
opening over the peg before release. Each task randomizes layouts and color
assignments. Test layouts are held out from training within each study. These
experiments evaluate layout generalization for the specified tasks and object
families.

**Comparison.** Base and Full use the same language model, task specification,
calibrated RGB-D access, public robot information, and action interface within
each task. Base can implement numerical perception in its program. Full
additionally receives supplied sensor-grounded geometry and textual feedback;
for bridge and stacking, this includes symbolic references and geometric
operators, while the ring interface exposes opening and wall geometry. Runtime
policies cannot access simulator object poses, dimensions, contacts, or private
evaluator state. Both arms receive task outcomes as training feedback. This
comparison evaluates the supplied geometric interface as a package, including
perception, queries, and feedback.

**Training.** Each task uses five paired training chains. A pair shares five
training layouts, with disjoint training layouts across pairs. Base and Full
have separate persistent language-model contexts in every pair. We use
`gpt-6-astra` with high reasoning effort and three generated revisions, R1–R3.
R0 is an observation-only scaffold; R1 is the first generated manipulation
program. Each version is run on all five training layouts. The scene resets
between rollouts, and the program executes with local sensing and control
without further language-model calls within a rollout. The agent receives its
own training outcomes, bounded diagnostic feedback, and images under the study's
feedback rule before proposing the next revision. This gives 30 completed
program generations and 200 training slots per task, including R0.

**Training-only retention.** We maintain an incumbent policy to prevent accepted
revisions from losing previously successful training layouts. A candidate
replaces the incumbent only if its training success set is a strict superset of
the incumbent's success set. Ties and proposals that lose an earlier success
retain the incumbent. We preserve every generated candidate and report both
raw-candidate and retained-policy curves. Retention never uses test outcomes,
and it does not guarantee monotonic test performance.

**Evaluation.** After all training chains are complete and their candidates are
frozen, every saved version is evaluated on the same 100 held-out layouts per
task. The resulting matrix contains 4,000 evaluation slots per task:
2 arms × 5 chains × 4 versions × 100 layouts. Invalid programs count as failed
slots. Task success is determined by a private physical evaluator: bridge and
stacking require the prescribed support structure and a five-second stability
check; ring placement requires the peg to pass through the opening, the ring
center to lie below the peg top, release from robot contact, and a two-second
stability check. Test execution makes no language-model calls and cannot revise
or select policies.

**Measures and study differences.** We report success against revision number and
cumulative model resources, together with paired chain variation. The five
policies in each arm share the same 100 test layouts; their 500 evaluations per
version are not 500 distinct layouts. Resource measures include input and output
tokens, cached input, image attachments, model-request time, and rollout cost.
The revision budget is matched within each task; actual token use is measured.
All three primary studies provide public interfaces without executor source.
Ring uses explicit pad/jaw commands and a different image-feedback rule. Bridge
and Ring registered their full evaluation matrices before training; Stacking's
expansion from 20 to 100 layouts was a post-hoc extension without retraining.
Ring also preserved two interrupted requests and an authorized transport
amendment. These differences delimit comparisons across tasks.

## Common configuration and counting units

| Setting | Value per task |
| --- | --- |
| Training model | `gpt-6-astra`, high reasoning effort |
| Context | Persistent within a chain; separate across arms and replicas |
| Independent paired chains | 5 |
| Training layouts | 5 per pair, 25 distinct layouts overall |
| Policy versions | Observation-only R0; generated R1, R2, R3 |
| Completed program generations | 2 × 5 × 3 = 30 |
| Training slots | 2 × 5 × 4 × 5 = 200 |
| Final test layout count | 100 shared across all chains and versions |
| Evaluation slots | 2 × 5 × 4 × 100 = 4,000 |
| Source limit | 50,000 characters |
| Model request timeout | 900 seconds |
| Policy execution timeout | 120 seconds |
| Runtime computation | CPU perception, planning and control; GPU rendering permitted |
| Policy selection | Strict training-success-set improvement |
| Test-time model calls | None |

A slot is a registered candidate/layout evaluation. A rejected source can consume
its slot without executing a physical rollout. Report slot completion separately
from the number of simulations actually executed. The 120-second timeout is the
policy execution limit, not a promise that complete worker wall time, including
setup and private final evaluation, is at most 120 seconds.

The replication unit for training is the paired chain. Chains differ in training
layouts and model generations; chain variation therefore includes both factors.
Within a task, test layouts are crossed with all chains and revisions. Results
should preserve this pairing rather than treating all rollout rows as independent
replicates. Five pairs support descriptive variability; they do not justify
precise claims about training variance by themselves.

## Differences to retain in the paper or appendix

| Component | Bridge | Three-cube stacking | Ring over peg |
| --- | --- | --- | --- |
| Main study | `geometry_bridge_contract_01` | `geometry_cube_stack_contract_01` | `ring_peg_comparison_01` |
| Task objects | Two cubes, beam, cylinder | Three cubes | Perforated hexagonal ring and fixed cylindrical peg |
| Shared action assistance | Visual pick/place actuator | Visual pick/place actuator | Explicit fixed-pad targets, wrist roll, jaw commands, robot-only kinematic previews |
| Executor source in actor prompt | Withheld from both arms | Withheld from both arms | Withheld from both arms |
| Full geometry interface | Current metric state, references, relations, alignment, checkpoints | Same interface family, retaining the existing cube observer | Sensor-grounded object observations with opening and wall geometry |
| Supplied starting information | Declarative goal and API scaffold | Declarative goal and API scaffold | API scaffold and shared qualitative ring/robot guidance |
| API-call limit per rollout | 1,200 | 1,200 | 600 |
| Later image feedback | Failure-triggered | Failure-triggered | Two training-layout side images each revision when available |
| Evaluation registration | All versions on 100 layouts before training | 20 layouts first; 80 added later | All versions on 100 layouts before training |

The bridge and stacking actuator performs visual object localization for picking
in both arms. Base thus retains useful shared sensing and motor assistance even
without the Full query interface. Both primary studies withhold executor source.
Historical source-disclosed Bridge and Stacking runs are preserved separately;
new training chains and layouts mean differences between those studies and the
primary comparisons do not isolate source disclosure causally.

Ring uses a different assistance level: its actuator executes explicit robot
commands and does not choose a ring-wall grasp or replace a submitted target with
a task-solving coordinate. Both arms receive qualitative guidance about material
versus empty space, three-dimensional jaw closure, and height-aware clearance.
Neither receives the capable development policy, a grasp-search procedure, or an
oracle manipulation sequence. The starting point is observation-only, but it is
not devoid of task knowledge.

The ring has 72 mm outer width across flats, a 40 mm hexagonal opening, and 15 mm
height. The peg has 16 mm diameter and 50 mm height. The nominal centered lateral
clearance is 12 mm per side. This is a wide-clearance placement task, not a
precision insertion benchmark. These are environment specifications; runtime
metric object estimates must come from observations.

### Image and feedback rules

Bridge and stacking give the initial revision visual context from up to two
training examples, using available side RGB and rendered depth images with
calibration. Later revisions attach up to two RGB views from one eligible
failure layout, selected using the registered failure rule; otherwise they add
no images. The rule is shared, but differing failures can produce different
attachment counts across arms. Feedback includes bounded own-policy diagnostics
and permitted records from the five training layouts.

Ring attaches the latest side image from each of the first two registered
training layouts for each generated revision, when available. Feedback includes
each layout's binary outcome, notes, and at most four latest permitted records.
Full records may include supplied geometry; Base records do not. Local camera
acquisitions and images attached to a model request are separate quantities.
The main-text comparison should not describe all three studies as using the same
image schedule or Full as receiving text alone.

## Curve definitions and resource reporting

Let $S_{a,j,k}$ be the training-layout success set for arm $a$, chain $j$, and
candidate revision $k$. The retained revision is initialized as $b_{a,j,0}=0$ and
updated by

$$
b_{a,j,k}=\begin{cases}
k,& S_{a,j,k}\supsetneq S_{a,j,b_{a,j,k-1}},\\
b_{a,j,k-1},&\text{otherwise}.
\end{cases}
$$

For test outcome $y(\pi,L)\in\{0,1\}$, raw and retained success rates are

$$
s_{a,j,k}=\frac{1}{100}\sum_{\ell=1}^{100}y(\pi_{a,j,k},L_\ell),
\qquad
s^{\mathrm{ret}}_{a,j,k}=s_{a,j,b_{a,j,k}}.
$$

Plot the mean over five chains and show chain variation or paired differences.
A retained curve must use the training-selected candidate at each revision; it
must not take the cumulative maximum of test success. R0 represents an inactive
scaffold and should not be described as zero-shot model performance. R1 is the
first generated solution after observing R0 feedback.

For curves against acquisition cost, include every attempted proposal up to that
revision, including rejected candidates. Difference persistent-session cumulative
token counters between completed turns before summing. Cached input is a subset
of input tokens, and reasoning output is a subset of output tokens; neither
should be added a second time. Report incomplete attempts separately whenever
usage cannot be recovered. Distinguish total worker/request time, queue time,
individual rollout latency, and parallel campaign wall time.

Ring preserved two approximately 900-second interrupted WebSocket requests with
unknown final token usage. Its two remaining generations used HTTP/SSE with the
same model, reasoning setting, prompts, and completed history. This yields 30
completed generations and 32 outer request attempts. Resource comparisons must
retain the interrupted time and disclose the transport amendment. Completed-turn
tokens alone do not measure the total cost of that campaign.

These measures address policy acquisition cost and the resulting programs. They
do not measure online language-model decision latency or establish that text
reduces end-to-end time during continuous interaction.

## Evaluation provenance

| Study | Training seeds | Final test seeds | Final evaluation evidence |
| --- | --- | --- | --- |
| Bridge | 2032100101–2032100125 | 2032101001–2032101100 | `outputs/geometry_bridge_contract_01/learning_curves/` |
| Stacking | 2029090101–2029090125 | 2029091001–2029091100 | `outputs/geometry_cube_stack_contract_01/evaluation_100/` |
| Ring | 2031101001–2031101025 | 2031109001–2031109100 | `outputs/ring_peg_comparison_01/` |

Bridge and Ring registered all versions on 100 layouts before training.
Stacking reused 800 evaluations on its original 20 layouts and added 3,200
on 80 new layouts after inspecting the first results. This extension changed no
policies and supplied no test feedback to training. Its additional 80-layout
results are reported alongside the combined 100-layout results in the
[results report](SINGLE_POLICY_RESULTS.md). The historical source-disclosed
Bridge's post-hoc learning-curve extension is not the primary Bridge study here.

The structure evaluator checks the required support graph, relative placement,
orientation, contacts, and low velocities. Its five-second final check uses the
historical contact helper; a later development diagnosis found that this helper
can miss unnamed robot collision geometry. Preserve that limitation when
interpreting the frozen bridge and stacking results. Ring uses body-ancestry
robot-contact checking, verifies peg-axis passage through the opening and ring
center height, and checks low motion for two seconds with at most 3 mm endpoint
drift. A geometric relation or successful action return alone is not task success.

The consolidated results verify the final summaries, keep raw and retained
outcomes distinct, and include failures and invalid candidates. The
single-policy studies support claims about these tasks and their layout splits;
they do not establish general unseen-shape reconstruction or cross-task transfer.

## Protocol and implementation references

- [Contract-only Bridge protocol](BRIDGE_CONTRACT_ONLY_STUDY.md) and its [learning-curve registration](../outputs/geometry_bridge_contract_01/learning_curves/registration.json).
- [Contract-only stacking protocol and 100-layout amendment](THREE_CUBE_CONTRACT_ONLY_STUDY.md).
- [Paired ring protocol and transport amendments](RING_PEG_COMPARISON_STUDY.md).
- [Shared qualitative ring guidance](RING_PEG_STARTER_GUIDANCE.md).
- [Training-only retention](../scripts/geometry_iteration.py) and [shared geometry experiment feedback](../scripts/run_constraint_geometry_experiment.py).
- [Task definitions](../src/aspire_so100/structure_tasks.py), [structure evaluation](../src/aspire_so100/inventory_structure.py), and [ring scene and evaluator](../scripts/ring_peg_scene.py).
- [Geometric abstraction methodology](GEOMETRIC_ABSTRACTION_METHODOLOGY.md).
