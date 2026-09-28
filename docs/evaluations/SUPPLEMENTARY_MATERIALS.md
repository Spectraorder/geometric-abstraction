# Supplementary results and sparse observation replays

This bundle accompanies the single-policy acquisition and continuous online
experiments. It contains the reported figures, complete exported outcome and
resource tables, and six illustrative replays. It can be browsed without running
code. The paper and its PDF appendix describe the methods and experimental setup.

## Start here

| Material | Main entry | Useful files |
| --- | --- | --- |
| Single-policy acquisition | [Data guide](single_policy/README.md) | [Test SR and cumulative tokens](single_policy/iteration_sr_tokens.pdf), [per-chain curves](single_policy/chain_curves.csv), [all test slots](single_policy/test_outcomes.csv) |
| Continuous online execution | [Data guide](online/README.md) | [Outcomes](online/outcome_bars.pdf), [model interaction cost](online/model_resources.pdf), [success versus time](online/success_vs_time.pdf), [all episodes](online/episodes.csv) |
| Six task replays | [Video guide](videos/README.md) | [Video index](videos/index.csv), [preview](videos/preview.png), [frame provenance](videos/manifest.json) |

PDFs are suitable for reading and printing. PNGs are previews, and SVGs preserve
editable chart text. CSVs contain the values behind the figures.

## Experimental units and resource accounting

- Single-policy: three tasks, five paired training chains per task, five training
  layouts per pair, R0--R3, and 100 shared held-out layouts per task. The bundle
  contains 12,000 test slots and 600 training slots, including failures. Retention
  uses training success only; five policies on the same layouts do not provide
  500 distinct layouts. Use the `all_100` subset for the primary curves.
- Online: three tasks, five paired layouts per task, 30 episodes in total, and
  a 200-second deadline. The scene continues after failed actions. Capped-time
  means include failures as 200 seconds.
- Input-plus-output tokens count cached input once. Interrupted token usage is
  unknown and excluded from recorded totals. Model-request time includes waiting;
  it is not isolated inference compute time. Image attachments are distinct from
  local camera acquisitions.

## What the videos show

The six clips are selected successful Full examples from the primary studies.
They show sparse archived camera observations with their original simulation
timestamps. Uniform display holds and missing frames mean playback duration is
not execution duration. No motion is interpolated.

Online clips additionally show a labeled render of the archived final simulator
state. Bridge-tower and Relocation retain only two external observations in the
selected episodes, so these are short visual summaries of those records.
Single-policy clips use the policy retained at the end of training. Success
labels come from the original evaluator records. Selection details and source
hashes are in the video guide; the complete tables retain every evaluation.

## Provenance and scope

`single_policy/provenance.json` and `online/provenance.json` identify the original
archives and exported data by SHA-256. Repository-relative `source` paths are
archive identifiers; raw observations, policy code, prompts, and private physics
traces are not distributed in this bundle. The package manifest lists included
files and their hashes. No training, model requests, or evaluation rollouts were
run to produce this supplement.
