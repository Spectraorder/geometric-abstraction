# Five-layout online geometric-text comparison at 200 seconds

This is a separate bounded comparison under the continuing experiment objective,
following the user's five-layout scope and 200-second budget. It evaluates the
complete compact Full interface against Base; it does not rerun the completed
180-second study or estimate a pure serializer effect from different layouts.

Status: completed all ten slots in 611.62 seconds. Final source/resource,
interface and compact-prompt audits passed. Full succeeded on 4/5 layouts and
Base on 2/5; Full had lower mean capped completion time but higher recorded token
use. See [all outcomes and limitations](ONLINE_COMPACT_TOWER_200_RESULTS.md).

## Fixed design

- Five new paired seeds: 2028105001–2028105005, selected without outcome screening.
- Base and Full once each per layout: ten episodes, no replacements or repeats.
- 200 seconds per episode after the sole initial reset, including initialization,
  observation, model requests, actions and settling. No reset after failure.
- At most four workers in paired 4/4/2 batches, one BLAS/OpenMP thread each.
- Ordinary unassembled bridge plus two stacked cubes; no injected collapse,
  support displacement, loaded-stack recovery or artificial stress fixture.
- Identical Astra medium routing, persistent contexts, public support graph,
  adapted bridge/stack skills, numeric target action schema and physical evaluator.
- Both arms have calibrated RGB-D pixel queries and image requests. Base receives
  its normal side-image and qualitative feedback. Full receives current typed
  references encoded with shared metadata, plus initial/requested images.
- The same compact actor wrapper runs in both arms. It leaves Base prompts
  byte-identical and encodes only Full's geometry. It preserves values and evidence;
  unavailable references do not inherit a pose or observation claim.
- Actors choose worksite, order and explicit support-surface coordinates. The
  shared actuator does not compute or replace task destinations.

The choice of compact text follows component tests, exact development-prompt
replay and the completed 200-second raw/compact Full development pair. Both
conditions succeeded in that pair, with fewer compact tokens but slightly longer
time. This evidence is separate from evaluation and does not presume a favorable
Base/Full outcome. See [development validation](ONLINE_COMPACT_GEOMETRY_TEXT.md).

## Outcomes and stopping

Report success out of five per arm, the mean capped completion time with every
unsuccessful episode counted at 200 seconds, and every paired outcome. Also report
input, cached-input and output tokens; incomplete requests with unknown usage;
model-request durations; image attachments; metric queries; camera acquisitions;
action/feedback time; and campaign wall time. Do not double-count cached input.
Do not sum overlapping timer categories as independent wall-clock components.

The same private support/contact and five-second stability criterion defines
success in both arms. Private geometry and evaluator state never enter model
feedback. Flag excessive simulation lag above two seconds. Five paired layouts
are exploratory, with no guarantee of a stable population-level advantage.

Infrastructure errors stop new batches after current workers finish. Preserve all
attempts and STOP receipts. There are no automatic retries, seed substitutions,
budget extensions or extra episodes based on observed outcomes. Once these ten
slots are terminal, analyze and report this fixed comparison, including null or
unfavorable results. Do not start another comparison simply to obtain a positive
finding. Historical studies remain separate evidence.

## Registration and verification

Use `scripts/run_online_compact_study.py` with
`--root outputs/online_compact_tower_200_01`. Preparation freezes sources/assets,
generated geometry, runtime/client versions, development evidence hashes, seeds,
slots, budget, representation and metrics. `--phase run` reuses the bounded paired
scheduler; workers verify source consistency before and after each episode.

The final analysis includes resource/source, submitted-target, RGB-D query and
terminal-contact audits. An additional audit independently reconstructs the actual
compact prompts and compares all public feedback fields, then verifies delivery
in model transcripts. Base geometry exclusion is checked separately. Missing
transcripts from interrupted requests are reported as unverified delivery rather
than silently treated as verified. No-encoding Full turns remain valid when the
original unavailable/minimal geometry would not benefit from factoring.

This compares two representation packages at a common budget. Differences from
the previous 180-second study also involve new layouts, model variability and the
serializer; cross-study success-rate changes cannot isolate any one of those.
