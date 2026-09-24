# Dual-system branch verification

Verified on 2026-09-24 on macOS with Python 3.14, the project's installed MuJoCo/mjlab environment, and the Codex in-app browser. Base: `3a2a48a`; branch: `codex/dual-system-benchmark-english`.

## Automated checks

- Main suite: **87 tests, 85 passed, 2 skipped**. Both skips require Linux bubblewrap; the macOS isolation checks ran.
- Recovery contract suite: **14 passed**.
- Empty benchmark evidence correctly reports `incomplete_or_below_target`; it never inherits the historical pilot success rates.
- All **21** bundled MP4 files decoded without FFmpeg errors. All 15 historical and 6 new video SHA-256 values match their manifests.
- Six real model/physics integration attempts are recorded, across two protocols. Protocol 2 hybrid and all-Astra completed the tower on development seed 4001; Jev-only stopped before motion. These are not independent heldout success-rate estimates.
- Failed provider calls retain attempted-call timing and unknown usage/cost rather than becoming free, invisible calls. No automatic paid retry or mock fallback is used.

Commands:

```bash
uv run python -m unittest discover -s tests
uv run python -m unittest discover -s research/recovery_lab_20260924/harness -p 'test_*.py'
uv run python scripts/benchmark_policy.py --profile all_supported --output outputs/readiness_new.json
```

## Browser checks

Served `web/recovery` with the repository's range-capable static server at `http://127.0.0.1:8093/`.

| Check | Observed result |
|---|---|
| App boot | Correct title and populated English interface; no error overlay |
| Console | No captured warning/error entries |
| Routing | Buttons, direct hash navigation and reload select the intended panel |
| Evolution | All 15 attempts; selection changes parameter detail, outcome and video |
| Tasks | All 13 catalog entries carry explicit evidence status |
| Design draft | Bridge spacing changes the diagram and overlap estimate; export confirms no execution started |
| Benchmark | All 10 cells and all six actual model trials are displayed |
| Model replay | Hybrid Protocol 2 video plays; duration 49.25 seconds, no media error |
| Other new replay files | All six media metadata entries load with valid durations |
| Mobile | 390 × 844 viewport; document width 390; no horizontal overflow |
| Language | Page text is English; archival raw evidence and original research sources retain original language |

Evidence: [desktop task catalog](media/dual_lab_desktop.png), [mobile evolution](media/dual_lab_mobile.png), [model results](evaluations/dual_system_live.json).

## Interpretation and remaining limits

This is an initial-observation model gate followed by one shared frozen visual skill. It does not yet call Jev during a grasp, placement, or tower repair. The successful hybrid run did not require an online Astra repair. The change between protocols was an offline harness repair, not autonomous online learning.

The confidence threshold is uncalibrated. Astra dollar cost is unknown because invocation used the Codex subscription client. The local timings include client overhead, and some background rendering/verification overlapped. No speed, accuracy or cost advantage is established by this one-scene check.

Benchmark release criteria are implemented, but the complete fresh heldout campaign has not been run. Attestations in normalized evaluator records are explicit audit inputs, not a cryptographic isolation proof. Real-robot validation, physical occluders, bridge execution, and selective dual-tower repair remain future work.

The English site is running locally and is committed as a standalone static bundle. This branch does not update the existing public deployment. Credentials and local deployment metadata are excluded from Git.
