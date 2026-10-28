#!/usr/bin/env python3
"""Verify and export the completed 50-pair study without running experiments."""
import argparse
import csv
import json
from pathlib import Path

from online_reporting.analysis import COHORTS, curves_for
from online_reporting.evidence import EvidenceCollector, ROOT, SOURCE, digest

DEFAULT_OUTPUT = ROOT / 'docs/evaluations/online_50'


def write_csv(output, name, rows):
    with (output / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for r in rows for k in r)),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=
        'Requires the original output archives. No simulation or model requests are made.')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true',
                       help='verify evidence and statistics without writing exports')
    modes.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                       help='export directory (default: docs/evaluations/online_50)')
    args = parser.parse_args(argv)
    collector = EvidenceCollector()
    try:
        frozen, episodes, requests, pairs, aggregates, epochs, usage_audit, integration = collector.collect()
    except FileNotFoundError as error:
        parser.error(f'Required evidence is missing: {error.filename}. Restore the original output archives.')
    summary = dict(status='completed_with_audited_paired_replacements', episodes=len(episodes), paired_layouts=len(pairs),
        completed_requests=sum(r['completed_requests'] for r in episodes),
        incomplete_requests=sum(r['incomplete_requests'] for r in episodes),
        unknown_token_requests=usage_audit['unknown_requests'],
        recovered_interrupted_requests=usage_audit['recovered_requests'],
        recovered_interrupted_tokens=usage_audit['recovered_total_tokens'],
        completed_receipt_total_tokens=sum(r['receipt_total_tokens'] for r in episodes),
        total_recorded_tokens=sum(r['total_tokens'] for r in episodes),
        quality=dict(timing_flagged=sum(r.get('timing_valid') is False for r in episodes),
            gpu_dispatch_query_failed=sum(r.get('gpu_dispatch_query_ok') is False for r in episodes),
            historical_timing_flagged=frozen['quality']['timing_flagged'],
            historical_gpu_dispatch_query_failed=frozen['quality']['gpu_dispatch_query_failed'],
            replaced_pairs=9, replaced_episodes=18, archive_recovered_episodes=2),
        cohorts=aggregates, runtime_epochs=epochs,
        runtime_change=frozen['runtime_change'], archive_recovery=frozen['archive_recovery'],
        infrastructure_overhead=frozen['infrastructure_overhead'],
        replacement={k: v for k, v in integration.items() if k not in ('changes', 'original_attempts', 'superseded_attempts')},
        campaign_recorded_tokens_including_superseded_and_startup=sum(r['total_tokens'] for r in episodes) +
            integration['superseded_recorded_tokens'] + frozen['infrastructure_overhead']['total_recorded_tokens'],
        interval_method='Paired-layout percentile bootstrap, 10000 resamples, seed 1050 per task; no multiplicity correction.',
        scope='300 selected episodes: 282 original and 18 audited repeats replacing nine whole pairs. '
            'Reporting amendment made after inspecting repeats; every selected repeat is used, including regressions. '
            'Original outcomes and superseded costs remain separate. No component ablations.')
    if args.check:
        print(json.dumps({k: summary[k] for k in ('status', 'episodes', 'paired_layouts',
            'completed_requests', 'incomplete_requests', 'total_recorded_tokens')}, indent=2))
        print('Verified evidence and statistics; no files written.')
        return
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    curves = curves_for(episodes)
    write_csv(output, 'episodes.csv', episodes)
    write_csv(output, 'model_requests.csv', requests)
    write_csv(output, 'paired_outcomes.csv', pairs)
    write_csv(output, 'success_vs_time.csv', curves)
    write_csv(output, 'arm_summary.csv', aggregates['pooled_50']['arms'])
    write_csv(output, 'cohort_summary.csv', [dict(cohort=c, **r) for c in COHORTS for r in aggregates[c]['arms']])
    write_csv(output, 'paired_differences.csv', [dict(cohort=c, **r) for c in COHORTS for r in aggregates[c]['paired_differences']])
    write_csv(output, 'runtime_epochs.csv', [dict(runtime_epoch=e, **r) for e in epochs for r in epochs[e]['arms']])
    write_csv(output, 'quality_flags.csv', [r for r in episodes if not r['pair_in_quality_sensitivity'] or r.get('archive_recovery_verified')])
    write_csv(output, 'replacements.csv', integration['changes'])
    write_csv(output, 'original_attempts.csv', integration['original_attempts'])
    write_csv(output, 'superseded_attempts.csv', integration['superseded_attempts'])
    (output / 'replacement_manifest.json').write_text(json.dumps({k: v for k, v in integration.items()
        if k not in ('original_attempts', 'superseded_attempts')}, indent=2) + '\n')
    (output / 'interrupted_usage_audit.json').write_text(json.dumps(usage_audit, indent=2) + '\n')
    write_csv(output, 'recovered_interrupted_usage.csv', [dict(
        **{k: v for k, v in r.items() if k != 'usage'}, **r['usage'])
        for r in usage_audit['records'] if r['recovered']])
    for name in ('outcome_bars.csv', 'model_resources.csv'):
        write_csv(output, name, aggregates['pooled_50']['arms'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    from online_reporting.plotting import plot
    plot(episodes, aggregates, pairs, curves, output)
    provenance = dict(exporter=str(Path(__file__).relative_to(ROOT)), exporter_sha256=digest(Path(__file__)),
        usage_auditor_sha256=digest(ROOT / 'scripts/online_interrupted_usage.py'),
        style_source_sha256=digest(ROOT / 'scripts/export_online_results.py'), source_files=collector.sources,
        report_module_sha256={str(p.relative_to(ROOT)): digest(p)
                              for p in sorted((ROOT / 'scripts/online_reporting').glob('*.py'))},
        source_summary_sha256=digest(SOURCE / 'summary.json'),
        exported_files={p.name: digest(p) for p in sorted(output.iterdir())
                        if p.suffix in {'.csv', '.json', '.pdf', '.png', '.svg'} and p.name != 'provenance.json'})
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: summary[k] for k in ('status', 'episodes', 'paired_layouts', 'completed_requests',
                                            'incomplete_requests', 'total_recorded_tokens')}, indent=2))
    print(f'Exported to {output}')


if __name__ == '__main__':
    main()
