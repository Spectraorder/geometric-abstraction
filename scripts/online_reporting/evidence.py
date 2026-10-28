"""Verify archived evidence and construct the selected 300-episode report.

Sources are read-only. Each collector owns its provenance ledger so separate
exports cannot inherit hashes from an earlier collection.
"""
import hashlib
import json
from pathlib import Path

from online_interrupted_usage import audit_interrupted_requests, apply_recoveries
from .analysis import (aggregate, check_aggregate, cohort_rows, equal, paired,
                       require, substitute_pairs)

ROOT = Path(__file__).absolute().parents[2]
SOURCE = ROOT / 'outputs/online_extension_50_01/terminal_resume_01'
REPEAT = ROOT / 'outputs/online_targeted_repeat_01'
INTEGRATION = REPEAT / 'report_integration_01'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path):
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    for archive in (SOURCE.parent, REPEAT):
        if path.is_relative_to(archive.resolve()):
            return str(archive.relative_to(ROOT) / path.relative_to(archive.resolve()))
    return str(path.relative_to(ROOT))


class EvidenceCollector:
    """Load and audit the frozen originals, usage recoveries, and paired repeats."""

    def __init__(self):
        self.sources = {}

    def read(self, path, expected=None):
        path = ROOT / path
        actual = digest(path)
        require(expected is None or actual == expected, f'Changed evidence: {path}')
        self.sources[portable(path)] = actual
        return json.loads(path.read_text())

    def collect_records(self, records):
        episodes, requests = [], []
        for original in records:
            row = dict(original)
            result = self.read(row['source'], row['source_sha256'])
            equal(result['success'], row['success'], 'Success')
            equal(result['capped_completion_seconds'], row['capped_seconds'], 'Capped time')
            require(row['budget_seconds'] == 200 and result['budget_s'] == 200, 'Changed budget')
            require(row['status'] in ('success', 'timeout'), 'Unexpected status')
            require((row['success_seconds'] is not None) == bool(row['success']), 'Missing success time')
            equal(row['capped_seconds'], row['success_seconds'] if row['success'] else 200, 'Timeout handling')
            require(0 <= row['capped_seconds'] <= 200, 'Time outside budget')
            equal(row['input_tokens'] + row['output_tokens'], row['total_tokens'], 'Token arithmetic')
            directory = (ROOT / row['source']).parent
            current = []
            common = {k: row[k] for k in ('task', 'cohort', 'seed', 'arm')}
            for path in sorted(directory.glob('context_*/decision_*/receipt.json')):
                receipt = self.read(path)
                equal(receipt['usage']['input_tokens'] + receipt['usage']['output_tokens'],
                      receipt['usage']['total_tokens'], 'Receipt tokens')
                current.append(dict(**common, context=path.parent.parent.name, decision=path.parent.name,
                    status='completed', usage_complete=True, wall_seconds=receipt['wall_seconds'],
                    image_count=receipt['image_count'], **receipt['usage'], source=portable(path)))
            for path in sorted(directory.glob('context_*/decision_*/attempt_failure.json')):
                failure, invocation = self.read(path), self.read(path.parent / 'invocation.json')
                current.append(dict(**common, context=path.parent.parent.name, decision=path.parent.name,
                    status=failure['type'], usage_complete=False, wall_seconds=failure['wall_seconds'],
                    image_count=len(invocation['images']), input_tokens=None, cached_input_tokens=None,
                    output_tokens=None, total_tokens=None, source=portable(path)))
            completed = [r for r in current if r['usage_complete']]
            equal(len(completed), row['completed_requests'], 'Completed requests')
            equal(len(current) - len(completed), row['incomplete_requests'], 'Interrupted requests')
            require(row['usage_complete'] == (len(current) == len(completed)), 'Usage completeness')
            for key in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'total_tokens'):
                equal(sum(r[key] for r in completed), row[key], key)
            equal(sum(r['wall_seconds'] for r in current), row['all_model_seconds'], 'Model wait total')
            equal(sum(r['image_count'] for r in current), row['images'], 'Image attachments')
            row['source'] = portable(row['source'])
            for key in ('audit_source',):
                if row.get(key):
                    row[key] = portable(row[key])
            row['completed_images'] = sum(r['image_count'] for r in completed)
            row['incomplete_images'] = row['images'] - row['completed_images']
            episodes.append(row)
            requests.extend(current)
        return episodes, requests

    def collect_original(self):
        """Audit original receipts before adding verified interrupted usage."""
        audit = self.read(SOURCE / 'completion_audit.json')
        require(audit['status'] == 'verified_complete_with_preserved_quality_flags', 'Incomplete audit')
        summary = self.read(SOURCE / 'summary.json', audit['summary_sha256'])
        require(digest(SOURCE / 'episodes.csv') == audit['csv_sha256'], 'Changed source CSV')
        self.sources[portable(SOURCE / 'episodes.csv')] = audit['csv_sha256']
        episodes, requests = self.collect_records(summary['records'])
        require(len(episodes) == 300, 'Expected 300 episodes')
        pairs = paired(episodes)
        require(len(pairs) == 150, 'Expected 150 pairs')
        eligible = {(p['task'], p['seed']) for p in pairs if p['quality_eligible']}
        for row in episodes:
            row['pair_in_quality_sensitivity'] = (row['task'], row['seed']) in eligible
        subsets = cohort_rows(episodes)
        aggregates = {}
        for cohort, rows in subsets.items():
            aggregates[cohort] = aggregate(rows)
            check_aggregate(aggregates[cohort], summary[cohort])
        epochs = {}
        for epoch, ref in summary['by_runtime_epoch'].items():
            epochs[epoch] = aggregate([r for r in episodes if r.get('runtime_epoch') == epoch])
            check_aggregate(epochs[epoch], ref)
        require(sum(r.get('timing_valid') is False for r in episodes) == 4, 'Timing flags')
        require(sum(r.get('gpu_dispatch_query_ok') is False for r in episodes) == 16, 'GPU flags')
        require(sum(not r['pair_in_quality_sensitivity'] for r in episodes) == 18, 'Quality pair union')
        require(sum(r.get('archive_recovery_verified', False) for r in episodes) == 2, 'Archive labels')
        # Validate the frozen receipt-based totals first, then add only independently
        # verified terminal usage. The original results and decision counts stay fixed.
        usage_audit = audit_interrupted_requests(requests)
        self.sources.update(usage_audit['source_files'])
        apply_recoveries(episodes, requests, usage_audit)
        aggregates = {cohort: aggregate(rows) for cohort, rows in subsets.items()}
        epochs = {epoch: aggregate([r for r in episodes if r.get('runtime_epoch') == epoch])
                  for epoch in summary['by_runtime_epoch']}
        return summary, episodes, requests, pairs, aggregates, epochs, usage_audit

    def collect(self):
        """Apply every registered repeat to its pair and preserve original costs."""
        frozen, original, original_requests, _, original_aggregates, _, original_usage = self.collect_original()
        registration = self.read(INTEGRATION / 'registration.json')
        require(digest(INTEGRATION / 'registration.json') ==
                (INTEGRATION / 'registration.sha256').read_text().strip(), 'Integration registration changed')
        for name, sha in registration['source_sha256'].items():
            path = ROOT / name
            require(digest(path) == sha, 'Integration evidence changed: ' + name)
            self.sources[portable(path)] = sha
        repeated = self.read(REPEAT / 'summary.json')
        require(repeated['status'] == 'completed' and repeated['episodes'] == 18, 'Incomplete repeat')
        require(self.read(REPEAT / 'progress.json')['audited'] == 18, 'Repeat audits incomplete')
        for path in sorted((REPEAT / 'audits/final').rglob('*.json')):
            audit = self.read(path)
            if isinstance(audit, dict):
                require(audit['status'] == 'passed', 'Failed final repeat audit')
        original_by_key = {(r['task'], r['seed'], r['arm']): r for r in original}
        raw_repeats = []
        for row in repeated['records']:
            old = original_by_key[(row['task'], row['seed'], row['arm'])]
            require(row['original_source_sha256'] == old['source_sha256'], 'Repeat refers to different original')
            raw_repeats.append(dict(row, cohort=old['cohort']))
        repeats, repeat_requests = self.collect_records(raw_repeats)
        repeat_usage = audit_interrupted_requests(repeat_requests)
        self.sources.update(repeat_usage['source_files'])
        apply_recoveries(repeats, repeat_requests, repeat_usage)
        selected_keys = {(r['task'], r['seed']) for r in original if not r['pair_in_quality_sensitivity']}
        require(len(selected_keys) == 9, 'Expected nine historical quality pairs')
        registered = self.read(REPEAT / 'registration.json')['slots']
        require({(s['task'], s['seed']) for s in registered} == selected_keys, 'Repeat selection changed')
        episodes, changes = substitute_pairs(original, repeats, selected_keys)
        requests = [r for r in original_requests if (r['task'], r['seed']) not in selected_keys] + repeat_requests
        usage_records = [r for r in original_usage['records'] if (r['task'], r['seed']) not in selected_keys] + repeat_usage['records']
        recovered = [r for r in usage_records if r['recovered']]
        usage_audit = dict(status='verified', interrupted_requests=len(usage_records),
            recovered_requests=len(recovered), unknown_requests=len(usage_records) - len(recovered),
            recovered_total_tokens=sum(r['usage']['total_tokens'] for r in recovered), records=usage_records,
            source_files={**original_usage['source_files'], **repeat_usage['source_files']},
            reason_counts={kind: sum(r['interruption_type'] == kind for r in usage_records)
                           for kind in sorted({r['interruption_type'] for r in usage_records})},
            interpretation='Selected 300-episode accounting; superseded attempts are reported separately. Unknown usage is not zero.')
        pairs = paired(episodes)
        require(len(episodes) == 300 and len(pairs) == 150, 'Replacement changed sample size')
        subsets = cohort_rows(episodes, original)
        aggregates = {cohort: aggregate(rows) for cohort, rows in subsets.items()}
        for name in ('original_5', 'sensitivity_valid_pairs_new', 'sensitivity_valid_pairs_pooled',
                     'sensitivity_without_archive_recoveries_new'):
            require(aggregates[name] == original_aggregates[name], 'Unaffected cohort changed: ' + name)
        epochs = {epoch: aggregate([r for r in episodes if r.get('runtime_epoch') == epoch])
                  for epoch in sorted({r['runtime_epoch'] for r in episodes if r.get('runtime_epoch')})}
        superseded = [r for r in original if (r['task'], r['seed']) in selected_keys]
        integration = dict(registration=portable(INTEGRATION / 'registration.json'),
            replaced_pairs=9, replaced_episodes=18, changes=changes,
            original_attempts=original, superseded_attempts=superseded,
            superseded_recorded_tokens=sum(r['total_tokens'] for r in superseded),
            superseded_unknown_requests=sum(r['unknown_token_requests'] for r in superseded),
            repeat_recorded_tokens=sum(r['total_tokens'] for r in repeats),
            repeat_unknown_requests=repeat_usage['unknown_requests'],
            interpretation=registration['selection'], timing=registration['timing'])
        return frozen, episodes, requests, pairs, aggregates, epochs, usage_audit, integration
