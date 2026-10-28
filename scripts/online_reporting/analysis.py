"""Pure pairing, cohort selection, statistics, and success-time curves."""
import math
import statistics

import numpy as np

TASKS = ('tower', 'ring', 'relocation')
ARMS = ('Base', 'Full')
MEASURES = ('success', 'capped_seconds', 'total_tokens', 'all_model_seconds',
            'completed_requests', 'images', 'camera_acquisitions', 'action_seconds')
COHORTS = ('original_5', 'new_45', 'pooled_50', 'sensitivity_valid_pairs_new',
           'sensitivity_valid_pairs_pooled', 'sensitivity_without_archive_recoveries_new',
           'original_attempts_50')


def require(condition, message):
    if not condition:
        raise ValueError(message)

def equal(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8),
            f'{label}: {actual} != {expected}')

def paired(rows):
    """Validate paired identities and retain current and historical quality flags."""
    result = []
    for task, seed in sorted({(r['task'], r['seed']) for r in rows}):
        selected = [r for r in rows if (r['task'], r['seed']) == (task, seed)]
        require(len(selected) == 2 and {r['arm'] for r in selected} == set(ARMS), 'Incomplete pair')
        base, full = [next(r for r in selected if r['arm'] == arm) for arm in ARMS]
        require(base['initial_state_sha256'] == full['initial_state_sha256'], 'Initial states differ')
        require(base['cohort'] == full['cohort'], 'Cohorts differ within pair')
        result.append(dict(task=task, seed=seed, cohort=base['cohort'],
            base_success=base['success'], full_success=full['success'],
            base_capped_seconds=base['capped_seconds'], full_capped_seconds=full['capped_seconds'],
            full_minus_base_sr=full['success'] - base['success'],
            full_minus_base_seconds=full['capped_seconds'] - base['capped_seconds'],
            quality_eligible=all(r.get('timing_valid') is not False and
                                 r.get('gpu_dispatch_query_ok') is not False for r in selected),
            historical_quality_eligible=all(r.get('historical_timing_valid', r.get('timing_valid')) is not False and
                r.get('historical_gpu_dispatch_query_ok', r.get('gpu_dispatch_query_ok')) is not False for r in selected),
            replaced_pair=any(r.get('attempt_replaced', False) for r in selected),
            archive_recovered=any(r.get('archive_recovery_verified', False) for r in selected)))
    return result

def aggregate(rows):
    """Summarize all episodes, including timeouts and recorded partial token costs."""
    arms, differences = [], []
    for task in TASKS:
        for arm in ARMS:
            selected = [r for r in rows if r['task'] == task and r['arm'] == arm]
            require(bool(selected), 'Missing task/arm')
            record = dict(task=task, arm=arm, episodes=len(selected),
                successes=sum(r['success'] for r in selected),
                usage_complete=all(r['usage_complete'] for r in selected),
                episodes_with_unknown_tokens=sum(not r['usage_complete'] for r in selected),
                incomplete_requests=sum(r['incomplete_requests'] for r in selected),
                recovered_interrupted_requests=sum(r.get('recovered_interrupted_requests', 0) for r in selected),
                recovered_interrupted_tokens=sum(r.get('recovered_interrupted_tokens', 0) for r in selected),
                unknown_token_requests=sum(r.get('unknown_token_requests', r['incomplete_requests']) for r in selected),
                timing_flagged=sum(r.get('timing_valid') is False for r in selected),
                gpu_query_flagged=sum(r.get('gpu_dispatch_query_ok') is False for r in selected),
                archive_recovered=sum(r.get('archive_recovery_verified', False) for r in selected))
            record.update({'mean_' + k: statistics.mean(r[k] for r in selected) for k in MEASURES})
            arms.append(record)
        pairs = paired([r for r in rows if r['task'] == task])
        indices = np.random.default_rng(1050).integers(0, len(pairs), size=(10000, len(pairs)))
        for metric in ('full_minus_base_sr', 'full_minus_base_seconds'):
            values = np.array([p[metric] for p in pairs])
            low, high = np.quantile(values[indices].mean(axis=1), [.025, .975])
            differences.append(dict(task=task, pairs=len(pairs), metric=metric,
                mean=float(values.mean()), ci95_low=float(low), ci95_high=float(high)))
    return dict(arms=arms, paired_differences=differences)

def check_aggregate(actual, frozen):
    """Compare recomputed receipt-based summaries with the frozen registration."""
    for row, ref in zip(actual['arms'], frozen['arms'], strict=True):
        require((row['task'], row['arm']) == (ref['task'], ref['arm']), 'Aggregate order')
        for key, value in ref.items():
            if key not in ('task', 'arm'):
                equal(row[key], value, key)
    for row, ref in zip(actual['paired_differences'], frozen['paired_differences'], strict=True):
        require((row['task'], row['metric']) == (ref['task'], ref['metric']), 'Difference order')
        for key in ('pairs', 'mean'):
            equal(row[key], ref[key], key)
        equal(row['ci95_low'], ref['ci95'][0], 'Lower interval')
        equal(row['ci95_high'], ref['ci95'][1], 'Upper interval')

def substitute_pairs(originals, repeats, selected_keys):
    """Use every selected repeat, including regressions, and preserve pair identity."""
    key = lambda r: (r['task'], r['seed'], r['arm'])
    previous = {key(r): r for r in originals}
    replacement = {key(r): r for r in repeats}
    require(len(previous) == len(originals), 'Duplicate original slot')
    require(len(replacement) == len(repeats), 'Duplicate repeat slot')
    expected = {(task, seed, arm) for task, seed in selected_keys for arm in ARMS}
    require(set(replacement) == expected and expected <= set(previous), 'Repeat coverage differs from selected pairs')
    merged, changes = [], []
    for old in originals:
        new = dict(replacement.get(key(old), old))
        changed = key(old) in expected
        if changed:
            require(new['initial_state_sha256'] == old['initial_state_sha256'], 'Repeated initial state differs')
            require(new['timing_valid'] and new['gpu_dispatch_query_ok'], 'Repeat quality audit failed')
            require(new['budget_seconds'] == old['budget_seconds'] == 200, 'Repeated budget differs')
            new['cohort'] = old['cohort']
            changes.append(dict(task=old['task'], seed=old['seed'], arm=old['arm'],
                original_source=old['source'], original_source_sha256=old['source_sha256'],
                selected_source=new['source'], selected_source_sha256=new['source_sha256'],
                original_success=old['success'], selected_success=new['success'],
                original_capped_seconds=old['capped_seconds'], selected_capped_seconds=new['capped_seconds'],
                original_tokens=old['total_tokens'], selected_tokens=new['total_tokens'],
                original_usage_complete=old['usage_complete'], selected_usage_complete=new['usage_complete']))
        new.update(attempt_replaced=changed, selected_attempt='targeted_repeat' if changed else 'original',
            historical_timing_valid=old.get('timing_valid'),
            historical_gpu_dispatch_query_ok=old.get('gpu_dispatch_query_ok'),
            pair_in_quality_sensitivity=(old['task'], old['seed']) not in selected_keys)
        merged.append(new)
    paired(merged)
    return merged, changes

def curves_for(episodes):
    """Count only verified successes; a timeout stays a failure at the deadline."""
    curves = []
    for task in TASKS:
        for arm in ARMS:
            rows = [r for r in episodes if r['task'] == task and r['arm'] == arm]
            times = sorted({0., 200., *(r['success_seconds'] for r in rows if r['success'])})
            for time in times:
                n = sum(r['success'] and r['success_seconds'] <= time for r in rows)
                curves.append(dict(task=task, arm=arm, budget_seconds=time, successes=n,
                                   episodes=len(rows), sr_percent=100 * n / len(rows)))
    return curves


def cohort_rows(episodes, original_attempts=None):
    """Keep historical pair exclusions fixed when audited repeats are selected."""
    recovered_pairs = {(p['task'], p['seed']) for p in paired(episodes) if p['archive_recovered']}
    subsets = {
        'original_5': [r for r in episodes if r['cohort'] == 'original_5'],
        'new_45': [r for r in episodes if r['cohort'] == 'new_45'],
        'pooled_50': episodes,
        'sensitivity_valid_pairs_new': [r for r in episodes if r['cohort'] == 'new_45' and r['pair_in_quality_sensitivity']],
        'sensitivity_valid_pairs_pooled': [r for r in episodes if r['pair_in_quality_sensitivity']],
        'sensitivity_without_archive_recoveries_new': [r for r in episodes if r['cohort'] == 'new_45' and
            r['pair_in_quality_sensitivity'] and (r['task'], r['seed']) not in recovered_pairs],
    }
    if original_attempts is not None:
        subsets['original_attempts_50'] = original_attempts
    return subsets
