"""Training-only retention and bounded, event-centered actor evidence."""
import json

RULE = 'preserve_successes_strict_improvement'


def retention(curve):
    """Accept strict success-set supersets; ties and regressions retain the parent."""
    if not curve or [r['revision'] for r in curve] != list(range(len(curve))):
        raise ValueError('Require consecutive training revisions starting at zero')
    incumbent = set()
    seeds = None
    best = 0
    decisions = []
    for row in curve:
        results = row['results']
        current_seeds = [r['seed'] for r in results]
        if len(set(current_seeds)) != len(current_seeds) or not current_seeds:
            raise ValueError('Require distinct nonempty training seeds')
        if seeds is not None and current_seeds != seeds:
            raise ValueError('Training seeds changed across revisions')
        seeds = current_seeds
        successes = {r['seed'] for r in results if r['success']}
        gained, lost = sorted(successes-incumbent), sorted(incumbent-successes)
        accepted = row['revision'] == 0 or (bool(gained) and not lost)
        parent = best
        if accepted:
            best, incumbent = row['revision'], successes
        decisions.append({'revision': row['revision'], 'parent_revision': parent,
                          'accepted': accepted, 'selected_revision': best,
                          'repaired_seeds': gained, 'regressed_seeds': lost,
                          'protected_seeds': sorted(incumbent)})
    return decisions


def select_revision(curve, manifest):
    if manifest.get('retention_rule') == RULE:
        return retention(curve)[-1]['selected_revision']
    return max(curve, key=lambda row: row['successes'])['revision']


def stage(root, arm, revision):
    return root / arm / ('revision_000' if revision == 0 else f'round_{revision:03d}')


def training_curve(root, arm, revision, seeds):
    """Read only completed, own-arm training feedback before the proposal budget."""
    curve = []
    for index in range(revision):
        results = []
        for seed in seeds:
            path = stage(root, arm, index) / 'trials' / f'seed_{seed}' / 'actor_feedback.json'
            value = json.loads(path.read_text())
            results.append({'seed': seed, 'success': bool(value['success'])})
        curve.append({'revision': index, 'results': results,
                      'successes': sum(r['success'] for r in results)})
    return curve


def failure_kind(feedback):
    if feedback['success']:
        return None
    text = str(feedback.get('policy_error') or '') + ' ' + ' '.join(feedback.get('notes', []))
    if 'too far from the expected tool neighborhood' in text:
        return 'object_tool_mismatch'
    if 'No usable current view' in text or 'No reliable visible' in text:
        return 'insufficient_current_view'
    if feedback.get('policy_error'):
        return 'policy_exception'
    return 'unexplained_failure' if not feedback['success'] else None


def event_records(records):
    """Keep the first explicit failure with same-object evidence before/after it."""
    if not records:
        return []
    failure = next((i for i, r in enumerate(records)
                    if 'exception' in r.get('step', '') or
                    r.get('action_status') in ('exception', 'placement_unconfirmed', 'grasp_unconfirmed')), None)
    index = len(records)-1 if failure is None else failure
    obj = records[index].get('object')
    selected = {index}
    before = [i for i in range(index) if records[i].get('object') == obj]
    if before:
        selected.add(before[-1])
        valid = [i for i in before if records[i].get('status') == 'current_estimate']
        if valid:
            selected.add(valid[-1])
    after = next((i for i in range(index+1, len(records)) if records[i].get('object') == obj), None)
    if after is not None:
        selected.add(after)
    return [records[i] for i in sorted(selected)]


def behavior_feedback(root, arm, revision, seeds):
    """Compare issued commands to the proposal's parent, not private physics."""
    if revision < 1:
        return {}
    parent = retention(training_curve(root, arm, revision, seeds))[-1]['selected_revision']
    output = {}
    for seed in seeds:
        sequences = []
        for index in (parent, revision):
            path = stage(root, arm, index)/'trials'/f'seed_{seed}'/'api_calls.json'
            if not path.exists():
                sequences.append(None)
                continue
            calls = json.loads(path.read_text())
            sequences.append([{'method': c['method'], 'params': c['params']} for c in calls
                              if c['method'] in ('move_tool', 'move_joints', 'gripper', 'wait')])
        before, after = sequences
        if before is None or after is None:
            output[seed] = {'parent_revision': parent, 'status': 'commands_unavailable'}
            continue
        changed = before != after
        first = next((i for i, pair in enumerate(zip(before, after)) if pair[0] != pair[1]),
                     min(len(before), len(after)) if changed else None)
        output[seed] = {'parent_revision': parent, 'status': 'recorded',
                        'motion_commands_identical': not changed,
                        'parent_command_count': len(before), 'candidate_command_count': len(after),
                        'first_changed_command_index': first,
                        'scope': 'Issued commands, not measured trajectories or success certificates'}
    return output


def policy_diagnostics(raw, limit=4, record_budget=1600):
    """Keep bounded own-policy reports, explicitly distinct from supplied fits."""
    records = [r for r in raw if isinstance(r, dict) and r.get('kind') != 'geometry_step']
    # Give explicit hypothesis probes priority while retaining recent ad hoc logs.
    probes = [r for r in records if r.get('kind') == 'policy_probe']
    other = [r for r in records if r.get('kind') != 'policy_probe']
    selected = (other[-2:] + probes[-2:]) if probes else other[-limit:]
    output = []
    for record in selected[-limit:]:
        encoded = json.dumps(record, separators=(',', ':'))
        data = record if len(encoded) <= record_budget else {
            'truncated_json_preview': encoded[:record_budget], 'original_characters': len(encoded)}
        output.append({'provenance': 'own_policy_report_unverified', 'data': data})
    return output, len(records)-len(output)


def event_feedback(root, arm, revision, seeds, budget=16000):
    rows = []
    all_records = {}
    behavior = behavior_feedback(root, arm, revision-1, seeds)
    for seed in seeds:
        trial = stage(root, arm, revision-1) / 'trials' / f'seed_{seed}'
        feedback = json.loads((trial/'actor_feedback.json').read_text())
        path = trial/'geometry_records.json'
        raw = json.loads(path.read_text()) if path.exists() else []
        records = [r['record'] for r in raw if r.get('kind') == 'geometry_step']
        diagnostics, omitted = policy_diagnostics(raw)
        all_records[seed] = records
        rows.append({'seed': seed, 'revision': revision-1, 'success': feedback['success'],
                     'failure_kind': failure_kind(feedback),
                     'status': 'recorded' if records else 'no_target_checkpoint',
                     'checkpoints': event_records(records),
                     'policy_diagnostics': diagnostics, 'omitted_policy_diagnostics': omitted,
                     'behavior_comparison': behavior.get(seed, {'status': 'starter_has_no_parent'})})
    # Public task color permutations identify the same role across layouts.
    manifest_path = root/'manifest.json'
    color_maps = json.loads(manifest_path.read_text()).get('color_maps', {}) if manifest_path.exists() else {}
    for row in rows:
        if row['success'] or not row['checkpoints']:
            continue
        obj = row['checkpoints'][-1].get('object')
        canonical = next((key for key, value in color_maps.get(str(row['seed']), {}).items() if value == obj), None)
        if canonical is None:
            continue
        for other in rows:
            other_object = color_maps.get(str(other['seed']), {}).get(canonical)
            matching = [r for r in all_records[other['seed']] if r.get('object') == other_object]
            if other['success'] and matching:
                row['successful_reference'] = {'seed': other['seed'], 'historical': True,
                                               'canonical_task_color': canonical,
                                               'checkpoints': event_records(matching)}
                break
    size = lambda: len(json.dumps(rows, separators=(',', ':')))
    if size() > budget:
        for row in rows:
            row.pop('successful_reference', None)
    if size() > budget:
        for row in rows:
            dropped = max(0, len(row['policy_diagnostics'])-1)
            row['omitted_policy_diagnostics'] += dropped
            row['policy_diagnostics'] = row['policy_diagnostics'][-1:]
    if size() > budget:
        for row in rows:
            row['checkpoints'] = row['checkpoints'][-2:]
    if size() > budget:
        for row in rows:
            row['checkpoints'] = row['checkpoints'][-1:]
    if size() > budget:
        raise RuntimeError('Event feedback for every training layout exceeds the budget')
    return rows


def failure_images(trial):
    """Select at most two existing RGB frames nearest the first failure event.

    This host-side selector does not infer object geometry or inspect reserved data.
    Calibration and timestamps accompany each side/wrist frame. No new model call.
    """
    import numpy as np
    calls = json.loads((trial/'api_calls.json').read_text()) if (trial/'api_calls.json').exists() else []
    cameras = [c for c in calls if c['method'] == 'camera']
    files = sorted(trial.glob('camera_*.npz'))
    if len(cameras) != len(files):
        raise ValueError('Camera metadata mismatch')
    failures = [c['time'] for c in calls if c['method'] == 'failed']
    path = trial/'geometry_records.json'
    if path.exists():
        failures += [r['record']['time'] for r in json.loads(path.read_text())
                     if r.get('kind') == 'geometry_step' and
                     ('exception' in r['record'].get('step', '') or r['record'].get('action_status') == 'exception')]
    time = min(failures) if failures else (cameras[-1]['time'] if cameras else 0)
    images, metadata = [], []
    for view in ('side', 'wrist'):
        eligible = [(p, c) for p, c in zip(files, cameras) if c['params'].get('view', 'side') == view and c['time'] <= time+1e-6]
        if not eligible:
            continue
        path, call = min(eligible, key=lambda pair: abs(pair[1]['time']-time))
        with np.load(path) as frame:
            metadata.append({'view': view, 'time': float(frame['time']), 'failure_time': time,
                             'intrinsics': frame['intrinsics'].tolist(),
                             'camera_to_world': frame['camera_to_world'].tolist()})
        images.append(path.with_suffix('.png'))
    return images, metadata
