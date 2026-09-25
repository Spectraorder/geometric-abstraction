"""Offline evidence and selection for the repair curriculum. No simulator imports."""
import ast
import difflib
import json
from pathlib import Path

VERSION = 2
STAGES = ('identify', 'plan', 'grasp', 'repair')


def diagnostics(calls):
    """Summarize only actor-visible measurements, declarations and SDK replies."""
    plans = []
    reports = []
    failures = []
    cameras = []
    for call in calls:
        method, params = call['method'], call.get('params', {})
        if method == 'plan_tool':
            # Missing legacy responses stay unknown; a request is not a success.
            plans.append({'position': params.get('position'),
                          'result': call.get('observation'), 'error': call.get('error')})
        if method == 'report':
            reports.append(params)
        if call.get('error'):
            failures.append({'method': method, 'error': call['error']})
        if method == 'camera':
            cameras.append(params.get('view', 'side'))
    return {'motion_requests': sum(c['method'] in ('move_tool', 'move_joints') for c in calls),
            'gripper_close_requests': sum(c['method'] == 'gripper' and c['params'].get('angle', 1) < 0 for c in calls),
            'camera_calls': {v: cameras.count(v) for v in set(cameras)},
            'camera_alias_warning': 'camera() and camera(side) are identical; wrist provides RGB only',
            'ik_previews': plans[-8:], 'sensor_reports': reports[-8:], 'rpc_errors': failures[-5:]}


def case_summary(results):
    """Absolute final offset is never rewarded as progress from an initial offset."""
    summary = {}
    for case in ('centered', 'offset'):
        trials = [r for r in results if r.get('scene', {}).get('repair_case') == case]
        safe = []
        for r in trials:
            repair = r.get('repair', {})
            safe.append(bool(repair) and not repair.get('collapse_seen', True)
                        and repair.get('max_base_drift_m', 1) <= .003
                        and not r.get('policy_error'))
        stages = {stage: 0 for stage in STAGES}
        improvement = 0.
        for r, preserved in zip(trials, safe):
            stage_record = r.get('repair_progress', {})
            verified = stage_record.get('verified', {}) if stage_record.get('version') == VERSION else {}
            for stage in STAGES[:-1]:
                stages[stage] += int(preserved and verified.get(stage, False))
            stages['repair'] += int(r['success'])
            if preserved and case == 'offset' and verified.get('grasp', False):
                initial = sum(x*x for x in r['repair']['initial_offset_m'])**.5
                gain = initial-r['repair']['final_center_error_m']
                # Millimetre deadband avoids ranking numerical settling noise.
                improvement += max(0., gain-.001)
        summary[case] = {'passed': sum(r['success'] for r in trials), 'total': len(trials),
                         'preserved': sum(safe), 'verified': stages,
                         'mean_verified_improvement_m': improvement/max(1,len(trials))}
    return summary


def annotate(entry):
    return {**entry, 'case_metrics': case_summary(entry['results'])}


def eligible(entry):
    cases = entry['case_metrics']
    return (cases['centered']['total'] > 0 and cases['offset']['total'] > 0
            and cases['centered']['passed'] == cases['centered']['total']
            and cases['offset']['preserved'] == cases['offset']['total'])


def progress_key(entry):
    case = entry['case_metrics']['offset']
    v = case['verified']
    return (case['passed'], v['grasp'], v['plan'], v['identify'],
            round(case['mean_verified_improvement_m'], 4))


def select_candidates(history):
    entries = [annotate(e) for e in history]
    safe = [e for e in entries if eligible(e)]
    # Keep the original as an explicit fallback if every candidate regresses.
    candidates = safe or entries[:1]
    champion = max(candidates, key=lambda e: (e['case_metrics']['offset']['passed'],
                                              e['case_metrics']['centered']['passed']))
    # A verified intermediate stage can outrank a no-op for the next proposal,
    # without relabelling it as a successfully learned repair skill.
    parent = max(enumerate(candidates), key=lambda pair: (progress_key(pair[1]), pair[0]))[1]
    archive = {}
    for stage in STAGES:
        measured = [e for e in candidates if e['case_metrics']['offset']['verified'][stage] > 0]
        if measured:
            winner = max(measured, key=lambda e: (e['case_metrics']['offset']['verified'][stage], progress_key(e)))
            archive[stage] = {'candidate': winner['candidate'], 'sha256': winner['sha256'],
                              'passed': winner['case_metrics']['offset']['verified'][stage],
                              'total': winner['case_metrics']['offset']['total']}
    focus = next((s for s in STAGES
                  if parent['case_metrics']['offset']['verified'][s] < parent['case_metrics']['offset']['total']), 'repair')
    return champion, parent, archive, focus


def source_for_prompt(source):
    """Omit unreferenced top-level legacy functions, but never change saved code."""
    tree = ast.parse(source)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    lines = source.splitlines(keepends=True)
    for node in reversed(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name != 'run' and node.name not in names:
            start = min([node.lineno]+[d.lineno for d in node.decorator_list])-1
            lines[start:node.end_lineno] = [f'# Unreferenced legacy function {node.name} omitted from display only.\n']
    return ''.join(lines)


def build_context(parent, history, focus, contract, limit=85000):
    """Keep the actual parent plus recent executable diffs, not just hypotheses."""
    feedback = []
    seen = set()
    for entry in (parent, history[-1]):
        if entry['candidate'] in seen:
            continue
        seen.add(entry['candidate'])
        for result in entry['results']:
            folder = Path(result['directory'])
            item = json.loads((folder/'actor_feedback.json').read_text())
            notes = item.pop('notes', [])
            item['notes'] = [str(x)[:500] for x in notes[:2]+notes[-5:]]
            if 'diagnostics' not in item and (folder/'api_calls.json').exists():
                item['diagnostics'] = diagnostics(json.loads((folder/'api_calls.json').read_text()))
            feedback.append({'policy': Path(entry['candidate']).parent.name,
                             'case': result['scene']['repair_case'], **item})
    by_hash = {e['sha256']: e for e in history}
    memory = []
    for entry in history[-4:]:
        folder = Path(entry['candidate']).parent
        if not (folder/'response.json').exists():
            continue
        response = json.loads((folder/'response.json').read_text())
        invocation = json.loads((folder/'invocation.json').read_text())
        ancestor = by_hash.get(invocation.get('parent_sha256'))
        diff = ''
        if ancestor and entry['candidate'] != parent['candidate']:
            diff = ''.join(difflib.unified_diff(
                source_for_prompt(Path(ancestor['candidate']).read_text()).splitlines(True),
                source_for_prompt(Path(entry['candidate']).read_text()).splitlines(True),
                fromfile='prior_parent', tofile='prior_candidate', n=2))
        memory.append({'candidate': folder.name, 'hypothesis': response['hypothesis'][:800],
                       'case_metrics': case_summary(entry['results']),
                       'actual_code_diff': diff[:4000], 'diff_truncated': len(diff)>4000})
    source = source_for_prompt(Path(parent['candidate']).read_text())
    intro = ('\nCURRENT FOCUS: '+focus+'\nThe full repair pass criterion is unchanged. '
             'First resolve this earliest missing milestone. Centered control success alone '
             'does not demonstrate identification or repair. Log each rejection numerically.\n')
    def assemble():
        return contract+intro+'\nCURRENT SOURCE:\n'+source+'\nPOST-TRIAL FEEDBACK:\n'+json.dumps(feedback)+'\nPRIOR CODE REPAIRS:\n'+json.dumps(memory)
    prompt = assemble()
    while len(prompt)>limit and len(memory)>1:
        memory.pop(0)
        prompt = assemble()
    if len(prompt)>limit:
        raise RuntimeError('Context exceeds the campaign cap; no model call made')
    return prompt
