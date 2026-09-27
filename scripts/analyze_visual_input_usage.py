"""Describe submitted training policies and compact records; never actor feedback."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def analyze(root):
    policies = []
    arms = {arm: {'records': [], 'inputs': [], 'outputs': [], 'calls': Counter()}
            for arm in ('basic', 'symbolic')}
    methods = ('camera', 'observe', 'get', 'reference', 'relation', 'align',
               'target', 'goal_error', 'checkpoint')
    fields = ('axis_dimensions', 'rotation', 'vertices', 'position_fully_observed',
              'placement_goal', 'remaining_translation_m')
    for source in sorted(root.glob('replica_*/*/round_*/candidate.py')):
        generation = source.parent/'generation.json'
        if not generation.is_file():
            continue
        code = source.read_text()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        meta = json.loads(generation.read_text())
        if meta['candidate_sha256'] != digest:
            raise ValueError('Submitted candidate changed: '+str(source))
        tree = ast.parse(code)
        calls = Counter(n.func.attr for n in ast.walk(tree)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute))
        literals = Counter(n.value for n in ast.walk(tree)
                           if isinstance(n, ast.Constant) and isinstance(n.value, str))
        arm = source.parent.parent.name
        row = {'path': str(source.parent.relative_to(root)), 'candidate_sha256': digest,
               'source_characters': len(code),
               'static_attribute_call_sites': {name: calls[name] for name in methods},
               'literal_key_mentions': {name: literals[name] for name in fields},
               'usage': meta.get('usage', {})}
        policies.append(row)
        arms[arm]['calls'].update(row['static_attribute_call_sites'])
        for key, dest in (('input_tokens', 'inputs'), ('output_tokens', 'outputs')):
            value = row['usage'].get(key)
            if value is not None:
                arms[arm][dest].append(value)
    for path in sorted(root.glob('replica_*/*/*/trials/seed_*/geometry_records.json')):
        arm = path.parts[-5]
        if arm not in arms:
            continue
        for record in json.loads(path.read_text()):
            if record.get('kind') == 'geometry_step':
                arms[arm]['records'].append(len(json.dumps(record['record'], separators=(',', ':'))))
    summary = {}
    for arm, values in arms.items():
        sizes = values['records']
        summary[arm] = {'submitted_policies': sum('/'+arm+'/' in p['path'] for p in policies),
                        'training_checkpoint_count': len(sizes),
                        'checkpoint_characters': {'median': statistics.median(sizes), 'min': min(sizes), 'max': max(sizes)} if sizes else None,
                        'actor_input_tokens': {'sum': sum(values['inputs']), 'median': statistics.median(values['inputs'])} if values['inputs'] else None,
                        'actor_output_tokens': {'sum': sum(values['outputs']), 'median': statistics.median(values['outputs'])} if values['outputs'] else None,
                        'static_attribute_call_sites': dict(values['calls'])}
    return {'scope': 'Post-submission training-only analysis. Never supplied to actors. Attribute-name counts may include unrelated receivers; they are not runtime counts or evidence of internal reasoning. Checkpoint lengths include all recorded training steps, not only model-attached feedback. Token counts do not establish a latency effect.',
            'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'summary': summary, 'policies': policies}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summary'], indent=2))
