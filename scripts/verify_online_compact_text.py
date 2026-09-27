"""Replay development prompts to measure lossless text compaction without model calls."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import time
from online_compact_geometry import compact_prompt, expand_geometry, wire

DEVELOPMENT_SEEDS = frozenset(range(2028100101, 2028100104))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, output):
    if root.resolve() == output.resolve() or root.resolve() in output.resolve().parents:
        raise ValueError('Replay output must be separate from archived inputs')
    paths = sorted(root.glob('*/result.json'))
    if not paths:
        raise ValueError('No development episodes found')
    for path in paths:
        result = json.loads(path.read_text())
        if (result['seed'] not in DEVELOPMENT_SEEDS or result['kind'] != 'explicit_target_development'
                or result['scripted']):
            raise ValueError('Only non-scripted development data may be used')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'sources').mkdir()
    source_hashes = {}
    for name in ('online_compact_geometry.py', 'online_compact_actor.py', 'verify_online_compact_text.py'):
        path = Path(__file__).with_name(name)
        (output / 'sources' / name).write_bytes(path.read_bytes())
        source_hashes[name] = digest(path)
    hashes = {}
    rows = []
    for result_path in paths:
        episode = result_path.parent
        result = json.loads(result_path.read_text())
        hashes[str(result_path.resolve())] = digest(result_path)
        feedback_path = episode / 'feedback.jsonl'
        hashes[str(feedback_path.resolve())] = digest(feedback_path)
        feedback = [json.loads(line)['packet'] for line in feedback_path.read_text().splitlines()]
        prompt_paths = sorted(episode.glob('context_*/decision_*/prompt.txt'))
        if len(prompt_paths) != len(feedback):
            raise ValueError('Archived prompt and feedback counts differ')
        explained = False
        for index, prompt_path in enumerate(prompt_paths):
            prompt = prompt_path.read_text()
            packet = json.loads(prompt.partition('Current feedback:\n')[2])
            if wire(packet) != wire(feedback[index]):
                raise ValueError('Archived prompt does not match public feedback')
            hashes[str(prompt_path.resolve())] = digest(prompt_path)
            transformed, encoded = compact_prompt(prompt, explain=not explained)
            restored = json.loads(transformed.partition('Current feedback:\n')[2])
            if 'geometry' in restored:
                restored['geometry'] = expand_geometry(restored['geometry'])
            if wire(restored) != wire(packet):
                raise ValueError('Feedback round trip failed')
            if result['arm'] == 'basic' and transformed != prompt:
                raise ValueError('Base prompt changed')
            start = time.perf_counter()
            for _ in range(100):
                compact_prompt(prompt, explain=not explained)
            milliseconds = (time.perf_counter() - start) * 10.
            explained |= encoded
            target = output / episode.name / prompt_path.parent.name
            target.mkdir(parents=True)
            (target / 'prompt.txt').write_text(transformed)
            rows.append(dict(seed=result['seed'], arm=result['arm'], turn=index,
                             encoded=encoded, original_bytes=len(prompt.encode()),
                             compact_bytes=len(transformed.encode()),
                             mean_transform_ms=milliseconds,
                             original_prompt=str(prompt_path.resolve()),
                             compact_prompt=str(target.relative_to(output) / 'prompt.txt')))
    for path, expected in hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError('Input changed during replay')
    arms = {}
    for arm in sorted({row['arm'] for row in rows}):
        selected = [row for row in rows if row['arm'] == arm]
        before = sum(row['original_bytes'] for row in selected)
        after = sum(row['compact_bytes'] for row in selected)
        arms[arm] = dict(prompts=len(selected), encoded_prompts=sum(r['encoded'] for r in selected),
                         original_bytes=before, compact_bytes=after,
                         reduction_fraction=1 - after / before,
                         mean_transform_ms=mean(r['mean_transform_ms'] for r in selected))
    report = dict(status='passed', model_calls=0, rollouts=0, source_sha256=source_hashes,
                  input_sha256=hashes, arms=arms, prompts=rows,
                  interpretation='Fresh prompt UTF-8 bytes including the one-time format explanation. '
                  'Not model tokens, cached context size, runtime latency or task performance. '
                  'All geometry and other public feedback round-trip exactly; no numbers rounded.')
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'model_calls', 'rollouts', 'arms')}, indent=2))
