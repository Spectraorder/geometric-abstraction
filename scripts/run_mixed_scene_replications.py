"""Independent numeric/symbolic chains with a study-wide evaluation freeze gate."""
import argparse
import json
from pathlib import Path
import time
from run_mixed_scene_experiment import prepare, campaign, digest, write, assert_frozen

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('numeric', 'symbolic')


def read(path):
    return json.loads(path.read_text())


def initialize(root, pilot):
    """Register every split and verify unchanged runtime before generating actors."""
    previous = read(pilot/'manifest.json')
    root.mkdir(parents=True, exist_ok=False)
    replicas = []
    reserved = list(range(79001, 79021))
    for index, start in enumerate((78001, 78101, 78201), 1):
        directory = root/f'replica_{index:02d}'
        manifest = prepare(directory, 3, list(range(start, start+5)), reserved, ARMS)
        for arm in ARMS:
            if manifest['library_sha256'][arm] != previous['library_sha256'][arm]:
                raise RuntimeError('Representation changed since the pilot')
        for field in ('starter_sha256', 'executor_sha256', 'model', 'reasoning_effort',
                      'max_candidate_characters', 'max_prompt_characters', 'physics', 'sensor',
                      'recipe', 'feedback', 'temporal_pose_fusion', 'conditional_prediction'):
            if manifest[field] != previous[field]:
                raise RuntimeError('Method changed since pilot: '+field)
        changed = [name for name, sha in manifest['source_sha256'].items()
                   if previous['source_sha256'].get(name) != sha]
        if changed != ['scripts/run_mixed_scene_experiment.py']:
            raise RuntimeError('Only arm-selection orchestration may change: '+str(changed))
        replicas.append({'directory': directory.name, 'train_seeds': manifest['train_seeds'],
                         'manifest_sha256': digest(directory/'manifest.json')})
    protocol = {
        'schema': 'mixed_scene_replications_v1',
        'registered_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'arms': list(ARMS), 'replicas': replicas, 'heldout_seeds': reserved,
        'training_revisions_per_arm': 3, 'training_slots': 120, 'reserved_slots': 480,
        'model_generations': 18, 'all_training_frozen_before_any_reserved': True,
        'source_pilot': str(pilot), 'source_pilot_manifest_sha256': digest(pilot/'manifest.json'),
        'coordinator_sha256': digest(Path(__file__)),
        'method': 'Unchanged pilot representations, observer, starter, executor and actor feedback.',
        'independence': 'Separate isolated model contexts and disjoint training seeds; common fresh reserved layouts.',
        'primary_outcomes': ['per-chain reserved SR at every revision',
                             'per-chain mean reserved SR across R0-R3',
                             'first reserved revision reaching 80 percent',
                             'training-selected reserved SR'],
        'analysis': 'Report all three paired chains, including regressions. Common reserved layouts and correlated revisions must not be pooled as independent Bernoulli trials.',
        'selection': 'Highest training SR, earliest revision breaks ties; no reserved selection.',
        'exclusions': 'No previous candidates, development planner, checkpoint prototype, failure audit or reserved feedback in actor contexts.',
        'storage': 'Keep full existing rollout and sensor records. Three GiB free-space guard. Train phase may finish before capacity for evaluation is available.'}
    write(root/'protocol.json', protocol)
    (root/'coordinator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return protocol


def check(root, protocol):
    if digest(Path(__file__)) != protocol['coordinator_sha256']:
        raise RuntimeError('Coordinator changed after freeze')
    if digest(root/'coordinator_snapshot.py') != protocol['coordinator_sha256']:
        raise RuntimeError('Archived coordinator changed')
    for row in protocol['replicas']:
        directory = root/row['directory']
        if digest(directory/'manifest.json') != row['manifest_sha256']:
            raise RuntimeError('Replica manifest changed')
        manifest = read(directory/'manifest.json')
        if manifest['arms'] != protocol['arms'] or manifest['heldout_seeds'] != protocol['heldout_seeds']:
            raise RuntimeError('Replica protocol mismatch')
        assert_frozen(directory, manifest)


def freeze(root, protocol):
    """Preserve the original gate timestamp on resume; verify every revision."""
    check(root, protocol)
    frozen = {}
    for row in protocol['replicas']:
        directory = root/row['directory']
        trained = read(directory/'training_summary.json')
        if trained['manifest_sha256'] != row['manifest_sha256']:
            raise RuntimeError('Training provenance mismatch')
        local = read(directory/'all_candidates_frozen.json')
        if local != {'selected': trained['frozen'], 'revisions': trained['revision_frozen']}:
            raise RuntimeError('Replica freeze mismatch')
        for arm in protocol['arms']:
            if digest(directory/arm/'frozen.py') != trained['frozen'][arm]['sha256']:
                raise RuntimeError('Selected candidate changed')
            for entry in trained['revision_frozen'][arm]:
                if digest(directory/arm/'frozen_revisions'/f"revision_{entry['revision']:03d}.py") != entry['sha256']:
                    raise RuntimeError('Frozen revision changed')
        frozen[row['directory']] = local
    gate = {'protocol_sha256': digest(root/'protocol.json'), 'replicas': frozen}
    path = root/'all_candidates_frozen.json'
    if path.exists():
        if read(path) != gate:
            raise RuntimeError('Study-wide freeze changed')
    else:
        write(path, gate)
    return gate


def run(root, protocol, phase='all'):
    check(root, protocol)
    if phase != 'evaluate':
        for row in protocol['replicas']:
            directory = root/row['directory']
            if not (directory/'training_summary.json').exists():
                write(root/'progress.json', {'status': 'training', 'replica': row['directory']})
                campaign(directory, read(directory/'manifest.json'), 'train')
                print('TRAINING FROZEN '+row['directory'], flush=True)
    gate = freeze(root, protocol)
    if phase == 'train':
        write(root/'progress.json', {'status': 'trained', 'frozen': gate})
        return
    for row in protocol['replicas']:
        check(root, protocol)
        directory = root/row['directory']
        if not (directory/'summary.json').exists():
            write(root/'progress.json', {'status': 'evaluating', 'replica': row['directory']})
            campaign(directory, read(directory/'manifest.json'), 'evaluate')
    write(root/'progress.json', {'status': 'completed', 'frozen': gate})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=ROOT/'outputs/mixed_scene_learning_pilot_01_schema_corrected')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--phase', choices=('all', 'train', 'evaluate'), default='all')
    args = parser.parse_args()
    if args.phase == 'evaluate' and not args.resume:
        parser.error('Evaluation requires registered frozen training')
    root = args.output.resolve()
    protocol = read(root/'protocol.json') if args.resume else initialize(root, args.pilot.resolve())
    run(root, protocol, args.phase)
