"""Audit the completed development probes without claiming agent learning gains."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_campaign(name):
    root = ROOT/'outputs'/name
    progress = json.loads((root/'progress.json').read_text())
    if progress.get('status') != 'completed':
        raise ValueError('Campaign is not complete: '+name)
    manifest = json.loads((root/'manifest.json').read_text())
    if sha(root/'candidate.py') != manifest['candidate_sha256']:
        raise ValueError('Candidate differs from manifest')
    rows = []
    for seed in manifest['seeds']:
        trial = root/f'seed_{seed}'
        result = json.loads((trial/'result.json').read_text())
        if result['seed'] != seed or result['candidate_sha256'] != manifest['candidate_sha256']:
            raise ValueError('Trial source or seed mismatch')
        if (result['task'] != 'inventory_structure' or result['scene']['recipe'] != 'thin_beam_bridge'
                or result['scene']['object_count'] != 4 or result['physics']['noslip_iterations'] != 0
                or result['sensor_blackout'] or not result['wrist_depth']):
            raise ValueError('Task, physics or sensor contract mismatch')
        if result['success'] != result['metrics']['success']:
            raise ValueError('Success fields disagree')
        record_path = trial/'geometry_records.json'
        records = json.loads(record_path.read_text()) if record_path.exists() else []
        if len(records) != result['geometry_record_count']:
            raise ValueError('Geometry record count mismatch')
        calls = json.loads((trial/'api_calls.json').read_text())
        observed = [r for r in records if r.get('kind') == 'development_observer']
        for call in calls:
            if call['method'] == 'record_geometry':
                record = call['params']['record']
                if record.get('kind') == 'development_observer':
                    if abs(call['time']-record['observation_time']) > 1e-6:
                        raise ValueError('Stale observer timestamp')
        rows.append({'seed': seed, 'success': result['success'], 'policy_error': result['policy_error'],
                     'camera_calls': result['camera_calls'], 'observer_records': len(observed),
                     'wrist_observations': sum(r.get('view') == 'wrist' for r in observed),
                     'unavailable_view_recoveries': sum(bool(r.get('rejected_views')) for r in observed),
                     'initial_image_sha256': sha(trial/'initial.png'),
                     'result_sha256': sha(trial/'result.json'),
                     'stages': [r for r in records if r.get('kind') == 'manual_development_stage']})
    return {'directory': name, 'manifest_sha256': sha(root/'manifest.json'),
            'candidate_sha256': manifest['candidate_sha256'], 'successes': sum(r['success'] for r in rows),
            'trials': len(rows), 'policy_errors': sum(r['policy_error'] is not None for r in rows), 'rows': rows}


def main(output):
    names = ['mixed_pnp_observer_development_01', 'mixed_pnp_observer_development_02',
             'mixed_pnp_observer_development_03', 'mixed_pnp_basic_development_04']
    campaigns = {name: read_campaign(name) for name in names}
    mixed = campaigns[names[2]]; basic = campaigns[names[3]]
    if campaigns[names[1]]['candidate_sha256'] != mixed['candidate_sha256']:
        raise ValueError('Mixed candidate changed before new-layout probe')
    if basic['candidate_sha256'] != sha(ROOT/'outputs/mixed_pnp_development_03/candidate.py'):
        raise ValueError('Basic candidate differs from archived original')
    pairs = []
    for first, second in zip(basic['rows'], mixed['rows'], strict=True):
        if (first['seed'] != second['seed'] or first['initial_image_sha256'] != second['initial_image_sha256']):
            raise ValueError('Paired scenes or initial images differ')
        pairs.append({'seed': first['seed'], 'basic_success': first['success'], 'mixed_success': second['success']})
    paired = {'pairs': pairs, 'both_success': sum(p['basic_success'] and p['mixed_success'] for p in pairs),
              'basic_only': sum(p['basic_success'] and not p['mixed_success'] for p in pairs),
              'mixed_only': sum(p['mixed_success'] and not p['basic_success'] for p in pairs),
              'both_failure': sum(not p['basic_success'] and not p['mixed_success'] for p in pairs)}
    report = {'schema': 'shared_pnp_observer_development_v1', 'campaigns': campaigns, 'paired_new_development': paired,
              'historical_executor_sha256': sha(ROOT/'policies/pick_place_rgbd_shared.py'),
              'source_verification': 'Campaign source hash matches the manifest and every runner-recorded result hash; temporary per-trial worker source copies are not retained.',
              'limits': ['Manual planner, no actor revisions and no test of symbolic representation or iteration efficiency.',
                        'Adaptive development variants on 75001–75004 are not independent replications.',
                        '75101–75108 were preselected new development layouts; both candidates stayed fixed across them.',
                        'The basic candidate was scored on those layouts after the mixed outcomes were visible; this is not a preregistered confirmatory study.',
                        'Mixed mode combines rigid fitting, alternate-view acquisition and robot visual masking; their individual contributions are not isolated.',
                        'All policy errors and unsuccessful partial structures remain failures. No threshold or physics relaxation.',
                        'No temporal pose prior or predicted object motion is used. Wrist selection uses the same simulation timestamp.',
                        'These scenes are now inspected development data and must not become a fresh reserved split.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'campaign_results': {key: [v['successes'], v['trials']] for key, v in campaigns.items()},
                      'paired': paired}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/evaluations/shared_pnp_observer_development.json')
    main(parser.parse_args().output)
