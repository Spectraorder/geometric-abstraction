"""Audit live compact feedback independently of the production encoder/decoder."""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from audit_online_bound_development import audit
from run_online_bridge_pilot import verify_sources


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def reconstruct(value):
    """Independent expansion; do not import the implementation under evaluation."""
    if 'encoding' not in value:
        return deepcopy(value)
    if (value['encoding'] != 'shared_reference_metadata_v1'
            or set(value) != {'encoding', 'reference_defaults', 'geometry'}):
        raise ValueError('Unexpected wire format')
    defaults = value['reference_defaults']
    allowed = {'coordinate_frame', 'units', 'observation_time', 'source',
               'position_fully_observed', 'inferred_orientation', 'semantics'}
    if not set(defaults) <= allowed:
        raise ValueError('Unexpected shared field')
    restored = deepcopy(value['geometry'])
    for obj in restored['objects']:
        for role, ref in list(obj['references'].items()):
            if '$shared' not in ref:
                continue
            if ref['$shared'] is not True or ref.get('status') == 'unavailable':
                raise ValueError('Unqualified reference inherited metadata')
            combined = dict(defaults)
            combined.update({key: val for key, val in ref.items() if key != '$shared'})
            if combined.get('position_fully_observed') is not True:
                raise ValueError('Partial reference inherited metadata')
            obj['references'][role] = combined
    return restored


def check(root):
    protocol = verify_sources(root)
    if protocol['name'] != 'compact_text_live_development' or read(root / 'progress.json')['status'] != 'completed':
        raise ValueError('Expected a completed compact-text development check')
    interface = audit(root, protocol['variants'])
    reports = {}
    for variant in protocol['variants']:
        episode = root / variant
        result = read(episode / 'result.json')
        feedback = lines(episode / 'feedback.jsonl')
        prompts = sorted(episode.glob('context_*/decision_*/prompt.txt'))
        if len(prompts) != len(feedback):
            raise ValueError('Prompt and feedback counts differ')
        seen_encoding = False
        roundtrips = 0
        transcripts = 0
        original_bytes = encoded_bytes = 0
        for prompt_path, recorded in zip(prompts, feedback):
            text = prompt_path.read_text()
            prefix, marker, body = text.partition('Current feedback:\n')
            if not marker:
                raise ValueError('Missing feedback boundary')
            packet = json.loads(body)
            original = recorded['packet']
            encoded = packet['geometry'].get('encoding') == 'shared_reference_metadata_v1'
            explanation = prefix.count('Compact geometry uses shared_reference_metadata_v1.')
            if variant == 'raw' and (encoded or explanation):
                raise ValueError('Raw reference condition was changed')
            if variant == 'compact' and explanation != int(encoded and not seen_encoding):
                raise ValueError('Format explanation count mismatch')
            seen_encoding |= encoded
            decoded = deepcopy(packet)
            decoded['geometry'] = reconstruct(packet['geometry'])
            if canonical(decoded) != canonical(original):
                raise ValueError('Model-visible geometry differs from original feedback')
            roundtrips += 1
            encoded_bytes += len(text.encode())
            original_bytes += len(json.dumps(original, separators=(',', ':')).encode())
            session = prompt_path.parent / 'session.jsonl'
            if session.exists():
                messages = [row['payload'] for row in lines(session)
                            if row.get('type') == 'response_item' and row.get('payload', {}).get('role') == 'user']
                text_parts = [part.get('text') for msg in messages for part in msg.get('content', [])
                              if part.get('type') == 'input_text']
                if text not in text_parts:
                    raise ValueError('Archived prompt was not present in the actual model transcript')
                transcripts += 1
            elif not (prompt_path.parent / 'attempt_failure.json').exists():
                raise ValueError('Completed request has no transcript')
        if variant == 'compact' and not seen_encoding:
            raise ValueError('Compact condition never exercised')
        receipts = [read(path) for path in sorted(episode.glob('context_*/decision_*/receipt.json'))]
        decisions = [row['decision'] for row in lines(episode / 'model_decisions.jsonl')]
        reports[variant] = dict(status=result['status'], success=result['success'],
                               capped_seconds=result['capped_completion_seconds'],
                               usage=result['usage'], completed_requests=result['model_requests_completed'],
                               incomplete_requests=result['incomplete_requests'], model_seconds=result['model_seconds'],
                               first_request_seconds=receipts[0]['wall_seconds'],
                               mean_request_seconds=mean(row['wall_seconds'] for row in receipts),
                               images=sum(len(row['images']) for row in feedback),
                               camera_acquisitions=result['camera_acquisitions'],
                               action_seconds=result['action_seconds'], feedback_seconds=result['feedback_seconds'],
                               geometry_serialization=result['geometry_serialization'],
                               roundtrip_prompts=roundtrips, transcript_verified_prompts=transcripts,
                               prompt_bytes=encoded_bytes, unencoded_feedback_bytes=original_bytes,
                               actions=[{k: d[k] for k in ('action', 'skill', 'node', 'surface_point_world_m')} for d in decisions],
                               initial_state_sha256=result['private_initial_state_sha256'])
    if len({row['initial_state_sha256'] for row in reports.values()}) != 1:
        raise ValueError('Initial physical states differ')
    report = dict(status='passed', variants=reports, interface_audit=interface['status'],
                  campaign_wall_seconds=read(root / 'progress.json')['wall_seconds'],
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  interpretation='One reused development layout, not a reserved success-rate comparison. '
                  'New live trajectories may differ; incomplete usage remains unknown.')
    (root / 'compact_auditor.py').write_bytes(Path(__file__).read_bytes())
    (root / 'compact_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    progress = read(root / 'progress.json')
    progress.update(audit_status='passed', audit_file='compact_audit.json')
    (root / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    result = check(parser.parse_args().root)
    print(json.dumps(result, indent=2))
