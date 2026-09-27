"""Verify actual compact-study model prompts against the original sensor feedback."""
import hashlib
import json
from pathlib import Path
from audit_online_compact_check import canonical, lines, reconstruct


def check_feedback(prompt, original, arm, *, explanation_needed):
    prefix, marker, body = prompt.partition('Current feedback:\n')
    if not marker:
        raise ValueError('Missing model feedback boundary')
    packet = json.loads(body)
    encoded = packet.get('geometry', {}).get('encoding') == 'shared_reference_metadata_v1'
    explanations = prefix.count('Compact geometry uses shared_reference_metadata_v1.')
    if arm == 'basic':
        if 'geometry' in packet or explanations:
            raise ValueError('Geometric feedback exposed to Base')
    elif arm == 'symbolic':
        if explanations != int(encoded and explanation_needed):
            raise ValueError('Compact explanation missing or repeated')
        packet['geometry'] = reconstruct(packet['geometry'])
    else:
        raise ValueError('Unknown arm')
    if canonical(packet) != canonical(original):
        raise ValueError('Model prompt did not preserve original feedback')
    return encoded


def audit_prompts(root, slots):
    reports = []
    for slot in slots:
        episode = root / slot['path']
        result = json.loads((episode / 'result.json').read_text())
        if result['geometry_serialization'] != 'shared_reference_metadata_v1':
            raise ValueError('Wrong serialization configuration')
        feedback = lines(episode / 'feedback.jsonl')
        prompts = sorted(episode.glob('context_*/decision_*/prompt.txt'))
        if len(prompts) != len(feedback):
            raise ValueError('Prompt and feedback counts differ')
        encoded_count = 0
        transcript_count = 0
        for prompt_path, row in zip(prompts, feedback):
            prompt = prompt_path.read_text()
            encoded_count += check_feedback(prompt, row['packet'], slot['arm'], explanation_needed=encoded_count == 0)
            transcript = prompt_path.parent / 'session.jsonl'
            if transcript.exists():
                messages = [event['payload'] for event in lines(transcript)
                            if event.get('type') == 'response_item' and event.get('payload', {}).get('role') == 'user']
                text = [part.get('text') for message in messages for part in message.get('content', [])
                        if part.get('type') == 'input_text']
                if prompt not in text:
                    raise ValueError('Prompt absent from actual model transcript')
                transcript_count += 1
            elif not (prompt_path.parent / 'attempt_failure.json').exists():
                raise ValueError('Completed request lacks transcript')
        reports.append(dict(episode=slot['path'], arm=slot['arm'], seed=slot['seed'],
                            checked_prompts=len(prompts), encoded_prompts=encoded_count,
                            transcript_verified_prompts=transcript_count,
                            prompts_without_transcript=len(prompts) - transcript_count))
    report = dict(status='passed', episodes=reports,
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Independent reconstruction, unavailable-state preservation, Base exclusion and actual model transcripts. '
                  'An interrupted request may have no transcript; that absence is reported, not treated as verified delivery.')
    (root / 'compact_prompt_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
