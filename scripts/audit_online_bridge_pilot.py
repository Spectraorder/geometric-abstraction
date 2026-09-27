"""Read-only evidence checks for completed online episodes; never rerun a policy."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from geometry_actor_session import DISABLED_TOOLS
from online_bridge_actor import audit_decision_events, validate_response
from run_online_bridge_episode import CONTRACT
from run_online_bridge_pilot import verify_sources
from summarize_online_bridge_pilot import summarize
from verified_token_usage import turn_usage


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def audit_episode(folder, protocol, slot):
    result = read(folder/'result.json')
    require(result['seed'] == slot['seed'] and result['arm'] == slot['arm'], 'Episode identity mismatch')
    require(result['budget_s'] == protocol['budget_s'], 'Episode budget mismatch')
    require(not (folder/'actor_home/auth.json').exists(), 'Completed episode retained temporary credentials')
    feedback = {row['turn']: row for row in records(folder/'feedback.jsonl')}
    dispatch = {row['turn']: row for row in records(folder/'dispatches.jsonl')}
    completed = {row['turn']: row for row in records(folder/'decisions.jsonl')}
    require(len(dispatch) <= result['decisions'], 'More dispatches than completed decisions')
    previous_transcript = b''
    previous_counts = []
    previous_totals = {}
    session_id = None
    receipts = []
    incomplete = []
    slow = True
    want_images = False
    failed_calls = 0
    simulation_advances = []
    directories = sorted(folder.glob('decision_*'))
    for index, directory in enumerate(directories):
        require(directory.name == f'decision_{index:03d}', 'Decision sequence has a gap')
        invocation = read(directory/'invocation.json')
        command = invocation['command']
        model = 'gpt-6-astra' if slow else 'gpt-6-luna'
        effort = 'medium' if slow else 'low'
        require(invocation['model'] == model and invocation['effort'] == effort, 'Model routing mismatch')
        require(invocation['session_before'] == session_id, 'Session lineage mismatch')
        for pair in (['--sandbox', 'read-only'], ['--model', model],
                     ['-c', 'project_doc_max_bytes=0'], ['-c', 'web_search="disabled"'],
                     ['-c', 'model_reasoning_effort="'+effort+'"']):
            require(any(command[i:i+2] == pair for i in range(len(command)-1)), 'Model command mismatch')
        require('--ignore-user-config' in command and '--ephemeral' not in command, 'Context isolation mismatch')
        for name in DISABLED_TOOLS:
            require(any(command[i:i+2] == ['--disable', name] for i in range(len(command)-1)), 'Tool isolation mismatch')
        require(('resume' in command) == (index > 0), 'Unexpected fresh/resumed session')
        if index:
            require(command[command.index('resume')+1] == session_id, 'Wrong resumed session')
        packet = feedback[index]['packet']
        require(('geometry' in packet) == (slot['arm'] == 'symbolic'), 'Treatment projection mismatch')
        count = 4 if want_images else (2 if index == 0 else int(slot['arm'] == 'basic'))
        require(len(invocation['images']) == count, 'Image schedule mismatch')
        require([x['source'] for x in invocation['images']] == feedback[index]['images'], 'Image source list mismatch')
        for image in invocation['images']:
            path = Path(image['source'])
            require(path.is_relative_to(folder), 'Image came from another episode')
            require(digest(path) == image['sha256'], 'Submitted image changed')
        prefix = (CONTRACT+'\nGoal: '+json.dumps(read(folder/'goal.json'))+'\n'
                  +('Interface: geometric text plus optional images.\n' if slot['arm'] == 'symbolic'
                    else 'Interface: current images and qualitative skill status.\n')) if index == 0 else ''
        expected_prompt = prefix+'Current feedback:\n'+json.dumps(packet, separators=(',', ':'))+'\nChoose the next operation.'
        prompt = (directory/'prompt.txt').read_text()
        require(prompt == expected_prompt, 'Prompt differs from the permitted feedback projection')
        if (directory/'attempt_failure.json').exists():
            require(index == len(directories)-1 and not (directory/'receipt.json').exists(), 'Incomplete turn was retried')
            started_ids = {e['thread_id'] for e in records(directory/'model.jsonl')
                           if e.get('type') == 'thread.started'}
            require(len(started_ids) <= 1, 'Incomplete request started multiple conversations')
            if started_ids:
                require(session_id is None or started_ids == {session_id}, 'Incomplete request changed conversations')
                session_id = next(iter(started_ids))
            incomplete.append(read(directory/'attempt_failure.json'))
            continue
        receipt = read(directory/'receipt.json')
        events = records(directory/'model.jsonl')
        raw_usage = audit_decision_events(events, model, index > 0)
        ids = {row['thread_id'] for row in events if row.get('type') == 'thread.started'}
        require(ids == {receipt['session_id']}, 'CLI session identity mismatch')
        require(session_id is None or session_id == receipt['session_id'], 'Conversation changed within the episode')
        session_id = receipt['session_id']
        transcript = (directory/'session.jsonl').read_bytes()
        require(transcript.startswith(previous_transcript), 'Persistent transcript history was rewritten')
        new_events = [json.loads(line) for line in transcript[len(previous_transcript):].decode().splitlines() if line.strip()]
        users = [e['payload'] for e in new_events if e.get('type') == 'response_item'
                 and e.get('payload', {}).get('role') == 'user']
        matched = [row for row in users if any(v.get('type') == 'input_text' and v.get('text') == prompt
                                               for v in row.get('content', []))]
        require(len(matched) == 1, 'Archived conversation does not contain the submitted prompt exactly once')
        embedded = [v['image_url'] for v in matched[0]['content'] if v.get('type') == 'input_image']
        actual_hashes = [hashlib.sha256(base64.b64decode(value.split(',', 1)[1])).hexdigest() for value in embedded]
        require(actual_hashes == [x['sha256'] for x in invocation['images']], 'Conversation images differ from submitted files')
        archive = [json.loads(line) for line in transcript.decode().splitlines()]
        counts = [e['payload']['info'] for e in archive
                  if e.get('type') == 'event_msg' and e.get('payload', {}).get('type') == 'token_count'
                  and e['payload'].get('info') is not None]
        require(counts[:len(previous_counts)] == previous_counts and bool(counts), 'Usage history changed')
        totals = counts[-1]['total_token_usage']
        usage = turn_usage(previous_totals, totals, raw_usage, counts[len(previous_counts):])
        require(usage == receipt['usage'] and raw_usage == receipt['raw_cumulative_usage'], 'Receipt token accounting mismatch')
        require(receipt['image_count'] == count and receipt['model'] == model, 'Receipt metadata mismatch')
        response = validate_response(read(directory/'response.json'))
        messages = [e['item']['text'] for e in events if e.get('type') == 'item.completed'
                    and e.get('item', {}).get('type') == 'agent_message']
        require(messages and json.loads(messages[-1]) == response, 'Saved decision differs from the model output')
        if index in dispatch:
            item = dispatch[index]
            require(item['decision'] == response and item['slow_model'] == slow, 'Dispatched decision changed')
            elapsed = item['sim_time_before_dispatch']-packet['time']
            require(elapsed > 0., 'Simulation did not advance during model inference')
            simulation_advances.append(elapsed)
        receipts.append(receipt)
        previous_transcript, previous_counts, previous_totals = transcript, counts, totals
        if index in completed:
            item = completed[index]
            require(item['decision'] == response, 'Completed decision changed')
            want_images = response['action'] == 'observe'
            slow = response['action'] == 'replan'
            if response['action'] in ('execute', 'clear'):
                failed_calls = (0 if item['result']['status'] in ('visually_consistent', 'cleared_visually')
                                else failed_calls+1)
                slow = failed_calls >= 2
        else:
            require(index == len(directories)-1, 'A nonterminal action record is missing')
    require(receipts == result['model_requests'], 'Episode receipt list differs from turn records')
    require(len(incomplete) == result['incomplete_requests'], 'Incomplete request count mismatch')
    require(result['usage_complete'] == (not incomplete), 'Usage completeness flag mismatch')
    for key, total in result['completed_request_usage'].items():
        require(total == sum(item['usage'][key] for item in receipts), 'Episode token sum mismatch')
    return {'session_id': session_id, 'completed_requests': len(receipts),
            'incomplete_requests': len(incomplete), 'verified_physics_advances': len(simulation_advances),
            'artifact_sha256': {str(path.relative_to(folder)): digest(path)
                               for path in folder.rglob('*') if path.is_file()
                               and 'actor_home' not in path.parts and 'actor_work' not in path.parts}}


def audit(root, partial=False):
    root = Path(root).resolve()
    protocol = verify_sources(root)
    results = []
    ids = set()
    pending = False
    for slot in protocol['slots']:
        folder = root/slot['path']
        if not (folder/'result.json').exists():
            require(partial, 'The reserved study is not complete')
            pending = True
            continue
        require(not pending, 'Completed episodes are not a registered prefix')
        report = audit_episode(folder, protocol, slot)
        if report['session_id'] is not None:
            require(report['session_id'] not in ids, 'A conversation was shared across episodes')
            ids.add(report['session_id'])
        results.append(dict(slot=slot, **report))
    if not pending:
        # Also validate paired initial states, success intervals, source/skill
        # parity and capped timing through the frozen result summarizer.
        summarize(root)
    return {'status': 'partial' if pending else 'passed', 'completed': len(results),
            'registered': len(protocol['slots']), 'protocol_sha256': digest(root/'protocol.json'),
            'auditor_sha256': digest(Path(__file__)), 'episodes': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(args.root, args.partial)
    if args.output:
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'episodes'}))
