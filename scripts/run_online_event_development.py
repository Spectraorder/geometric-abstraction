"""Development-only event-driven online episodes with bounded conversation windows."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

from online_bridge_actor import OnlineActor
from online_bridge_scene import OnlineBridgeTowerEnv, EpisodeSuccess
from online_bridge_skills import SensorAPI, append, plain
from online_event_skills import EventSkills, drive
from online_event_memory import PublicMemory, RetryGate, parse_plan, changed, validate_operation
from run_online_bridge_episode import feedback

CONTRACT = '''Control one continuing robot scene using the public support graph and shared skills.
There is no scene reset after failure. Geometry is sensor evidence, not a success oracle.
Choose JSON action execute/clear/observe/release/replan/wait, skill bridge/stack/none,
and public color node (empty for observe/release/replan/wait). execute bridge handles
foundations or the spanning beam; execute stack handles a single-support cube.
The same adapted bridge/stack skills and clearing primitive serve both interfaces.
clear moves one obstruction to observed free table space; it may be unavailable.
release opens at the current tool position, so consider what is below it.
observe requests side/wrist RGB and depth. wait settles locally for five seconds.

For execute, plan contains ONLY remaining operations as strings "bridge:red" or
"stack:blue", at most five, excluding the chosen action. Use actual public colors.
The local scheduler executes this plan while outcomes are visually_consistent.
It interrupts on failure or observed support movement and then asks for a new decision.
Plan all needed placements initially, preserving already consistent work. Old actions
and poses are historical. Skill calls refresh observations and bind current targets.
If holding is uncertain, observe or release. If holding is confirmed, resume that
object. A blocked operation failed with unchanged evidence: choose a different
recovery operation, view, or release instead of repeating it. Two distinct failed
operations may trigger Astra; replan explicitly only when a different strategy is needed.
Only request extra images when needed. Geometry Full normally uses current geometric
text; Visual Base uses images and qualitative status. Unknown does not mean false.
Translation residuals are target minus current geometry, not absolute coordinates.
The private success monitor ends the episode; never claim success yourself.
Return the required JSON. Keep reason to one sentence. No tools or code execution.
'''


def dump(path, value):
    path.write_text(json.dumps(plain(value), indent=2)+'\n')


def settle(env, seconds=5.5):
    end = time.monotonic()+seconds
    while time.monotonic() < end:
        env.advance_while_waiting()
        time.sleep(min(.01, max(0., end-time.monotonic())))


def compact_packet(packet):
    geometry = packet.get('geometry')
    if geometry:
        geometry.pop('semantics', None)
        for row in geometry['objects']:
            if row.get('placement', {}).get('translation_within_tolerance'):
                row.pop('center_m', None)
                row.pop('bottom_top_m', None)
    return packet


def run(output, seed, arm, budget=360., scripted=False, *, preparation=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    files = ('run_online_event_development.py', 'online_event_skills.py', 'online_event_memory.py',
             'online_bridge_actor.py', 'online_bridge_scene.py', 'online_bridge_skills.py', 'online_bridge_recovery.py',
             'online_recovery_fixture.py', 'run_online_recovery_development.py')
    dump(output/'sources.json', {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files})
    result = {'kind': 'development_only', 'seed': seed, 'arm': arm, 'budget_s': budget,
              'status': 'initializing', 'success': False, 'scripted': scripted}
    dump(output/'settings.json', result)
    env = OnlineBridgeTowerEnv(output, scene_seed=seed)
    actors = []; queue = []; memory = PublicMemory(); gate = RetryGate()
    api = None; skills = None; slow = True; want_images = False; requests = 0
    failures = set(); replanned = set(); suppressed = 0; executed = 0; dispatched = 0
    try:
        env.reset(seed)
        if preparation is None:
            env.start_episode(budget, realtime=not scripted)
        api = SensorAPI(env, output); skills = EventSkills(api)
        prepared = None
        if preparation is not None:
            prepared = preparation(env, api, skills)
            result['kind'] = 'controlled_recovery_development'
            result['setup_camera_acquisitions'] = api.camera_count
            result['setup_action_seconds'] = api.action_seconds
            api.camera_count = 0; api.action_seconds = 0.
            env.start_episode(budget, realtime=not scripted)
        dump(output/'goal.json', api.task()); dump(output/'skill_provenance.json', skills.provenance)
        if scripted:
            queue = [{'action': 'execute', 'skill': 'stack' if len(skills.nodes[key]['supports']) == 1 else 'bridge',
                      'node': key} for key in skills.order]
            if prepared is not None:
                queue = prepared['scripted_plan']
        while True:
            env.advance_while_waiting()
            evidence = skills.evidence()
            if not queue:
                if scripted:
                    # Accelerated physics smoke checks are not latency measurements.
                    api.wait(6.)
                    break
                if not actors or actors[-1].turn >= 3:
                    context = output/f'context_{len(actors):03d}'
                    context.mkdir()
                    actors.append(OnlineActor(context))
                actor = actors[-1]
                packet, images = feedback(api, skills, arm, requests, initial=requests == 0 and prepared is None,
                                          images_requested=want_images)
                packet = compact_packet(packet)
                if prepared is not None and actor.turn == 0:
                    packet['event_context'] = prepared['public']
                    if requests == 0:
                        images = prepared['images']+images
                packet['memory'] = memory.packet()
                packet['blocked_operations'] = [list(key) for key in gate.failed if gate.blocked(key, evidence)]
                packet['remaining_wall_seconds'] = max(0., env.deadline-time.monotonic())
                prefix = CONTRACT+'\nGoal: '+json.dumps(api.task())+'\n' if actor.turn == 0 else ''
                prompt = prefix+'Interface: '+arm+'\nCurrent feedback:\n'+json.dumps(packet, separators=(',', ':'))
                append(output/'feedback.jsonl', {'turn': requests, 'packet': packet, 'images': [str(p) for p in images]})
                decision = actor.request(prompt, images, slow=slow, advance=env.advance_while_waiting, deadline=env.deadline)
                requests += 1; want_images = False; slow = False
                append(output/'model_decisions.jsonl', {'turn': requests-1, 'decision': decision})
                try:
                    rest = parse_plan(decision['plan'], skills.nodes)
                    validate_operation(decision, skills.nodes)
                    queue = [decision]+(rest if decision['action'] == 'execute' else [])
                except ValueError:
                    memory.record(decision, {'status': 'invalid_plan'})
                    continue
            decision = queue.pop(0)
            operation = tuple(decision[k] for k in ('action', 'skill', 'node'))
            if decision['action'] in ('execute', 'clear'):
                if gate.blocked(operation, evidence):
                    suppressed += 1; queue.clear()
                    memory.record(decision, {'status': 'blocked_unchanged_evidence'})
                    append(output/'scheduler.jsonl', {'event': 'retry_suppressed', 'operation': operation})
                    if suppressed > 1:
                        # Poll locally, without repeated model calls, until evidence
                        # changes or the episode expires. Physics keeps advancing.
                        while not changed(evidence, skills.evidence()):
                            settle(env, 2.)
                    continue
                supports = skills.nodes[decision['node']]['supports']
                baseline = {key: evidence[key] for key in supports}
                last_check = skills.session.time()
                def boundary(event):
                    nonlocal last_check
                    env.advance_while_waiting()
                    if skills.holding() is None or event['time']-last_check < 2.:
                        return False
                    last_check = event['time']
                    for key, old in baseline.items():
                        fit = skills.bridge.observe(skills.session, api, key, attempts=1)
                        if old is not None and fit is not None:
                            import math
                            if math.dist(old, fit['center']) > .01:
                                return True
                    return False
                dispatched += 1
                append(output/'scheduler.jsonl', {'event': 'operation_started', 'operation': operation,
                                                'time': skills.session.time()})
                outcome = drive(skills, decision, boundary)
                success = outcome['status'] in ('visually_consistent', 'cleared_visually')
                gate.record(operation, skills.evidence(), success)
                executed += 1; suppressed = 0
                memory.record(decision, outcome)
                append(output/'scheduler.jsonl', {'event': 'operation_result', 'operation': operation, 'outcome': outcome,
                                                'queue_remaining': queue})
                if not success:
                    queue.clear(); failures.add(operation)
                    if len(failures) >= 2 and frozenset(failures) not in replanned:
                        slow = True; replanned.add(frozenset(failures))
                else:
                    failures.clear()
                if not queue:
                    if scripted and not success:
                        break
                    if success:
                        settle(env)
            elif decision['action'] == 'observe':
                want_images = True
                memory.record(decision, {'status': 'observation_requested'})
            elif decision['action'] == 'release':
                memory.record(decision, skills.release())
            elif decision['action'] == 'replan':
                key = frozenset(failures)
                slow = key not in replanned
                replanned.add(key)
                memory.record(decision, {'status': 'replan_requested' if slow else 'replan_already_requested'})
                if not slow:
                    settle(env)
            else:
                settle(env)
                memory.record(decision, {'status': 'settling_elapsed'})
    except EpisodeSuccess:
        result.update(status='success', success=True, time_to_success_s=env.success_time)
    except TimeoutError:
        result['status'] = 'timeout'
    except Exception as error:
        result.update(status='infrastructure_error', error=type(error).__name__+': '+str(error))
        (output/'error.txt').write_text(traceback.format_exc())
    finally:
        if env.episode_started is not None:
            result['timing'] = env.timing()
            result['private_success_evidence'] = env.success_evidence
        receipts = [row for actor in actors for row in actor.receipts]
        result.update(model_requests_completed=len(receipts), model_requests_attempted=sum(a.turn for a in actors),
                      contexts=len(actors), completed_operations=executed, dispatched_operations=dispatched,
                      usage={key: sum(row['usage'].get(key, 0) for row in receipts)
                             for key in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'total_tokens')},
                      model_seconds=sum(row['wall_seconds'] for row in receipts),
                      incomplete_requests=len(list(output.glob('context_*/decision_*/attempt_failure.json'))))
        if api is not None:
            result.update(camera_acquisitions=api.camera_count, action_seconds=api.action_seconds)
        result['capped_completion_seconds'] = env.success_time if result['success'] else budget
        if result['status'] == 'initializing': result['status'] = 'scripted_incomplete'
        if preparation is not None:
            result['private_final_structure'] = env.structure_status(list(env.order))
        env.close(); dump(output/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=range(2028100101, 2028100104), required=True)
    parser.add_argument('--arm', choices=('basic', 'symbolic'), required=True)
    parser.add_argument('--budget', type=float, default=360.)
    parser.add_argument('--scripted', action='store_true')
    args = parser.parse_args()
    result = run(args.output, args.seed, args.arm, args.budget, args.scripted)
    print(json.dumps(result, indent=2))
    if result['status'] == 'infrastructure_error': raise SystemExit(1)
