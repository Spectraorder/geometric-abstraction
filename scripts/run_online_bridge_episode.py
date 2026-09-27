"""One continuous bridge-tower episode with a shared Astra/Luna agent loop."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

from online_bridge_actor import OnlineActor
from online_bridge_scene import OnlineBridgeTowerEnv, EpisodeSuccess
from online_bridge_skills import SensorAPI, SharedSkills, plain, append
from online_bridge_recovery import clear_object

CONTRACT = '''You control one continuing robot scene. Build the public support graph:
bridge foundations and beam, then a two-cube tower on its center. There is no reset.
Use the same previously trained skill library throughout this episode. Choose one
bounded operation; its consequences persist. Never claim physical success from an
image, a release status, or a geometric support candidate; the private evaluator
ends the episode after the full structure is stable for five seconds.

Available JSON actions:
- execute / bridge / node: place or repair one foundation or spanning beam.
- execute / stack / node: place or repair one single-support cube.
Both bind node to its public color key, reobserve at execution, and derive targets
from the current supports and the shared chosen worksite. No coordinates needed.
Existing correctly placed objects can be verified without moving them. These are
adapted training-selected policies, not perfect oracles. Failures return control.
- clear / none / node: move one obstructing object to an observed free table
  location outside the planned worksite. Use this when a fallen block prevents
  rebuilding its supports; rebuild supports before returning that block to its
  goal. Clear upper obstructions before moving something they rest on. This uses
  the same shared pick/place controller, with a bounded sensor-based free-space
  search. Unavailable observations can prevent clearing. It does not count as
  completing the object's task goal.
- observe / none / empty node: acquire fresh side and wrist RGB and metric depth images for the next
  decision. In Geometry Full this also refreshes compact geometric evidence.
- release / none / empty node: open the gripper at its CURRENT location. This may
  drop or topple an object; use only when holding is uncertain or to recover a
  blocked grasp, and reason about what is below it. Then reobserve.
- replan / none / empty node: ask Astra to reassess the plan. Luna should do this
  for unresolved failures or a damaged supporting structure.
- wait / none / empty node: let the scene settle for one second, then reason again.

The initial decision and explicit replans use Astra; routine decisions use Luna.
Two consecutive unconfirmed skill calls also trigger Astra. Both arms use this
rule. Keep the plan short and reason to one sentence. The plan is advisory, not an
automatically executed program. Support prerequisites matter. Retain valid work;
when a lower support moves, reconsider descendants rather than blindly continuing.
Skill status visually_consistent is observation-based and does not certify contact.
It is sufficient to continue to the next dependent skill unless newer evidence
shows a violation. Unknown support evidence alone does not imply a failed placement.
In geometric feedback, remaining_translation_m is the error; center_m and target_m
are absolute world coordinates. Do not re-place an object merely because its
absolute coordinate is nonzero or a residual lies within the stated tolerance.
If an object is already held, execute for THAT object to resume placement; a new
pick is not allowed. If its lift is uncertain, obtain evidence or release first.
Images and text are timestamped. Old poses are history, not current measurements.
All inference, observations, actions and recovery consume the episode time budget.
No tools, file access, policy code generation, or external information are allowed.
Return the requested JSON only. Use public color keys, not invented role names.
'''


def feedback(api, skills, arm, turn, *, images_requested=False, initial=False):
    images = []
    packet = {'time': skills.session.time(), 'holding': skills.holding(),
              'last_action': None if skills.last_result is None else {
                  k: skills.last_result[k] for k in ('status', 'action', 'skill', 'node', 'holding')
                  if k in skills.last_result}}
    if initial or arm == 'basic' or images_requested:
        for view in (('side', 'wrist') if initial or images_requested else ('side',)):
            path, calibration = api.image(f'feedback_{turn:03d}_{view}', view)
            images.append(path)
            packet.setdefault('cameras', {})[view] = calibration
            if images_requested:
                depth_path, encoding = api.depth_image(path)
                images.append(depth_path)
                packet['cameras'][view]['depth_display'] = encoding
    if arm == 'symbolic':
        packet['geometry'] = skills.packet()
    return plain(packet), images


def run(output, seed, arm, budget=360., *, realtime=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {'seed': seed, 'arm': arm, 'budget_s': budget, 'success': False,
              'status': 'initializing', 'realtime': realtime, 'decisions': 0}
    (output/'settings.json').write_text(json.dumps(result, indent=2)+'\n')
    snapshots = output/'sources'
    snapshots.mkdir()
    hashes = {}
    for name in ('run_online_bridge_episode.py', 'online_bridge_actor.py',
                 'online_bridge_scene.py', 'online_bridge_skills.py',
                 'online_bridge_recovery.py',
                 'build_online_bridge_geometry.py', 'verified_token_usage.py'):
        source = Path(__file__).with_name(name).read_bytes()
        (snapshots/name).write_bytes(source)
        hashes[name] = hashlib.sha256(source).hexdigest()
    (output/'source_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    env = OnlineBridgeTowerEnv(output, scene_seed=seed)
    api = None
    actor = None
    try:
        env.reset(seed)
        result['initial_physics_state_sha256'] = hashlib.sha256(env.data.qpos.tobytes()
                                                              +env.data.qvel.tobytes()).hexdigest()
        actor = OnlineActor(output)
        env.start_episode(budget, realtime=realtime)
        api = SensorAPI(env, output)
        init_started = time.monotonic()
        skills = SharedSkills(api)
        result['initialization_seconds'] = time.monotonic()-init_started
        (output/'skill_provenance.json').write_text(json.dumps(skills.provenance, indent=2)+'\n')
        (output/'goal.json').write_text(json.dumps(api.task(), indent=2)+'\n')
        slow = True
        want_images = False
        consecutive_failures = 0
        perception_seconds = result['initialization_seconds']
        turn = 0
        while True:
            env.advance_while_waiting()
            observation_started = time.monotonic()
            packet, images = feedback(api, skills, arm, turn,
                                      initial=turn == 0, images_requested=want_images)
            perception_seconds += time.monotonic()-observation_started
            result['initialization_and_feedback_seconds'] = perception_seconds
            packet['remaining_wall_seconds'] = max(0., env.deadline-time.monotonic())
            ready = time.monotonic()
            prefix = (CONTRACT+'\nGoal: '+json.dumps(api.task())+'\n'
                      +('Interface: geometric text plus optional images.\n' if arm == 'symbolic'
                        else 'Interface: current images and qualitative skill status.\n')) if turn == 0 else ''
            prompt = prefix+'Current feedback:\n'+json.dumps(packet, separators=(',', ':'))+'\nChoose the next operation.'
            append(output/'feedback.jsonl', {'turn': turn, 'packet': packet,
                                             'images': [str(p) for p in images]})
            decision = actor.request(prompt, images, slow=slow,
                advance=env.advance_while_waiting, deadline=env.deadline)
            result['decisions'] += 1
            if time.monotonic() >= env.deadline:
                raise TimeoutError('Episode deadline reached before action dispatch')
            row = {'turn': turn, 'decision': decision, 'slow_model': slow,
                   'feedback_to_dispatch_seconds': time.monotonic()-ready,
                   'sim_time_before_dispatch': float(env.data.time)}
            append(output/'dispatches.jsonl', row)
            # Reject invented keys before a model-selected operation reaches a skill.
            want_images = decision['action'] == 'observe'
            slow = decision['action'] == 'replan'
            if decision['action'] in ('execute', 'clear'):
                try:
                    outcome = (skills.execute(decision['skill'], decision['node'])
                               if decision['action'] == 'execute' else clear_object(skills, decision['node']))
                except ValueError as error:
                    outcome = {'status': 'invalid_action', 'error': str(error)}
                    skills.last_result = outcome
                row['result'] = outcome
                consecutive_failures = (0 if outcome['status'] in ('visually_consistent', 'cleared_visually')
                                        else consecutive_failures+1)
                slow = consecutive_failures >= 2
            elif decision['action'] == 'release':
                row['result'] = skills.release()
            elif decision['action'] == 'wait':
                api.wait(1.)
            append(output/'decisions.jsonl', row)
            turn += 1
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
            result['private_final_status'] = env.structure_status(list(env.order))
            result['private_success_evidence'] = env.success_evidence
        if api is not None:
            result.update(camera_acquisitions=api.camera_count, camera_seconds=api.camera_seconds,
                          action_seconds=api.action_seconds)
        if actor is not None:
            result['model_requests'] = actor.receipts
            result['completed_request_usage'] = {
                key: sum(item['usage'].get(key, 0) for item in actor.receipts)
                for key in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'total_tokens')}
            result['incomplete_requests'] = len(list(output.glob('decision_*/attempt_failure.json')))
            result['usage_complete'] = result['incomplete_requests'] == 0
        result['capped_completion_seconds'] = env.success_time if result['success'] else budget
        env.close()
        (output/'result.json').write_text(json.dumps(plain(result), indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--arm', choices=('basic', 'symbolic'), required=True)
    parser.add_argument('--budget', type=float, default=360.)
    args = parser.parse_args()
    result = run(args.output, args.seed, args.arm, args.budget)
    print(json.dumps({k: v for k, v in result.items() if k not in
                     ('model_requests', 'private_final_status')}, indent=2))


if __name__ == '__main__':
    main()
