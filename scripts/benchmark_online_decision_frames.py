"""Measure decision preparation on development scenes without actions or model calls."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

from online_contact_scene import ContactCompleteEnv
from online_decision_frames import DecisionFrameReuse
from online_pick_clearance import ClearanceSensorAPI
from online_reasoning_skills import ReasoningSkills
from online_reasoning_state import project_feedback
from run_online_bridge_episode import feedback
from online_bridge_skills import plain


class CachedSensorAPI(DecisionFrameReuse, ClearanceSensorAPI):
    pass


def benchmark(output, seed, repetitions=3):
    output.mkdir(parents=True, exist_ok=False)
    env = ContactCompleteEnv(output, scene_seed=seed)
    records = []
    try:
        env.reset(seed)
        api = CachedSensorAPI(env, output)
        api.reserve_pick_space = True
        skills = ReasoningSkills(api)
        for index in range(repetitions):
            states = []
            for cached in ([False, True] if index % 2 == 0 else [True, False]):
                before = api.camera_count
                camera_seconds = api.camera_seconds
                stamp = skills.session.time()
                start = time.perf_counter()
                with api.decision_frames() if cached else nullcontext():
                    evidence = skills.evidence()
                    packet, images = feedback(api, skills, 'symbolic', 2*index+int(cached), initial=True)
                    packet = project_feedback(packet, skills.nodes)
                seconds = time.perf_counter()-start
                if stamp != skills.session.time():
                    raise ValueError('A decision readout advanced physics')
                # Paths differ between invocations; compare actual image bytes.
                state = plain(dict(evidence=evidence, packet=packet,
                                   images=[hashlib.sha256(p.read_bytes()).hexdigest() for p in images]))
                states.append(state)
                (output/f'state_{index}_{int(cached)}.json').write_text(json.dumps(state,indent=2)+'\n')
                records.append(dict(repetition=index, cached=cached, seconds=seconds,
                    camera_seconds=api.camera_seconds-camera_seconds,
                    acquisitions=api.camera_count-before,
                    reuse_hits=api.frame_reuse_hits if cached else 0))
            if states[0] != states[1]:
                raise ValueError('Cached and fresh decision evidence differ')
        result = dict(seed=seed, status='passed', model_requests=0, actions=0, records=records,
            equivalence='Evidence, projected feedback and image hashes match exactly at unchanged simulation time.',
            scope='Decision preparation only. Not end-to-end agent latency, token use or task success.')
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    except Exception as error:
        (output/'result.json').write_text(json.dumps(dict(seed=seed,status='failed',
            error=str(error),records=records,model_requests=0,actions=0),indent=2)+'\n')
        raise
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=range(2028100101,2028100104), required=True)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.output,args.seed),indent=2))
