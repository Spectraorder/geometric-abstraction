"""Replay sensor-only release estimates, then separately score saved physics.

Development diagnostic only. No policy is changed or executed. The basic observer
is loaded from the archived candidate, and the mixed observer uses current CPU
geometry. Initial dimensions come from RGB-D, never the private scorer.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from aspire_so100.mixed_rigid_state import observe_mixed


def frame(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


class FrameAPI:
    def __init__(self, initial):
        self.frame = initial

    def camera(self):
        return self.frame


def estimate(trial, candidate):
    """Read only public goal, sensor files and recorded policy actions."""
    calls = json.loads((trial/'api_calls.json').read_text())
    from aspire_so100.structure_tasks import structure_recipe
    nodes = structure_recipe('thin_beam_bridge').public_goal()['nodes']
    cameras = [(i, c) for i, c in enumerate(calls) if c['method'] == 'camera']
    files = sorted(trial.glob('camera_*.npz'))
    if len(cameras) != len(files):
        raise ValueError('Camera count mismatch')
    paths = {i: p for (i, _), p in zip(cameras, files)}
    initial = frame(files[0]); api = FrameAPI(initial)
    namespace = {}; exec(compile(candidate.read_text(), str(candidate), 'exec'), namespace)
    pnp = namespace['RGBDPickPlace'](api)
    models = {}; initial_errors = {}
    for node in nodes:
        pnp.observe(node['color'])
        try:
            models[node['key']] = observe_mixed(initial, node['color'], node['shape'], None,
                                                 pnp.plane, node['orientation'])['model']
        except (ValueError, RuntimeError) as error:
            initial_errors[node['key']] = str(error)
    stages = [(i, c) for i, c in enumerate(calls) if c['method'] == 'record_geometry'
              and c['params']['record'].get('kind') == 'manual_development_stage']
    rows = []; previous = 0
    for end, event in stages:
        stage = event['params']['record']
        node = next(n for n in nodes if n['key'] == stage['key'])
        if stage['result']['status'] != 'released':
            previous = end; continue
        releases = [i for i in range(previous, end) if calls[i]['method'] == 'gripper'
                    and calls[i]['params'] == {'angle': .8, 'seconds': 1.5}]
        if len(releases) != 1:
            raise ValueError('Expected exactly one release for completed stage')
        release = releases[0]
        capture = max(i for i in paths if i < release)
        own = next(c for c in reversed(calls[:release]) if c['method'] == 'proprioception')['observation']
        api.frame = frame(paths[capture]); stamp = float(api.frame['time'])
        basic = pnp.observe(node['color'], np.asarray(own['tool_position']))
        row = {'key': node['key'], 'color': node['color'], 'shape': node['shape'],
               'time': stamp, 'post_release_time': event['time'], 'camera': paths[capture].name,
               'target_surface_m': stage['surface_m'], 'logged_xy_error_m': stage['result']['xy_error_m'],
               'basic_center_m': basic['center'].tolist()}
        replay_error = float(np.linalg.norm(basic['center'][:2]-np.asarray(stage['surface_m'])[:2]))
        if abs(replay_error-stage['result']['xy_error_m']) > 1e-10:
            raise ValueError('Frozen basic observer does not reproduce recorded release error')
        if node['key'] in models:
            try:
                mixed = observe_mixed(api.frame, node['color'], node['shape'], models[node['key']],
                                      pnp.plane, node['orientation'])
                row.update(mixed_center_m=mixed['center'].tolist(),
                           mixed_constrained_axes=np.asarray(mixed['constrained_axes']).tolist(),
                           mixed_inferred_orientation=mixed['inferred_orientation'])
            except (ValueError, RuntimeError) as error:
                row['mixed_unavailable'] = str(error)
        else:
            row['mixed_unavailable'] = 'Initial model unavailable: '+initial_errors[node['key']]
        rows.append(row); previous = end
    return {'trial': str(trial), 'rows': rows, 'initial_model_errors': initial_errors}


def score(packet):
    """Private physics is read only after every sensor estimate has been saved."""
    import mujoco
    from aspire_so100.playback import recorded_env_for_result
    trial = Path(packet['trial']); result = json.loads((trial/'result.json').read_text())
    with np.load(trial/'rollout.npz') as data:
        states = data['states']; state_type = mujoco.mjtState(int(data['state_type']))
    with TemporaryDirectory() as tmp:
        env = recorded_env_for_result(result, Path(tmp))
        try:
            def restore(stamp):
                index = int(np.argmin(abs(states[:, 0]-stamp)))
                if abs(states[index, 0]-stamp) > 1.1*env.model.opt.timestep:
                    raise ValueError('Saved physics timestamp mismatch')
                mujoco.mj_setState(env.model, env.data, states[index], state_type)
                mujoco.mj_forward(env.model, env.data)
            for row in packet['rows']:
                name = row['color']+'_'+row['shape']; gid = env.model.geom(name+'/'+name+'_geom').id
                restore(row['time']); before = env.data.geom_xpos[gid].copy()
                restore(row['post_release_time']); after = env.data.geom_xpos[gid].copy()
                target = np.asarray(row['target_surface_m'])
                row.update(true_center_before_m=before.tolist(), true_center_after_m=after.tolist(),
                           true_release_xy_error_mm=float(np.linalg.norm(before[:2]-target[:2])*1000),
                           true_post_release_xy_error_mm=float(np.linalg.norm(after[:2]-target[:2])*1000),
                           post_release_xy_movement_mm=float(np.linalg.norm(after[:2]-before[:2])*1000))
                for method in ['basic', 'mixed']:
                    if method+'_center_m' in row:
                        estimate = np.asarray(row[method+'_center_m'])
                        row[method+'_center_error_mm'] = float(np.linalg.norm(estimate-before)*1000)
                        row[method+'_xy_bias_mm'] = ((estimate-before)[:2]*1000).tolist()
            restore(float(states[-1, 0]))
            packet['corrected_final_status'] = env.structure_status(env.order)
            packet['original_success'] = result['success']
        finally:
            env.close()
    return packet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), args.root/'candidate.py',
                    Path('src/aspire_so100/mixed_rigid_state.py'), Path('src/aspire_so100/beam_geometry.py'),
                    Path('src/aspire_so100/column_geometry.py'), Path('src/aspire_so100/ablation_geometry.py'),
                    Path('src/aspire_so100/instance_geometry.py'), Path('src/aspire_so100/inventory_structure.py'),
                    Path('src/aspire_so100/structure_tasks.py')]
    hashes = {}
    for source in source_paths:
        data = source.read_bytes(); hashes[str(source)] = hashlib.sha256(data).hexdigest()
        (args.output/source.name).write_bytes(data)
    (args.output/'provenance.json').write_text(json.dumps(hashes, indent=2)+'\n')
    estimates = [estimate(p.resolve(), args.root/'candidate.py') for p in sorted(args.root.glob('seed_*')) if p.is_dir()]
    (args.output/'sensor_only_estimates.json').write_text(json.dumps(estimates, indent=2)+'\n')
    scored = [score(p) for p in estimates]
    (args.output/'private_scored_estimates.json').write_text(json.dumps(scored, indent=2)+'\n')
    for packet in scored:
        print(Path(packet['trial']).name, packet['corrected_final_status'])
        for row in packet['rows']:
            print(row['key'], {key: round(row[key], 3) for key in [
                'basic_center_error_mm', 'mixed_center_error_mm', 'true_release_xy_error_mm',
                'post_release_xy_movement_mm', 'true_post_release_xy_error_mm'] if key in row})
