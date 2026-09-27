"""Run a real three-cube rollout with opt-in wrist depth and geometry records."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from build_inventory_policy import build_policy
from run_vision_trial import validate_source

ROOT = Path(__file__).resolve().parents[1]


def build_demo(task='cube_triple'):
    if task not in ('cube_triple', 'thin_beam_bridge'):
        raise ValueError('Unknown geometry demo task')
    tree = ast.parse(build_policy('structure_policy' if task == 'thin_beam_bridge' else 'cube_alignment_policy'))
    if task == 'thin_beam_bridge':
        class PlanHook(ast.NodeTransformer):
            count = 0
            def visit_Assign(self, node):
                self.generic_visit(node)
                if len(node.targets)==1 and ast.unparse(node.targets[0])=='structure_origin[0]':
                    self.count += 1
                    return [node, ast.parse('api.geometry_plan(structure_origin[0], unit)').body[0]]
                return node
        hook = PlanHook()
        tree = hook.visit(tree)
        if hook.count != 2:
            raise ValueError('Structure plan instrumentation must match both origin assignments')
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'run':
            node.name = 'execute_column'
    modules = ('ablation_geometry', 'rigid_state', 'rigid_state_demo')
    if task == 'thin_beam_bridge':
        modules += ('mixed_rigid_state', 'mixed_state_demo')
    for name in modules:
        module = ast.parse((ROOT / 'src/aspire_so100' / (name + '.py')).read_text())
        tree.body.extend(node for node in module.body
                         if not (isinstance(node, ast.ImportFrom) and (node.module or '').startswith('aspire_so100')))
    source = '# Generated observation demo; existing cube controller with diagnostic RGB-D geometry.\n' + ast.unparse(tree) + '\n'
    validate_source(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=('cube_triple', 'thin_beam_bridge'), default='cube_triple')
    parser.add_argument('--seed', type=int, default=3501)
    parser.add_argument('--side-only', action='store_true', help='Keep wrist RGB for the paired sensor control')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    candidate = args.output / 'policy.py'
    candidate.write_text(build_demo(args.task))
    manifest = {'seed': args.seed, 'purpose': 'geometry_observation_demo_not_success_rate_evaluation',
                'controller': 'structure_policy (unique-color declared shapes)' if args.task=='thin_beam_bridge' else 'cube_alignment_policy (refine=False)',
                'task': args.task, 'wrist_depth': not args.side_only,
                'candidate_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
                'object_state_access': 'RGB-D only; robot pad geometry uses model and encoders',
                'new_wrist_geometry_controls_actions': False, 'noslip_iterations': 0}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    command = [sys.executable, str(ROOT / 'scripts/run_vision_trial.py'), '--candidate', str(candidate),
               '--output', str(args.output/'trial'), '--task', 'inventory_tower', '--layout', 'sampled',
               '--tower-recipe', 'cube_triple', '--object-count', '3', '--seed', str(args.seed),
               '--timeout', '240', '--record-geometry']
    if args.task == 'thin_beam_bridge':
        command[command.index('inventory_tower')] = 'inventory_structure'
        command[command.index('--tower-recipe')] = '--structure-recipe'
        command[command.index('cube_triple')] = args.task
        command[command.index('--object-count')+1] = '4'
    if not args.side_only:
        command.append('--wrist-depth')
    started = time.perf_counter()
    with (args.output/'trial.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    manifest['trial_wall_seconds'] = time.perf_counter()-started
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    subprocess.run([sys.executable, str(ROOT/'scripts/render_rigid_state_demo.py'), '--demo', str(args.output)], check=True)
    print(args.output)


if __name__ == '__main__':
    main()
