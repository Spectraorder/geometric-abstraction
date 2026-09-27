"""Export a numerical mixed observer and a shared PnP callback subclass.

No task targets, support graph selection or motion controller are exported from
geometry modules. The historical shared executor is included byte-for-byte.
"""
import ast
from pathlib import Path
from run_vision_trial import validate_source

ROOT = Path(__file__).resolve().parents[1]


def build():
    selection = {
        'cube_geometry': {'rgbd_points', 'estimate_cube_geometry'},
        'instance_geometry': {'inventory_color_mask', 'inventory_points', 'inventory_table'},
        'cuboid_geometry': {'estimate_cuboid_geometry'},
        'ablation_geometry': {'observe_cube'},
        'column_geometry': None,
        'beam_geometry': {'estimate_flat_beam_geometry'},
        'mixed_rigid_state': {'observe_mixed'},
        'cuboid_hypotheses': None,
        'pnp_observation': {'MixedPnPObservation'},
        'robot_visual_geometry': None,
    }
    bodies = []
    for module, keep in selection.items():
        tree = ast.parse((ROOT/'src/aspire_so100'/f'{module}.py').read_text())
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('aspire_so100'):
                continue
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if keep is not None and node.name not in keep:
                    continue
                if node.name in {'update_partial_column_geometry', 'select_column_levels'}:
                    continue
            bodies.append(node)
    source = ast.unparse(ast.Module(body=bodies, type_ignores=[]))
    source += '\ncolor_mask = inventory_color_mask\n'
    source += (ROOT/'policies/pick_place_rgbd_shared.py').read_text()
    source += '\n'+(ROOT/'policies/pick_place_rgbd_observed.py').read_text()
    validate_source(source)
    return source


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build())
