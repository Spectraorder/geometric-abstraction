"""Export identical numerical observations with optional symbolic scene operators."""
import ast
from pathlib import Path
from build_mixed_pnp_observer import build as build_observer, ROOT
from run_vision_trial import validate_source


def build(symbolic=False):
    source = build_observer()
    modules = [('metric_scene', None)]
    if symbolic:
        modules += [('structure_geometry', {'convex_polygon', 'clip_polygon'}),
                    ('rigid_goals', None), ('symbolic_scene', None)]
    for module, keep in modules:
        nodes = []
        for node in ast.parse((ROOT/'src/aspire_so100'/f'{module}.py').read_text()).body:
            if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('aspire_so100'):
                continue
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and keep is not None and node.name not in keep:
                continue
            nodes.append(node)
        source += '\n'+ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    adapter = ast.parse((ROOT/'policies/mixed_scene_session.py').read_text())
    nodes = [node for node in adapter.body if symbolic or getattr(node, 'name', None) != 'SymbolicSession']
    source += '\n'+ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    validate_source(source)
    return source


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--symbolic', action='store_true')
    args = parser.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(args.symbolic))
