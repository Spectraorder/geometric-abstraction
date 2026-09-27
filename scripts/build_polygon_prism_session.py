"""Build the development polygon-prism extension without changing frozen sources."""
import argparse
import ast
from pathlib import Path

from build_geometry_session import library as geometry_library
from build_partial_geometry_session import library as partial_library
from run_vision_trial import validate_source

ROOT = Path(__file__).resolve().parents[1]


def definition(body, name):
    matches = [node for node in body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
               and node.name == name]
    if len(matches) != 1:
        raise ValueError('Expected exactly one definition: '+name)
    return matches[0]


def insert_dispatch(node, source):
    offset = int(bool(node.body and isinstance(node.body[0], ast.Expr)
                      and isinstance(node.body[0].value, ast.Constant)
                      and isinstance(node.body[0].value.value, str)))
    node.body[offset:offset] = ast.parse(source).body


def insert_after(body, statement, source):
    """Require a unique integration site so upstream changes fail explicitly."""
    matches = [i for i, item in enumerate(body) if ast.unparse(item) == statement]
    if len(matches) != 1:
        raise ValueError('Expected one integration site: '+statement)
    index = matches[0]+1
    body[index:index] = ast.parse(source).body


def validate_grasp_fraction(value):
    """Validate the optional actuator override for both CLI and Python callers."""
    if value is not None and (not isinstance(value, (int, float)) or not 0 <= value <= 1):
        raise ValueError('grasp_fraction must be finite and within 0..1')


def library(arm, *, grasp_fraction=None):
    """Extend observation and references, with an optional shared grasp default."""
    validate_grasp_fraction(grasp_fraction)
    tree = ast.parse(partial_library(geometry_library(arm)))
    if grasp_fraction is not None:
        # Override only an existing generic actuator parameter in this opt-in bundle.
        constructor = definition(definition(tree.body, 'RGBDPickPlace').body, '__init__')
        parameters = constructor.args.args[-len(constructor.args.defaults):]
        indices = [i for i, parameter in enumerate(parameters) if parameter.arg == 'grasp_fraction']
        if len(indices) != 1:
            raise ValueError('Expected one generic grasp_fraction parameter')
        constructor.args.defaults[indices[0]] = ast.Constant(float(grasp_fraction))
    legacy_shapes = ('cube', 'prism', 'thin_prism', 'cylinder')
    for owner, method in (('ObservationContext', '__init__'),
                          ('MixedPnPObservation', '__init__'), ('MetricScene', 'observe')):
        node = definition(definition(tree.body, owner).body, method)
        matches = [item for item in ast.walk(node) if isinstance(item, ast.Tuple)
                   and all(isinstance(value, ast.Constant) for value in item.elts)
                   and tuple(value.value for value in item.elts) == legacy_shapes]
        if len(matches) != 1:
            raise ValueError('Expected one shape contract in '+owner)
        matches[0].elts.append(ast.Constant('polygon_prism'))

    node = definition(definition(tree.body, 'SessionFactory').body, 'session')
    matches = [item for item in ast.walk(node) if isinstance(item, ast.Dict)
               and all(isinstance(key, ast.Constant) for key in item.keys)
               and set(key.value for key in item.keys) == set(legacy_shapes)]
    if len(matches) != 1:
        raise ValueError('Expected one initial orientation contract')
    matches[0].keys.append(ast.Constant('polygon_prism'))
    matches[0].values.append(ast.Constant('upright'))

    insert_dispatch(definition(tree.body, 'observe_mixed'), """
if shape == 'polygon_prism':
    return observe_polygon_prism(frame, color, model, table_plane)
""")
    node = definition(definition(tree.body, 'MetricScene').body, 'observe')
    matches = [item for item in ast.walk(node) if isinstance(item, ast.Compare)
               and ast.unparse(item) == "shape != 'cylinder'"]
    if len(matches) != 1:
        raise ValueError('Expected one box vertex-count contract')
    matches[0].ops = [ast.NotIn()]
    matches[0].comparators = [ast.Tuple(elts=[ast.Constant('cylinder'),
                                            ast.Constant('polygon_prism')], ctx=ast.Load())]
    insert_dispatch(node, """
if fit.get('shape') == 'polygon_prism':
    validate_polygon_prism(fit)
""")
    insert_dispatch(definition(tree.body, 'scene_face'), """
if fit['shape'] == 'polygon_prism':
    return polygon_prism_face(fit, top)
""")
    node = definition(definition(tree.body, 'SymbolicScene').body, 'reference')
    insert_after(node.body, 'fit = self.metric.get(name, stamp)', """
if fit['shape'] == 'polygon_prism':
    return polygon_prism_reference(fit, reference, role)
""")

    node = definition(definition(tree.body, 'MixedPnPObservation').body, 'observe_views')
    if not isinstance(node.body[-1], ast.Raise):
        raise ValueError('Expected a terminal unavailable-view error')
    node.body[-1:-1] = ast.parse("""
if model is not None and self.objects[color]['shape'] == 'polygon_prism' and len(frames) > 1:
    try:
        measured = observe_partial_polygon_views(frames, color, model, table_plane)
        return dict(measured, rejected_views=failures)
    except ValueError as error:
        failures.append({'view': 'joint_polygon_registration', 'reason': str(error)})
""").body

    # Preserve conditional evidence in compact text, without adding full geometry.
    node = definition(definition(tree.body, 'GeometryStepState').body, 'update')
    insert_after(node.body, "output.update(status='current_estimate', geometry=geometry)", """
if fit['shape'] == 'polygon_prism':
    geometry['observability_condition'] = fit['observability_condition']
    geometry['vertex_id_scope'] = fit['vertex_id_scope']
""")
    loops = [item for item in ast.walk(node) if isinstance(item, ast.For)
             and ast.unparse(item.target) == 'ref' and ast.unparse(item.iter) == 'references']
    if len(loops) != 1:
        raise ValueError('Expected one requested-reference checkpoint loop')
    loops[0].body += ast.parse("""
if 'vertex_sources' in ref:
    requested[role]['evidence_sources'] = sorted(set(ref['vertex_sources']))
    requested[role]['id_scope'] = ref['id_scope']
""").body

    # Full packets expose connectivity only when details are explicitly requested.
    node = definition(definition(tree.body, 'MetricScene').body, 'packet')
    branches = [item for item in ast.walk(node) if isinstance(item, ast.If)
                and ast.unparse(item.test) == 'current']
    if len(branches) != 1:
        raise ValueError('Expected one current-object packet branch')
    insert_after(branches[0].body, 'row.update({k: metric_copy(fit[k]) for k in keys if k in fit})', """
if fit['shape'] == 'polygon_prism':
    row['observability_condition'] = fit['observability_condition']
    row['vertex_id_scope'] = fit['vertex_id_scope']
    if details:
        for key in ('edge_vertex_indices', 'face_vertex_indices', 'vertex_sources'):
            row[key] = metric_copy(fit[key])
""")
    # Distinguish robot-model exclusions from missing sensor depth in visibility checks.
    node = definition(tree.body, 'exclude_robot_spheres')
    insert_after(node.body, "result['valid_depth'] = valid & ~excluded", """
result['robot_excluded_depth'] = excluded
""")
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree)+'\n'
    for name in ('point_geometry.py', 'polygon_prism/geometry.py', 'polygon_prism/partial_boundary.py'):
        source += (ROOT/'prototypes'/name).read_text()+'\n'
    validate_source(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('basic', 'symbolic'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grasp-fraction', type=float, default=None,
                        help='Override the existing generic actuator default; omission preserves it')
    args = parser.parse_args()
    try:
        source = library(args.arm, grasp_fraction=args.grasp_fraction)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(source)


if __name__ == '__main__':
    main()
