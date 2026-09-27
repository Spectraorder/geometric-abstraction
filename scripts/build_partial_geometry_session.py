"""Build a development-only constraint extension over a frozen geometry library.

The existing fit acceptance and actuator are unchanged. A failed fit may retain
its unique current-view center bounds for reasoning, never as an executable pose.
"""
import argparse
import ast
from pathlib import Path
from run_vision_trial import validate_source

ROOT = Path(__file__).resolve().parents[1]


def library(source):
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == 'observe_cuboid_correspondence']
    if len(nodes) != 1: raise ValueError('Require exactly one frozen cuboid fitter')
    node = nodes[0]
    original = '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])
    marker = '    if np.any(widths[~axes] > max_hidden_width_m):'
    if original.count(marker) != 1: raise ValueError('Unexpected frozen fit acceptance boundary')
    # Reuse the identical evidence extraction and unique-correspondence checks.
    # Stop before pose acceptance and landmark construction. No enlarged width
    # threshold is used, and the returned fit is only an internal constraint set.
    helper = original.split(marker)[0].replace('def observe_cuboid_correspondence(',
        'def fit_current_cuboid_constraints(', 1) + '    return fit\n'
    start = "    def current_observation(self, frame, color, model, plane):\n        robot = self.api.gripper_geometry()"
    if source.count(start) != 1: raise ValueError('Expected one observation callback')
    source = source.replace(start, "    partial_constraints = None\n\n" +
        start.replace('        robot =', '        self.partial_constraints = None\n        robot ='), 1)
    for old_view, new_view in (
        ('        side, side_exclusion = filter_robot_visual_geometry(frame, robot)',
         "        side, side_exclusion = filter_robot_visual_geometry(frame, robot)\n        partial_views = [('side', side)]"),
        ("                measured = self.backend.observe_views([('side', side), ('wrist', wrist)], color, model, plane)",
         "                partial_views.append(('wrist', wrist))\n                measured = self.backend.observe_views([('side', side), ('wrist', wrist)], color, model, plane)")):
        if source.count(old_view) != 1: raise ValueError('Unexpected camera observation site')
        source = source.replace(old_view, new_view, 1)
    old = """        except (ValueError, RuntimeError) as error:
            self.ledger.unavailable(color, float(frame['time']), str(error))
            raise
"""
    new = """        except (ValueError, RuntimeError) as error:
            self.ledger.unavailable(color, float(frame['time']), str(error))
            if model is not None and self.backend.objects[color]['shape'] in ('prism', 'thin_prism'):
                try:
                    self.partial_constraints = partial_geometry_packet(color,
                        partial_views, self.backend.objects[color], model, plane)
                except (ValueError, RuntimeError):
                    pass
            raise
"""
    if source.count(old) != 1: raise ValueError('Expected one unavailable observation path')
    source = source.replace(old, new, 1)
    old = "        self.api.record_geometry({'kind': 'geometry_step', 'record': record})"
    new = """        if not surface_only and record['status'] == 'unavailable':
            partial = current_partial_geometry(self.partial_constraints, color, now)
            if partial is not None:
                record['constraints'] = partial
        self.api.record_geometry({'kind': 'geometry_step', 'record': record})"""
    if source.count(old) != 1: raise ValueError('Expected one compact geometry checkpoint')
    source = source.replace(old, new, 1)
    source += '\n' + helper + '\n' + (ROOT/'policies/partial_geometry_state.py').read_text()
    validate_source(source)
    return source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(library(args.frozen_library.read_text()))
