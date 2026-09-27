"""Direct construction of the shared geometry session and public projections.

No experiment driver imports or source-string replacement chain is needed.
Geometry, relation/goal operators and compact checkpoints remain available.
"""
import argparse
from pathlib import Path
from build_mixed_scene import build
from run_vision_trial import validate_source

ROOT = Path(__file__).resolve().parents[1]
FACTORY = """
class SessionFactory:
    @staticmethod
    def session(api, objects):
        inventory = api.task()['nodes']
        # Public initial conditions, independent of desired goal orientation.
        initial = {'cube': 'upright', 'prism': 'upright',
                   'thin_prism': 'flat_x', 'cylinder': 'upright'}
        context = {'schema': 'initial_observation_context_v1',
                   'source': 'public_initial_scene_description',
                   'objects': [{'color': obj['color'], 'shape': obj['shape'],
                                'initial_orientation_assumption': initial[obj['shape']]}
                               for obj in inventory]}
        session = bind_session(GeometryStepSession, context)(api, objects)
        return VisualInputSession(session) if SURFACE_ONLY else session
GEOMETRY = SessionFactory()
"""


def library(arm, *, visual=True):
    if arm not in ('basic', 'symbolic'):
        raise ValueError('Unsupported geometry arm')
    source = build(True)
    for name in ('observation_context_prototype.py', 'surface_projection_prototype.py'):
        source += '\n'+(ROOT/'scripts'/name).read_text()
    source += '\nSURFACE_ONLY = '+str(arm == 'basic')+'\n'
    source += FACTORY if visual else FACTORY.replace('VisualInputSession(session)', 'SurfaceStepSession(session)')
    for name in ('geometry_step_state.py', 'geometry_step_session.py'):
        source += '\n'+(ROOT/'policies'/name).read_text()
    if visual:
        source += '\n'+(ROOT/'policies/visual_input_session.py').read_text()
    validate_source(source)
    return source


def surface_library(arm):
    """Shared-observer projection for retained surface-interface diagnostics."""
    return library(arm, visual=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('basic', 'symbolic'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = library(args.arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(source)
