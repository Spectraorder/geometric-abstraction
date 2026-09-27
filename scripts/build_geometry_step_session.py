"""Build target-focused logging for either existing shared-observer interface."""
from pathlib import Path
from run_shared_observer_experiment import library as shared_library
from run_vision_trial import validate_source

ROOT = Path(__file__).resolve().parents[1]


def library(arm):
    source = shared_library(arm)
    source = source.replace('bind_session(SymbolicSession, context)',
                            'bind_session(GeometryStepSession, context)')
    source = source.replace('SurfaceProjectionSession(session) if SURFACE_ONLY',
                            'SurfaceStepSession(session) if SURFACE_ONLY')
    for name in ('geometry_step_state.py', 'geometry_step_session.py'):
        source += '\n'+(ROOT/'policies'/name).read_text()
    validate_source(source)
    return source


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('basic', 'symbolic'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(library(args.arm))
