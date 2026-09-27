"""Expose raw visuals or compact geometry above the same hidden CPU PnP system."""
from pathlib import Path
from build_geometry_step_session import library as step_library
from run_vision_trial import validate_source

ROOT = Path(__file__).resolve().parents[1]


def library(arm):
    source = step_library(arm)
    old = 'SurfaceStepSession(session) if SURFACE_ONLY else session'
    if source.count(old) != 1:
        raise ValueError('Expected one public session projection')
    source = source.replace(old, 'VisualInputSession(session) if SURFACE_ONLY else session')
    source += '\n'+(ROOT/'policies/visual_input_session.py').read_text()
    validate_source(source)
    return source


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('basic','symbolic'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(library(args.arm))
