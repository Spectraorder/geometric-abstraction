"""Build a new cube-geometry candidate without overwriting evaluated snapshots."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WRAPPER = '''
        legacy_locate = locate

        def locate(name, model=None, near=None):
            # The baseline measures side length from the initial RGB-D top/table
            # separation. Reuse that measurement, never the simulator dimensions.
            estimate = legacy_locate(name, model, near)
            try:
                geometry = estimate_cube_geometry(api.camera(), name,
                    estimate['model']['height'], near=near, table_plane=plane)
            except ValueError:
                return estimate
            estimate['geometry_reliable'] = geometry['reliable']
            if not geometry['reliable']:
                return estimate
            side = geometry['side_m']
            estimate.update(center=geometry['centroid'], top=geometry['top_center'][2],
                normal=geometry['top_normal'], bottom=geometry['bottom_z'],
                extent_z=(geometry['top_z']-geometry['bottom_z'])/2,
                dims=np.array([side, side]), vertices=geometry['vertices'],
                top_vertices=geometry['top_vertices'])
            return estimate

'''


def build_policy():
    geometry = (ROOT / 'src/aspire_so100/cube_geometry.py').read_text()
    baseline = (ROOT / 'policies/stack_three_blocks_cubes_4cm.py').read_text()
    marker = "        blocks = {name: locate(name) for name in ('blue', 'red', 'green')}"
    if baseline.count(marker) != 1:
        raise ValueError('Baseline perception insertion point changed')
    policy = baseline.replace(marker, WRAPPER + marker)
    marker = "            if np.linalg.norm(aim_error) > 0.0055 or not contact_gap_min <= gap <= 0.004 or unsafe_overlap:"
    if policy.count(marker) != 1:
        raise ValueError('Baseline support check insertion point changed')
    policy = policy.replace(marker,
        "            if support.get('geometry_reliable', False):\n"
        "                unsafe_overlap = support_margin(measured['center'], support['top_vertices'], support['normal']) < .004\n"
        + marker)
    return geometry + '\n\n' + policy + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New candidate file; must not already exist')
    destination = parser.parse_args().output
    source = build_policy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        stream.write(source)
    print(destination)
