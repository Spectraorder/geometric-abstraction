"""Reproducible tabletop layout distributions; positions stay inside the evaluator."""
import numpy as np

RANDOM_XY_BOUNDS = ((-.14, .14), (-.32, -.21))
CUBE_SIDE_M = 0.03
CUBE_HALF_SIZE = (CUBE_SIDE_M / 2,) * 3
BLOCK_HALF_SIZES = {
    name: CUBE_HALF_SIZE
    for name in ('blue_block', 'small_red_block', 'green_block')
}


def random_block_positions(seed):
    """Sample all colors in the same area, with 25 mm edge clearance, yaw=0."""
    rng = np.random.default_rng(seed)
    for _ in range(100):
        positions = {}
        for name, half in BLOCK_HALF_SIZES.items():
            for _ in range(10000):
                xy = rng.uniform([b[0] for b in RANDOM_XY_BOUNDS],
                                 [b[1] for b in RANDOM_XY_BOUNDS])
                if all(np.any(np.abs(xy - other) >=
                              np.array(half[:2]) + np.array(BLOCK_HALF_SIZES[key][:2]) + .025)
                       for key, other in positions.items()):
                    positions[name] = xy
                    break
            else:
                # Earlier placements can block every remaining valid region.
                # Resample the entire arrangement using the same seeded RNG.
                break
        if len(positions) == len(BLOCK_HALF_SIZES):
            return positions
    raise RuntimeError('Could not sample separated block positions')
