"""Six-object training layouts. These simulator dimensions are not actor inputs."""
import numpy as np

PRISM_HALF_SIZES = {
    'blue_block': (.015, .015, .015),
    'small_red_block': (.015, .015, .015),
    'green_block': (.015, .015, .015),
    'orange_prism': (.015, .015, .03),
    'purple_thin_prism': (.0075, .015, .03),
    'cyan_cylinder': (.015, .015, .03),
}
PRISM_OBJECTS = tuple(PRISM_HALF_SIZES)
PRISM_ORDER = ('blue_block', 'purple_thin_prism')
PRISM_NOMINAL = {
    name: (*xy, .40 + PRISM_HALF_SIZES[name][2])
    for name, xy in zip(PRISM_OBJECTS, ((-.04, -.23), (-.15, -.35), (.10, -.36),
                                      (-.14, -.23), (.08, -.25), (.17, -.23)))
}


def prism_positions(seed, layout='random_xy'):
    if layout not in ('nominal', 'random_xy'):
        raise ValueError('Unknown prism layout: '+layout)
    rng = np.random.default_rng(seed)
    if layout == 'nominal':
        return {name: np.asarray(pos[:2])+rng.uniform(-.004, .004, 2)
                for name, pos in PRISM_NOMINAL.items()}
    for _ in range(100):
        positions = {}
        for name in PRISM_OBJECTS:
            for _ in range(1000):
                xy = rng.uniform([-.16, -.36], [.16, -.20])
                if all(np.any(abs(xy-other) >= np.asarray(PRISM_HALF_SIZES[name][:2])+
                              np.asarray(PRISM_HALF_SIZES[key][:2])+.03)
                       for key, other in positions.items()):
                    positions[name] = xy
                    break
            else:
                break
        if len(positions) == len(PRISM_OBJECTS):
            return positions
    raise RuntimeError('Could not sample six separated objects')
