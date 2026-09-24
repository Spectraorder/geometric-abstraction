"""The current five-shape, four-color tabletop inventory (simulator metadata)."""
from dataclasses import dataclass

import numpy as np

BLOCK_COLORS = {
    'red': (.9, .1, .1, 1.),
    'yellow': (.95, .75, .05, 1.),
    'blue': (.1, .3, .9, 1.),
    'green': (.1, .7, .2, 1.),
}

# Full bounding dimensions in metres. The cylinder has radius 15 mm;
# the triangular prism is half of a 30 mm cube, cut diagonally in XZ.
SHAPE_DIMENSIONS = {
    'cube': (.03, .03, .03),
    'thin_prism': (.015, .03, .06),
    'prism': (.03, .03, .06),
    'cylinder': (.03, .03, .06),
    'triangular_prism': (.03, .03, .03),
}

CATALOG_OBJECTS = tuple(f'{color}_{shape}' for color in BLOCK_COLORS for shape in SHAPE_DIMENSIONS)
CATALOG_POSITIONS = {
    f'{color}_{shape}': (-.24 + .12*column, -.22 - .10*row, .40 + dimensions[2]/2)
    for row, color in enumerate(BLOCK_COLORS)
    for column, (shape, dimensions) in enumerate(SHAPE_DIMENSIONS.items())
}

# Task requirements describe identities, not measurements sent to the actor.
# All task generation lives here so future tasks can specify different roles.
TASK_OBJECTS = {'upright_thin_prism': {'target': 'red_thin_prism', 'base': 'blue_cube'}}


@dataclass(frozen=True)
class BlockLayout:
    task: str
    positions: dict[str, tuple[float, float, float]]
    roles: dict[str, str]


def generate_block_layout(seed: int = 51, count: int = 6, task: str = 'random',
                          *, required_objects: tuple[str, ...] | None = None) -> BlockLayout:
    """Sample unique inventory objects and separated poses, with task constraints.

    The upright task reserves central slots for its target and base, then samples
    distractors into remaining slots. Every object starts upright with zero yaw.
    A jittered grid bounds density and avoids rejection-sampling failures at 20.
    """
    if task not in ('random', 'catalog', 'single_column', *TASK_OBJECTS):
        raise ValueError('Unknown scene generation task: ' + task)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= len(CATALOG_OBJECTS):
        raise ValueError('Object count must be an integer from 1 to 20')
    if (task == 'single_column') != (required_objects is not None):
        raise ValueError('single_column requires an explicit ordered object list')
    if task == 'catalog':
        return BlockLayout(task, dict(CATALOG_POSITIONS), {})
    roles = dict(TASK_OBJECTS.get(task, {}))
    if required_objects is not None:
        if not 2 <= len(required_objects) <= 4 or len(set(required_objects)) != len(required_objects):
            raise ValueError('A column requires 2..4 distinct objects')
        if not set(required_objects) <= set(CATALOG_OBJECTS):
            raise ValueError('Required object is outside the inventory')
        roles = {f'level_{i}': name for i, name in enumerate(required_objects)}
    required = list(roles.values())
    if count < len(required):
        raise ValueError(f'{task} requires at least {len(required)} objects')
    rng = np.random.default_rng(seed)
    remaining = [name for name in CATALOG_OBJECTS if name not in required]
    selected = required + rng.choice(remaining, count-len(required), replace=False).tolist()
    slots = [(x, y) for y in (-.20, -.28, -.36, -.44) for x in (-.20, -.10, 0., .10, .20)]
    positions = {}
    for name in selected:
        available = slots
        if name in required:
            # Reserve central workspaces for task-critical objects.
            available = [(x, y) for x, y in slots if abs(x) <= .10 and -.36 <= y <= -.20]
        slot = available[int(rng.integers(len(available)))]
        slots.remove(slot)
        xy = np.asarray(slot)+rng.uniform(-.006, .006, 2)
        shape = name.split('_', 1)[1]
        positions[name] = (float(xy[0]), float(xy[1]), .40+SHAPE_DIMENSIONS[shape][2]/2)
    return BlockLayout(task, positions, roles)
