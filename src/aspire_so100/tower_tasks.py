"""Public symbolic goals for single columns; scene poses remain private."""
from dataclasses import dataclass

import numpy as np

from aspire_so100.block_catalog import CATALOG_OBJECTS


@dataclass(frozen=True)
class TowerRecipe:
    name: str
    order: tuple[str, ...]

    def __post_init__(self):
        if not 2 <= len(self.order) <= 4:
            raise ValueError('Single-column recipes require 2..4 levels')
        if len(set(self.order)) != len(self.order) or not set(self.order) <= set(CATALOG_OBJECTS):
            raise ValueError('A recipe must use distinct objects from the inventory')
        if any(n.endswith('triangular_prism') for n in self.order[:-1]):
            raise ValueError('An upright triangular prism is only allowed as the final cap')

    def public_goal(self):
        return {'kind': 'single_column', 'levels': [dict(zip(('color', 'shape'), name.split('_', 1)))
                                                  for name in self.order]}


RECIPES = {
    'cube_pair': ('blue_cube', 'red_cube'),
    'cube_triple': ('blue_cube', 'red_cube', 'green_cube'),
    'cube_quad': ('blue_cube', 'red_cube', 'green_cube', 'yellow_cube'),
    'mixed_triple': ('blue_prism', 'green_cube', 'red_cylinder'),
    'slender_top': ('yellow_cube', 'blue_cube', 'green_thin_prism'),
    'cylinder_base': ('green_cylinder', 'yellow_cube', 'red_cube'),
    'wedge_cap': ('blue_cube', 'green_cube', 'red_triangular_prism'),
}


def tower_recipe(name='cube_triple', *, seed=51, levels=None):
    if name == 'random':
        levels = 3 if levels is None else levels
        if type(levels) is not int or not 2 <= levels <= 4:
            raise ValueError('Random towers require 2..4 levels')
        rng = np.random.default_rng(seed)
        available = list(CATALOG_OBJECTS)
        chosen = []
        for i in range(levels):
            shapes = ('cube', 'prism', 'cylinder') if i < levels-1 else ('cube', 'prism', 'cylinder', 'thin_prism')
            options = [n for n in available if n.split('_', 1)[1] in shapes]
            item = str(rng.choice(options))
            chosen.append(item)
            available.remove(item)
        return TowerRecipe('random', tuple(chosen))
    if name not in RECIPES:
        raise ValueError('Unknown tower recipe: '+name)
    if levels is not None:
        raise ValueError('levels is only used with the random recipe')
    return TowerRecipe(name, RECIPES[name])
