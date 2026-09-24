"""Symbolic low-rise support graphs; no scene poses in public goals."""
from dataclasses import dataclass
import math

from aspire_so100.block_catalog import CATALOG_OBJECTS


@dataclass(frozen=True)
class StructureNode:
    key: str
    object_name: str
    grid: tuple[float, float]
    supports: tuple[str, ...] = ()
    orientation: str = 'upright'


@dataclass(frozen=True)
class StructureRecipe:
    name: str
    nodes: tuple[StructureNode, ...]
    unit_node: str

    def __post_init__(self):
        if not 3 <= len(self.nodes) <= 8:
            raise ValueError('Structures require 3..8 objects')
        keys = [n.key for n in self.nodes]
        objects = [n.object_name for n in self.nodes]
        if any(not isinstance(k, str) or not k for k in keys) or len(set(keys)) != len(keys):
            raise ValueError('Node keys must be nonempty and distinct')
        if len(set(objects)) != len(objects) or not set(objects) <= set(CATALOG_OBJECTS):
            raise ValueError('Structure objects must be distinct inventory identities')
        if self.unit_node not in keys:
            raise ValueError('The visual length reference must be a node')
        reference = self.nodes[keys.index(self.unit_node)]
        if not reference.object_name.endswith('_cube') or reference.supports:
            raise ValueError('The visual length reference must be a table-supported cube')
        seen = set()
        for node in self.nodes:
            if len(node.grid) != 2 or not all(math.isfinite(v) and abs(v) <= 6 for v in node.grid):
                raise ValueError('Grid coordinates must be finite, bounded XY pairs')
            if len(set(node.supports)) != len(node.supports) or not set(node.supports) <= seen:
                raise ValueError('Support graph must be acyclic and in construction order')
            if node.object_name.endswith('_triangular_prism'):
                raise ValueError('Triangular structures require a validated orientation skill')
            if node.orientation not in ('upright', 'flat_x'):
                raise ValueError('Unknown desired object orientation')
            if node.orientation == 'flat_x' and not node.object_name.endswith('_thin_prism'):
                raise ValueError('The flat X skill currently supports thin prisms only')
            seen.add(node.key)
        if len({tuple(n.grid) for n in self.nodes if not n.supports}) < 2:
            raise ValueError('A structure requires at least two distinct foundation locations')

    @property
    def order(self):
        return tuple(n.object_name for n in self.nodes)

    def public_goal(self):
        return {'kind': 'support_graph', 'length_unit': {'observed_cube': self.unit_node},
                'axes': 'world_xy', 'nodes': [
                    {'key': n.key, 'color': n.object_name.split('_', 1)[0],
                     'shape': n.object_name.split('_', 1)[1], 'grid': list(n.grid),
                     'supports': list(n.supports), 'orientation': n.orientation} for n in self.nodes]}


STRUCTURE_RECIPES = {
    'thin_beam_bridge': (
        StructureNode('left', 'blue_cube', (-1., 0.)),
        StructureNode('right', 'yellow_cube', (1., 0.)),
        StructureNode('beam', 'green_thin_prism', (0., 0.), ('left', 'right'), 'flat_x'),
        StructureNode('cap', 'red_cylinder', (0., 0.), ('beam',)),
    ),
    'cube_bridge': (
        StructureNode('left', 'blue_cube', (-.55, 0.)),
        StructureNode('right', 'red_cube', (.55, 0.)),
        StructureNode('span', 'green_cube', (0., 0.), ('left', 'right')),
    ),
    'capped_bridge': (
        StructureNode('left', 'blue_cube', (-.55, 0.)),
        StructureNode('right', 'yellow_cube', (.55, 0.)),
        StructureNode('span', 'green_cube', (0., 0.), ('left', 'right')),
        StructureNode('cap', 'red_cylinder', (0., 0.), ('span',)),
    ),
    'staircase': (
        StructureNode('low', 'blue_cube', (-1.8, 0.)),
        StructureNode('middle_base', 'red_cube', (0., 0.)),
        StructureNode('high_base', 'green_prism', (1.8, 0.)),
        StructureNode('middle_top', 'yellow_cube', (0., 0.), ('middle_base',)),
        StructureNode('high_top', 'green_cube', (1.8, 0.), ('high_base',)),
    ),
}


def structure_recipe(name='cube_bridge', *, color_map=None):
    if name not in STRUCTURE_RECIPES:
        raise ValueError('Unknown structure recipe: '+name)
    nodes = STRUCTURE_RECIPES[name]
    if color_map is not None:
        colors = {'red', 'yellow', 'blue', 'green'}
        if not isinstance(color_map, dict) or set(color_map) != colors or set(color_map.values()) != colors:
            raise ValueError('Color mapping must be a four-color bijection')
        nodes = tuple(StructureNode(n.key, color_map[n.object_name.split('_', 1)[0]]+'_'+n.object_name.split('_', 1)[1],
                                    n.grid, n.supports, n.orientation) for n in nodes)
    return StructureRecipe(name, tuple(nodes), nodes[0].key)
