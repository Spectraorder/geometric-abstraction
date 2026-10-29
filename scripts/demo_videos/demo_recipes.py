"""Demo-only structure recipes registered at runtime; repo sources are unchanged."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]

from aspire_so100.structure_tasks import STRUCTURE_RECIPES, StructureNode  # noqa: E402

# Bridge-tower: a thin beam across two cube foundations with a two-cube column on the beam.
STRUCTURE_RECIPES.setdefault('bridge_tower', (
    StructureNode('left', 'blue_cube', (-1., 0.)),
    StructureNode('right', 'yellow_cube', (1., 0.)),
    StructureNode('beam', 'green_thin_prism', (0., 0.), ('left', 'right'), 'flat_x'),
    StructureNode('middle', 'red_cube', (0., 0.), ('beam',)),
    StructureNode('top', 'green_cube', (0., 0.), ('middle',)),
))
