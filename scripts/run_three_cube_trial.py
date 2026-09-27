"""Three-cube column as a declarative support graph with explicit shared physics."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import run_vision_trial as trial
from aspire_so100 import inventory_structure
from aspire_so100.structure_tasks import StructureNode
from aspire_so100.tower_tasks import RECIPES
from run_unseeded_vision_trial import CONTRACT, declarative_goal

RECIPE = 'cube_triple'
PHYSICS = {'noslip_iterations': 0, 'collision_pipeline': 'libccd',
           'disableflags': int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)}
APPLY_CONTACT_SETTINGS = trial.apply_contact_settings


@dataclass(frozen=True)
class ThreeCubeRecipe:
    name: str
    nodes: tuple
    unit_node: str

    @property
    def order(self):
        return tuple(node.object_name for node in self.nodes)

    def public_goal(self):
        return declarative_goal(self)


def cube_recipe(name=RECIPE, *, color_map=None):
    if name != RECIPE:
        raise ValueError('This registered task requires cube_triple')
    colors = {'red', 'green', 'blue', 'yellow'}
    mapping = {color: color for color in colors} if color_map is None else color_map
    if not isinstance(mapping, dict) or set(mapping) != colors or set(mapping.values()) != colors:
        raise ValueError('Color mapping must be a four-color bijection')
    objects = tuple(mapping[name.split('_')[0]]+'_cube' for name in RECIPES[RECIPE])
    nodes = tuple(StructureNode('level_'+str(i), name, (0., 0.),
                               () if i == 0 else ('level_'+str(i-1),))
                  for i, name in enumerate(objects))
    return ThreeCubeRecipe(RECIPE, nodes, nodes[0].key)


def apply_physics(model, settings):
    if settings != {'noslip_iterations': 0}:
        raise ValueError('Unregistered contact settings')
    result = APPLY_CONTACT_SETTINGS(model, settings)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    result.update(collision_pipeline='libccd', disableflags=int(model.opt.disableflags))
    if result != PHYSICS:
        raise ValueError('Compiled physics differs from the registered variant')
    return result


def main():
    with patch.object(trial, 'STRUCTURE_RECIPES', {RECIPE: RECIPES[RECIPE]}), \
         patch.object(trial, 'structure_recipe', cube_recipe), \
         patch.object(inventory_structure, 'structure_recipe', cube_recipe), \
         patch.object(trial, 'apply_contact_settings', apply_physics):
        trial.main()
    output = Path(sys.argv[sys.argv.index('--output')+1])
    receipt = {'contract': CONTRACT, 'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'goal': json.loads((output/'result.json').read_text())['goal']}
    (output/'goal_contract.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
