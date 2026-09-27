"""Run the shared trial with declarative goals that do not supply execution order."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import run_vision_trial as trial
from aspire_so100.structure_tasks import StructureRecipe

PUBLIC_GOAL = StructureRecipe.public_goal
CONTRACT = 'declarative_color_sorted_goal_v1'


def declarative_goal(recipe):
    goal = deepcopy(PUBLIC_GOAL(recipe))
    keys = {node['key']: node['color'] for node in goal['nodes']}
    if len(set(keys.values())) != len(keys):
        raise ValueError('This color-addressed interface requires unique object colors')
    goal['length_unit']['observed_cube'] = keys[goal['length_unit']['observed_cube']]
    for node in goal['nodes']:
        node['key'] = keys[node['key']]
        node['supports'] = [keys[key] for key in node['supports']]
    goal['nodes'].sort(key=lambda node: node['color'])
    goal['node_order'] = 'color_label_not_execution_order'
    return goal


if __name__ == '__main__':
    with patch.object(StructureRecipe, 'public_goal', declarative_goal):
        trial.main()
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    receipt = {'contract': CONTRACT,
               'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'goal': json.loads((output/'result.json').read_text())['goal']}
    (output/'goal_contract.json').write_text(json.dumps(receipt, indent=2) + '\n')
