"""Private physical evaluation of low-rise, multiple-support structures."""
import math
import numpy as np

from aspire_so100.block_catalog import generate_block_layout
from aspire_so100.inventory_tower import InventoryTowerEnv
from aspire_so100.stacking_env import StackingEnv
from aspire_so100.structure_geometry import support_equilibrium
from aspire_so100.structure_tasks import structure_recipe
from aspire_so100.tower import TowerMetrics


def configure_structure_initial_state(model, recipe):
    """Initialize flat beam sources on the table, before any policy runs.

    Only the reset keyframe is changed. No assembly pose or in-episode object
    motion is supplied. The original sampled XY locations remain unchanged.
    """
    for node in recipe.nodes:
        if node.orientation != 'flat_x':
            continue
        name = node.object_name
        geom = model.geom(f'{name}/{name}_geom')
        address = int(model.joint(f'{name}/{name}_joint').qposadr[0])
        if np.allclose(model.key_qpos[0, address+3:address+7], [math.sqrt(.5), 0., math.sqrt(.5), 0.]):
            continue
        model.key_qpos[0, address+2] += float(geom.size[0]-geom.size[2])
        # Local long Z points along world X; the broad YZ face rests on the table.
        model.key_qpos[0, address+3:address+7] = [math.sqrt(.5), 0., math.sqrt(.5), 0.]


class InventoryStructureEnv(InventoryTowerEnv):
    task_id = 'inventory_structure'

    def __init__(self, output_dir, render=False, *, scene_seed=51, object_count=6,
                 recipe='cube_bridge', color_map=None):
        self.recipe = structure_recipe(recipe, color_map=color_map)
        self.order = self.recipe.order
        self.task_description = 'build support graph: '+recipe
        layout = generate_block_layout(scene_seed, object_count, 'support_graph', required_objects=self.order)
        self.scene_seed = scene_seed
        self.objects = tuple(layout.positions)
        self.scene_metadata = {'generator': 'block_catalog_v1', 'scene_task': layout.task,
                               'seed': scene_seed, 'object_count': object_count,
                               'objects': list(self.objects), 'roles': dict(layout.roles),
                               'recipe': recipe, 'order': list(self.order), 'color_map': color_map,
                               'goal': self.recipe.public_goal()}
        StackingEnv.__init__(self, output_dir, render, scene_task=layout.task, scene_seed=scene_seed,
                             object_count=object_count, required_objects=self.order)
        configure_structure_initial_state(self.model, self.recipe)
        initial = {node.object_name: node.orientation for node in self.recipe.nodes
                   if node.orientation != 'upright'}
        if initial:
            self.scene_metadata['initial_orientations'] = initial
            self.scene_metadata['generator'] = 'block_catalog_flat_beam_v1'
        self.checks = []
        self.failure = None

    def private_model(self, name):
        geom = self.model.geom(f'{name}/{name}_geom')
        body = self.model.body(f'{name}/{name}')
        rotation = self.data.geom_xmat[geom.id].reshape(3, 3)
        center = self.data.geom_xpos[geom.id].copy()
        shape = name.split('_', 1)[1]
        size = geom.size
        node = next(node for node in self.recipe.nodes if node.object_name == name)
        # Unmarked cube faces are interchangeable, including in support graphs.
        vertical_axis = (int(np.argmax(abs(rotation[2]))) if shape == 'cube'
                         else 0 if node.orientation == 'flat_x' else 2)
        normal = rotation[:, vertical_axis].copy()
        if (shape == 'cube' or node.orientation == 'flat_x') and normal[2] < 0:
            normal *= -1
        horizontal_axes = [axis for axis in range(3) if axis != vertical_axis]
        dims = np.repeat(2*size[0], 2) if shape == 'cylinder' else 2*size[horizontal_axes]
        result = {'shape': shape, 'center': self.data.xipos[body.id].copy(),
                  'dims': dims, 'mass': float(body.mass[0]), 'normal': normal,
                  'height': float(2*size[1] if shape == 'cylinder' else 2*size[vertical_axis])}
        if shape != 'cylinder':
            signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
            vertices = center+(signs*size) @ rotation.T
            sign = 1. if rotation[:, vertical_axis] @ normal > 0 else -1.
            result['top_vertices'] = vertices[signs[:, vertical_axis] == sign]
            result['bottom_vertices'] = vertices[signs[:, vertical_axis] == -sign]
        return result

    def structure_status(self, names):
        names = list(names)
        if not names or names != list(self.order[:len(names)]):
            raise ValueError('Structure checks require a nonempty construction prefix')
        nodes = self.recipe.nodes[:len(names)]
        by_key = {n.key: n.object_name for n in nodes}
        states = {name: self._object_state(name) for name in names}
        models = {name: self.private_model(name) for name in names}
        reference = next(n for n in self.recipe.nodes if n.key == self.recipe.unit_node)
        unit_geom = self.model.geom(f'{reference.object_name}/{reference.object_name}_geom')
        unit = float(2*unit_geom.size[2])
        origins = np.array([states[n.object_name]['position'][:2]-unit*np.asarray(n.grid) for n in nodes])
        origin = np.mean(origins, axis=0)
        parents = {n.object_name: [by_key[k] for k in n.supports] for n in nodes}
        reasons = []
        for node in nodes:
            name = node.object_name
            state = states[name]
            if np.linalg.norm(origins[names.index(name)]-origin) > .006:
                reasons.append(name+': incorrect relative XY position')
            if np.linalg.norm(state['linear_velocity']) > .015:
                reasons.append(name+': translating')
            if np.linalg.norm(state['angular_velocity']) > .10:
                reasons.append(name+': rotating')
            rotation = self.data.xmat[self.model.body(f'{name}/{name}').id].reshape(3, 3)
            threshold = math.cos(math.radians(8))
            if node.orientation == 'flat_x':
                if abs(rotation[2, 0]) < threshold or abs(rotation[0, 2]) < threshold:
                    reasons.append(name+': broad face is not flat with the long axis along X')
            elif name.endswith('_cube'):
                if models[name]['normal'][2] < threshold:
                    reasons.append(name+': no cube face is upright')
            elif rotation[2, 2] < threshold:
                reasons.append(name+': local Z axis is not upright')
            if self._objects_in_contact(name, 'robot'):
                reasons.append(name+': still touching robot')
            if not node.supports:
                if not self._objects_in_contact(name, 'table'):
                    reasons.append(name+': missing table support')
            else:
                if self._objects_in_contact(name, 'table'):
                    reasons.append(name+': upper object touches table')
                for parent in parents[name]:
                    if not self._objects_in_contact(name, parent):
                        reasons.append(name+': missing required support '+parent)
                    if abs(self.geometry_bounds(name)[0][2]-self.geometry_bounds(parent)[1][2]) > .004:
                        reasons.append(name+': incorrect support height on '+parent)
            for other in self.objects:
                permitted = other in parents[name] or name in parents.get(other, [])
                if other != name and not permitted and self._objects_in_contact(name, other):
                    reasons.append(name+': unintended object contact with '+other)
        balance = support_equilibrium(models, parents, margin=.002)
        if not balance['feasible']:
            reasons.append('structure: '+balance['reason'])
        return {'stable': not reasons, 'reasons': reasons}

    def evaluate(self):
        complete = self.structure_status(list(self.order))['stable']
        placed = 0
        if complete:
            placed = len(self.order)
        else:
            for count in range(1, len(self.order)+1):
                if self.structure_status(list(self.order[:count]))['stable']:
                    placed = count
        last = self.checks[-1] if self.checks else {}
        stable = bool(complete and last.get('stable') and last.get('blocks') == list(self.order)
                      and last.get('duration_s', 0.) >= 5.)
        return TowerMetrics(stable and self.failure is None, placed/len(self.order), placed,
                            len(self.order), stable, self.failure)
