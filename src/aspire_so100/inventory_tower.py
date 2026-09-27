"""Private evaluation of ordered, upright columns with shape-aware support."""
import math

import mujoco
import numpy as np

from aspire_so100.block_catalog import generate_block_layout
from aspire_so100.inventory_prism import InventoryPrismEnv
from aspire_so100.stacking_env import StackingEnv
from aspire_so100.tower import TowerMetrics
from aspire_so100.tower_tasks import TowerRecipe, tower_recipe


class InventoryTowerEnv(InventoryPrismEnv):
    task_id = 'inventory_tower'
    pixel_aligned_rgbd = True

    def __init__(self, output_dir, render=False, *, scene_seed=51, object_count=6,
                 recipe='cube_triple', levels=None, recipe_order=None):
        self.recipe = (TowerRecipe(recipe, tuple(recipe_order)) if recipe_order is not None
                       else tower_recipe(recipe, seed=scene_seed, levels=levels))
        self.order = self.recipe.order
        self.task_description = 'build an upright column, bottom to top: '+', '.join(self.order)
        layout = generate_block_layout(scene_seed, object_count, 'single_column', required_objects=self.order)
        self.scene_seed = scene_seed
        self.objects = tuple(layout.positions)
        self.scene_metadata = {'generator': 'block_catalog_v1', 'scene_task': layout.task,
                               'seed': scene_seed, 'object_count': object_count,
                               'objects': list(self.objects), 'roles': dict(layout.roles),
                               'recipe': recipe, 'order': list(self.order)}
        StackingEnv.__init__(self, output_dir, render, scene_task=layout.task, scene_seed=scene_seed,
                             object_count=object_count, required_objects=self.order)
        self.checks = []
        self.failure = None

    def geometry_bounds(self, name):
        geom = self.model.geom(f'{name}/{name}_geom')
        rotation = self.data.geom_xmat[geom.id].reshape(3, 3)
        center = self.data.geom_xpos[geom.id]
        if geom.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent = geom.size[0]*np.sqrt(rotation[:, 0]**2+rotation[:, 1]**2) + geom.size[1]*abs(rotation[:, 2])
        elif geom.type[0] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(geom.dataid[0])
            start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            vertices = self.model.mesh_vert[start:start+count] @ rotation.T + center
            return vertices.min(axis=0), vertices.max(axis=0)
        else:
            extent = abs(rotation) @ geom.size
        return center-extent, center+extent

    def support_frame(self, name):
        """Private support plane; all faces of an unmarked cube are equivalent."""
        geom = self.model.geom(f'{name}/{name}_geom')
        rotation = self.data.geom_xmat[geom.id].reshape(3, 3)
        cube = name.endswith('_cube') and geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX
        axis = int(np.argmax(abs(rotation[2]))) if cube else 2
        normal = rotation[:, axis].copy()
        if cube and normal[2] < 0:
            normal *= -1
        half_height = geom.size[1] if geom.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER else geom.size[axis]
        top = self.data.geom_xpos[geom.id] + normal * half_height
        return normal, top, [i for i in range(3) if i != axis]

    def support_margin(self, name, point):
        geom = self.model.geom(f'{name}/{name}_geom')
        rotation = self.data.geom_xmat[geom.id].reshape(3, 3)
        local = rotation.T @ (np.asarray(point)-self.data.geom_xpos[geom.id])
        if geom.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return float(geom.size[0]-np.linalg.norm(local[:2]))
        if geom.type[0] == mujoco.mjtGeom.mjGEOM_MESH:
            return -np.inf  # The upright wedge has no horizontal support face.
        horizontal = self.support_frame(name)[2]
        return float(np.min(geom.size[horizontal]-abs(local[horizontal])))

    def structure_status(self, names):
        names = list(names)
        if not names or names != list(self.order[:len(names)]):
            raise ValueError('Structure checks require a nonempty recipe prefix')
        states = {name: self._object_state(name) for name in names}
        reasons = []
        for name, state in states.items():
            if np.linalg.norm(state['linear_velocity']) > .015:
                reasons.append(name+': translating')
            if np.linalg.norm(state['angular_velocity']) > .10:
                reasons.append(name+': rotating')
            up = self.support_frame(name)[0]
            if up[2] < math.cos(math.radians(8)):
                reason = 'no cube face is upright' if name.endswith('_cube') else 'local Z axis is not upright'
                reasons.append(name+': '+reason)
            if self._objects_in_contact(name, 'robot'):
                reasons.append(name+': still touching robot')
            for other in self.objects:
                if other not in names and self._objects_in_contact(name, other):
                    reasons.append(name+': touching an object outside the checked column')
        if not self._objects_in_contact(names[0], 'table'):
            reasons.append('base: missing table support')
        for i, (lower, upper) in enumerate(zip(names, names[1:])):
            if not self._objects_in_contact(lower, upper):
                reasons.append(upper+': missing contact with '+lower)
            if abs(self.geometry_bounds(upper)[0][2]-self.geometry_bounds(lower)[1][2]) > .004:
                reasons.append(upper+': incorrect support height')
            bodies = [self.model.body(f'{n}/{n}').id for n in names[i+1:]]
            com = np.average(self.data.xipos[bodies], axis=0, weights=self.model.body_mass[bodies])
            # Project the upper substack COM onto the lower object's top plane.
            up, top, _ = self.support_frame(lower)
            if abs(up[2]) < .5:
                reasons.append(lower+': support face is not horizontal')
                continue
            com = com.copy()
            com[2] = top[2] - np.dot(up[:2], com[:2]-top[:2])/up[2]
            if self.support_margin(lower, com) < .003:
                reasons.append(lower+': insufficient upper-substack support margin')
        return {'stable': not reasons, 'reasons': reasons}

    def evaluate(self):
        placed = 0
        for length in range(2, len(self.order)+1):
            if self.structure_status(list(self.order[:length]))['stable']:
                placed = length-1
        # Once built, upper levels contact lower prefixes. Score the full chain
        # independently so these required contacts cannot invalidate success.
        complete = self.structure_status(list(self.order))['stable']
        if complete:
            placed = len(self.order)-1
        final = self.checks[-1] if self.checks else {}
        stable = bool(complete and final.get('stable') and final.get('blocks') == list(self.order)
                      and final.get('duration_s', 0) >= 5.)
        required = len(self.order)-1
        return TowerMetrics(stable and self.failure is None, placed/required, placed, required, stable, self.failure)
