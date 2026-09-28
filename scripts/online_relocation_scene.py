"""Four equal cubes: relocate the upper column without moving either foundation."""
from dataclasses import dataclass
import math
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from aspire_so100.inventory_structure import InventoryStructureEnv
from aspire_so100.structure_tasks import StructureNode
from online_bridge_scene import OnlineRecipe, scene_catalog
import online_bridge_scene as shared_scene
from online_contact_scene import ContactCompleteEnv
from run_three_cube_trial import apply_physics


@dataclass(frozen=True)
class RelocationRecipe(OnlineRecipe):
    def public_goal(self):
        goal = super().public_goal()
        names = {n.key: n.object_name.split('_', 1)[0] for n in self.nodes}
        for node in goal['nodes']:
            node.pop('grid', None)
        goal.update(kind='tower_relocation', roles=names,
            objective='Move the middle and upper cubes onto destination, preserving their order. '
                      'Leave the source and destination foundations in place.',
            assumptions='All four rigid cubes have equal size. Destination initially rests on the table. '
                        'The shared local skills measure its size from RGB-D; no numeric size is supplied.',
            foundation_translation_tolerance_m=.005,
            foundation_constraint='At success, both foundations must be within 5 mm of their initial positions throughout the stability hold; accidental displacement may be repaired.')
        return goal


def recipe(seed):
    colors = np.random.default_rng(seed).permutation(('red','yellow','blue','green'))
    names = [str(c)+'_cube' for c in colors]
    return RelocationRecipe('online_bridge_tower', (
        StructureNode('source', names[0], (-2.,0.)),
        StructureNode('destination', names[1], (2.,0.)),
        StructureNode('middle', names[2], (2.,0.), ('destination',)),
        StructureNode('upper', names[3], (2.,0.), ('middle',))), 'destination')


class RelocationEnv(ContactCompleteEnv):
    def __init__(self, output_dir, *, scene_seed):
        self.reset_count = 0
        self.episode_started = self.deadline = None
        self.realtime = True
        self.max_simulation_lag_s = 0.
        self.total_steps = 0
        self.stable_since = self.stable_positions = None
        self.success_time = self.success_evidence = None
        self.last_success_check = -math.inf
        self.anchor_positions = None
        self.anchor_violation = False
        self.monitor_rows = []
        with patch.object(shared_scene, 'recipe', recipe), scene_catalog(scene_seed):
            InventoryStructureEnv.__init__(self, output_dir, scene_seed=scene_seed,
                                           object_count=4, recipe='online_bridge_tower')
        self.physics = apply_physics(self.model, {'noslip_iterations':0})
        self.roles = {n.key:n.object_name for n in self.recipe.nodes}
        rng = np.random.default_rng(scene_seed)
        # Scene construction changes only the initial keyframe, before the sole reset.
        source_xy = np.array([-.100, -.315]) + rng.uniform(-.01,.01,2)
        dest_xy = np.array([.080, -.265]) + rng.uniform(-.01,.01,2)
        for role, name in self.roles.items():
            address = int(self.model.joint(f'{name}/{name}_joint').qposadr[0])
            size = float(self.model.geom(f'{name}/{name}_geom').size[2])
            level = {'source':0,'middle':1,'upper':2,'destination':0}[role]
            xy = dest_xy if role == 'destination' else source_xy
            yaw = float(rng.uniform(-.15,.15))
            self.model.key_qpos[0,address:address+7] = [*xy,.4+(2*level+1)*size,
                                                       math.cos(yaw/2),0,0,math.sin(yaw/2)]
        self.scene_metadata.update(generator='equal_cube_relocation_v1', goal=self.recipe.public_goal())

    def reset(self, seed, layout='sampled'):
        result = super().reset(seed, layout)
        self.anchor_positions = {r:self._object_state(self.roles[r])['position'].copy()
                                 for r in ('source','destination')}
        return result

    def structure_status(self, names):
        if list(names) != list(self.order):
            raise ValueError('Relocation requires the complete goal')
        states = {n:self._object_state(n) for n in self.order}
        reasons = []
        for name, state in states.items():
            if np.linalg.norm(state['linear_velocity']) > .015 or np.linalg.norm(state['angular_velocity']) > .10:
                reasons.append(name+': moving')
            if self.support_frame(name)[0][2] < math.cos(math.radians(8)):
                reasons.append(name+': not upright')
            if self._objects_in_contact(name, 'robot'):
                reasons.append(name+': robot contact')
        by_role = self.roles
        if self.anchor_positions is not None:
            self.anchor_violation = any(np.linalg.norm(states[by_role[r]]['position']-p) > .005
                                        for r,p in self.anchor_positions.items())
        if self.anchor_violation:
            reasons.append('A foundation is beyond the allowed 5 mm from its initial position')
        for role in ('source','destination'):
            if not self._objects_in_contact(by_role[role],'table'):
                reasons.append(role+': missing table support')
        permitted = {frozenset((by_role[a],by_role[b]))
                     for a,b in (('destination','middle'),('middle','upper'))}
        for index, first in enumerate(self.order):
            for second in self.order[index+1:]:
                if self._objects_in_contact(first,second) and frozenset((first,second)) not in permitted:
                    reasons.append('Unintended object contact')
        column = [by_role[r] for r in ('destination','middle','upper')]
        for index,(lower,upper) in enumerate(zip(column,column[1:])):
            if not self._objects_in_contact(lower,upper):
                reasons.append(upper+': missing support')
            if self._objects_in_contact(upper,'table'):
                reasons.append(upper+': touches table')
            if abs(self.geometry_bounds(upper)[0][2]-self.geometry_bounds(lower)[1][2]) > .004:
                reasons.append(upper+': incorrect support height')
            bodies = [self.model.body(f'{n}/{n}').id for n in column[index+1:]]
            com = np.average(self.data.xipos[bodies],axis=0,weights=self.model.body_mass[bodies])
            normal,top,_ = self.support_frame(lower)
            projected = com.copy()
            projected[2] = top[2]-np.dot(normal[:2],com[:2]-top[:2])/max(normal[2],.01)
            if self.support_margin(lower,projected) < .003:
                reasons.append(lower+': insufficient support margin')
        return {'stable':not reasons,'reasons':reasons}

    def _monitor_success(self):
        previous = self.last_success_check
        super()._monitor_success()
        if self.last_success_check != previous:
            self.monitor_rows.append({'time':float(self.data.time),'stable_since':self.stable_since,
                                      'anchor_violation':self.anchor_violation})

    def archive_terminal_physics(self, output):
        super().archive_terminal_physics(output)
        (Path(output)/'private_monitor.json').write_text(json.dumps(self.monitor_rows)+'\n')
        (Path(output)/'private_initial_anchors.json').write_text(json.dumps(
            {k:v.tolist() for k,v in self.anchor_positions.items()})+'\n')
