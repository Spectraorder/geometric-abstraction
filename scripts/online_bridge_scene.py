"""Five-object bridge-tower development scene with a single episode reset.

All catalog changes are process-local and scoped to scene construction. Runtime
sensor consumers receive calibrated frames and public identities, never state.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import math
import time
from unittest.mock import patch
import numpy as np

from aspire_so100 import block_catalog as catalog
from aspire_so100 import inventory_structure, structure_tasks, stacking_env
from aspire_so100.structure_tasks import StructureNode, StructureRecipe
from run_three_cube_trial import apply_physics
from run_unseeded_vision_trial import declarative_goal

COLORS = ('red', 'yellow', 'blue', 'green', 'magenta')
MAGENTA = (.85, .08, .85, 1.)
NAME = 'online_bridge_tower'


class EpisodeSuccess(Exception):
    """Private termination after the common sustained-success criterion."""


@dataclass(frozen=True)
class OnlineRecipe(StructureRecipe):
    def public_goal(self):
        return declarative_goal(self)


def recipe(seed):
    colors = list(np.random.default_rng(seed).permutation(COLORS))
    objects = [colors[i]+('_thin_prism' if i == 2 else '_cube') for i in range(5)]
    nodes = (StructureNode('left', objects[0], (-.8,0.)),
             StructureNode('right', objects[1], (.8,0.)),
             StructureNode('beam', objects[2], (0.,0.), ('left','right'), 'flat_x'),
             StructureNode('lower', objects[3], (0.,0.), ('beam',)),
             StructureNode('upper', objects[4], (0.,0.), ('lower',)))
    return OnlineRecipe(NAME,nodes,'left')


@contextmanager
def scene_catalog(seed):
    scene = stacking_env._SCENE_MODULE
    colors = dict(catalog.BLOCK_COLORS,magenta=MAGENTA)
    objects = tuple(f'{color}_{shape}' for color in colors for shape in catalog.SHAPE_DIMENSIONS)
    positions = {f'{color}_{shape}':(-.24+.12*column,-.22-.10*row,.40+dimensions[2]/2)
                 for row,color in enumerate(colors)
                 for column,(shape,dimensions) in enumerate(catalog.SHAPE_DIMENSIONS.items())}
    with patch.multiple(catalog,BLOCK_COLORS=colors,CATALOG_OBJECTS=objects,CATALOG_POSITIONS=positions), \
         patch.object(structure_tasks,'CATALOG_OBJECTS',objects), \
         patch.multiple(scene,BLOCK_COLORS=colors,CATALOG_POSITIONS=positions):
        cfg=deepcopy(scene.scene_cfg);cfg.entities=scene.make_catalog_entities()
        task=recipe(seed)
        def factory(name=NAME,*,color_map=None):
            if name != NAME or color_map is not None:
                raise ValueError('Online scene uses its seeded five-color task')
            return task
        with patch.object(scene,'scene_cfg',cfg),patch.object(inventory_structure,'structure_recipe',factory):
            yield task


class OnlineBridgeTowerEnv(inventory_structure.InventoryStructureEnv):
    """A persistent environment; retrying actions never permits another reset."""
    def __init__(self, output_dir, *, scene_seed):
        self.reset_count=0
        self.episode_started=None
        self.deadline=None
        self.realtime=True
        self.max_simulation_lag_s=0.
        self.total_steps=0
        self.stable_since=None
        self.stable_positions=None
        self.success_time=None
        self.success_evidence=None
        self.last_success_check=-math.inf
        with scene_catalog(scene_seed):
            super().__init__(output_dir,scene_seed=scene_seed,object_count=5,recipe=NAME)
        self.physics=apply_physics(self.model,{'noslip_iterations':0})
        self.scene_metadata.update(generator='online_five_color_v1',color_palette=COLORS)

    def reset(self, seed, layout='sampled'):
        if self.reset_count:
            raise RuntimeError('Online episodes cannot reset the environment')
        self.reset_count+=1
        return super().reset(seed,layout)

    def start_episode(self,budget_s=300.,*,realtime=True):
        if self.reset_count!=1 or self.episode_started is not None:
            raise RuntimeError('Start exactly once after the initial reset')
        if not math.isfinite(budget_s) or budget_s<=0:
            raise ValueError('Budget must be finite and positive')
        self.episode_started=time.monotonic()
        self.deadline=self.episode_started+budget_s
        self.initial_sim_time=float(self.data.time)
        self.initial_step_count=self.total_steps
        self.realtime=realtime

    def _step(self):
        if self.deadline is not None and time.monotonic()>=self.deadline:
            raise TimeoutError('Online episode wall-clock budget exhausted')
        super()._step()
        self.total_steps+=1
        if self.episode_started is not None:
            target=self.episode_started+float(self.data.time)-self.initial_sim_time
            now=time.monotonic()
            self.max_simulation_lag_s=max(self.max_simulation_lag_s,now-target)
            if self.realtime and target>now:
                time.sleep(min(target-now,max(0.,self.deadline-now)))
            self._monitor_success()
            if self.success_time is not None:
                raise EpisodeSuccess('Full structure remained stable for five seconds')

    def _monitor_success(self):
        # Private termination evidence never enters the agent observation packet.
        now=float(self.data.time)
        if self.success_time is not None or now-self.last_success_check<.05:
            return
        self.last_success_check=now
        stable=self.structure_status(list(self.order))['stable']
        positions={name:self._object_state(name)['position'].copy() for name in self.order}
        if stable and self.stable_positions is not None:
            stable=all(np.linalg.norm(positions[name]-self.stable_positions[name])<=.003
                       for name in self.order)
        if not stable:
            self.stable_since=None
            self.stable_positions=None
        elif self.stable_since is None:
            self.stable_since=now
            self.stable_positions=positions
        elif now-self.stable_since>=5.:
            self.success_time=time.monotonic()-self.episode_started
            self.success_evidence={'stable_start_sim_time':self.stable_since,
                                   'stable_end_sim_time':now,
                                   'duration_sim_seconds':now-self.stable_since,
                                   'maximum_final_displacement_m':max(
                                       float(np.linalg.norm(positions[name]-self.stable_positions[name]))
                                       for name in self.order),
                                   'drift_threshold_m':.003}

    def advance_while_waiting(self):
        """Called by the simulator owner while a model request is pending."""
        if self.episode_started is None:
            raise RuntimeError('Episode clock has not started')
        if time.monotonic()>=self.deadline:
            raise TimeoutError('Online episode wall-clock budget exhausted')
        target=self.initial_sim_time+time.monotonic()-self.episode_started
        while float(self.data.time)<target:
            self._step()
        if self.realtime:
            time.sleep(min(.005,max(0.,self.deadline-time.monotonic())))

    def timing(self):
        return {'reset_count':self.reset_count,'physics_steps':self.total_steps-self.initial_step_count,
                'wall_seconds':time.monotonic()-self.episode_started,
                'simulated_seconds':float(self.data.time)-self.initial_sim_time,
                'max_simulation_lag_seconds':self.max_simulation_lag_s,
                'physics_during_model_wait':True,'paced_actions':self.realtime}
