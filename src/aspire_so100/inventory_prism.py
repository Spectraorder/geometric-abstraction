"""Private upright-stack evaluation for task-conditioned inventory samples."""
import mujoco

from aspire_so100.block_catalog import generate_block_layout
from aspire_so100.prism import PrismEnv
from aspire_so100.stacking_env import StackingEnv


class InventoryPrismEnv(PrismEnv):
    task_id = 'inventory_prism'
    task_description = 'place red slender prism upright on blue cube'

    def __init__(self, output_dir, render=False, *, scene_seed=51, object_count=6):
        layout = generate_block_layout(scene_seed, object_count, 'upright_thin_prism')
        self.scene_seed = scene_seed
        self.objects = tuple(layout.positions)
        self.order = (layout.roles['base'], layout.roles['target'])
        self.scene_metadata = {'generator': 'block_catalog_v1', 'scene_task': layout.task,
                               'seed': scene_seed, 'object_count': object_count,
                               'objects': list(self.objects), 'roles': dict(layout.roles)}
        StackingEnv.__init__(self, output_dir, render, scene_task=layout.task,
                             scene_seed=scene_seed, object_count=object_count)
        self.checks = []
        self.failure = None

    def reset(self, seed, layout='sampled'):
        if layout != 'sampled':
            raise ValueError('Inventory tasks require layout=sampled')
        if seed != self.scene_seed:
            raise ValueError('Construct a new inventory environment for a different seed')
        self.seed = seed
        self.traces = []
        self._frame_index = 0
        self.checks = []
        self.failure = None
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self._step_for(.65)
        return self.observation()
