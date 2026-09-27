"""Complete private contact grouping and replayable terminal physics for new runs."""
from pathlib import Path
import mujoco
import numpy as np
from online_bridge_scene import OnlineBridgeTowerEnv


class ContactCompleteEnv(OnlineBridgeTowerEnv):
    """Include unnamed collision geoms by their owning body ancestry."""
    def _contact_group(self, name):
        if not hasattr(self, '_contact_groups'):
            self._contact_groups = {}
        if name not in self._contact_groups:
            def matches(value):
                return bool(value) and (value == name or value.startswith(name+'/'))
            mask = np.zeros(self.model.ngeom, dtype=bool)
            for geom_id in range(self.model.ngeom):
                mask[geom_id] = matches(self.model.geom(geom_id).name)
                body_id = int(self.model.geom_bodyid[geom_id])
                while body_id:
                    mask[geom_id] |= matches(self.model.body(body_id).name)
                    body_id = int(self.model.body_parentid[body_id])
            self._contact_groups[name] = mask
        return self._contact_groups[name]

    def _objects_in_contact(self, first, second):
        a, b = self._contact_group(first), self._contact_group(second)
        return any((a[c.geom1] and b[c.geom2]) or (b[c.geom1] and a[c.geom2])
                   for c in self.data.contact)

    def archive_terminal_physics(self, output):
        """Private evaluator artifact; never attached to a model request."""
        output = Path(output)
        mujoco.mj_saveModel(self.model, str(output/'private_model.mjb'))
        np.savez_compressed(output/'private_terminal_state.npz', time=float(self.data.time),
                            qpos=self.data.qpos, qvel=self.data.qvel, act=self.data.act,
                            ctrl=self.data.ctrl, qacc_warmstart=self.data.qacc_warmstart)
