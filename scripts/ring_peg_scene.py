"""Hexagonal ring over a fixed vertical peg, in the shared online robot/table scene.

The robot, table, cameras, contact settings (no-slip 0, libccd) and body-ancestry
contact grouping match the online bridge studies; only the objects are replaced.
The ring is six convex trapezoidal prisms, so MuJoCo keeps the hexagonal hole
without a mesh convex decomposition. Object poses, sizes and contacts are private
evaluator data. This is a development scene; no study is registered here.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import mujoco
import numpy as np

from aspire_so100.simulation_settings import apply_contact_settings
from aspire_so100.stacking_env import StackingEnv, TABLE_HEIGHT, _SCENE_MODULE as scene_module
from online_bridge_scene import COLORS, MAGENTA
from aspire_so100.block_catalog import BLOCK_COLORS

RING, PEG = 'ring', 'peg'
PALETTE = dict(BLOCK_COLORS, magenta=MAGENTA)


@dataclass(frozen=True)
class RingPegGeometry:
    # Hexagon inradii (center to flat), meters. Wall = outer - hole inradius.
    outer_inradius: float = .036
    hole_inradius: float = .020
    height: float = .015
    ring_mass: float = .04
    peg_radius: float = .008
    peg_height: float = .050

    def __post_init__(self):
        if not 0 < self.peg_radius < self.hole_inradius < self.outer_inradius:
            raise ValueError('Require peg radius < hole inradius < outer inradius')
        if self.height <= 0 or self.peg_height <= 0 or self.ring_mass <= 0:
            raise ValueError('Dimensions and mass must be positive')

    def hexagon(self, inradius):
        # Vertices at 30 + 60k degrees, so flats face +-X at yaw zero.
        radius = inradius/math.cos(math.pi/6)
        angles = math.pi/6+np.arange(6)*math.pi/3
        return radius*np.column_stack((np.cos(angles), np.sin(angles)))

    def segments(self):
        """Six convex trapezoidal prisms (8 vertices each) in the ring body frame."""
        outer, inner = self.hexagon(self.outer_inradius), self.hexagon(self.hole_inradius)
        half = self.height/2
        pieces = []
        for i in range(6):
            j = (i+1) % 6
            quad = np.array([inner[i], outer[i], outer[j], inner[j]])
            pieces.append(np.vstack((np.column_stack((quad, np.full(4, -half))),
                                     np.column_stack((quad, np.full(4, half))))))
        return pieces


def ring_spec(geometry, color):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=RING)
    body.add_freejoint(name=RING+'_joint')
    pieces = geometry.segments()
    for index, vertices in enumerate(pieces):
        spec.add_mesh(name=f'segment_{index}', uservert=vertices.ravel().tolist())
        body.add_geom(name=f'{RING}_geom_{index}', type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=f'segment_{index}', mass=geometry.ring_mass/len(pieces),
                      rgba=color, friction=(1., .005, .0001))
    return spec


def peg_spec(geometry, color):
    # No joint: the entity is welded to the world at its initial position.
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=PEG)
    body.add_geom(name=PEG+'_geom', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=(geometry.peg_radius, geometry.peg_height/2), rgba=color,
                  friction=(1., .005, .0001))
    return spec


def sample_layout(seed, geometry):
    """Seeded colors, peg site and ring pose; both objects start in the central workspace."""
    rng = np.random.default_rng(seed)
    ring_color, peg_color = (str(c) for c in rng.choice(COLORS, 2, replace=False))
    outer = geometry.outer_inradius/math.cos(math.pi/6)
    for _ in range(100):
        # Both regions stay clear of the hanging home-pose jaws near (0, -0.24).
        peg = np.array([rng.uniform(.06, .11), rng.uniform(-.34, -.26)])
        ring = np.array([rng.uniform(-.13, -.07), rng.uniform(-.32, -.23)])
        if np.linalg.norm(peg-ring) > outer+geometry.peg_radius+.03:
            break
    else:
        raise RuntimeError('Could not separate ring and peg')
    return {'seed': int(seed), 'ring_color': ring_color, 'peg_color': peg_color,
            'peg_xy': peg.tolist(), 'ring_xy': ring.tolist(),
            'ring_yaw': float(rng.uniform(0., math.pi/3))}


class RingPegEnv(StackingEnv):
    """One ring and one fixed peg; the rest of the scene is the shared online scene."""
    ARM_JOINTS = StackingEnv.ARM_JOINTS

    def __init__(self, output_dir, *, scene_seed, geometry=None, render=False):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.render_enabled = render
        self.geometry = geometry or RingPegGeometry()
        self.layout = sample_layout(scene_seed, self.geometry)
        g, layout = self.geometry, self.layout
        cfg = deepcopy(scene_module.scene_cfg)
        cfg.entities = {
            'robot': deepcopy(scene_module.robot_cfg), 'table': deepcopy(scene_module.table_cfg),
            RING: scene_module.make_shape_cfg(lambda: ring_spec(g, PALETTE[layout['ring_color']]),
                                              (*layout['ring_xy'], TABLE_HEIGHT+g.height/2)),
            PEG: scene_module.make_shape_cfg(lambda: peg_spec(g, PALETTE[layout['peg_color']]),
                                             (*layout['peg_xy'], TABLE_HEIGHT+g.peg_height/2)),
        }
        self.scene = scene_module.Scene(cfg, device='cpu')
        self.model = self.scene.compile()
        self.physics = apply_contact_settings(self.model, {'noslip_iterations': 0})
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        self.physics.update(collision_pipeline='libccd', disableflags=int(self.model.opt.disableflags))
        self.data = mujoco.MjData(self.model)
        self.site_id = self.model.site('robot/ee_site').id
        self.arm_joints = [self.model.joint('robot/'+name) for name in self.ARM_JOINTS]
        self.arm_qpos_addresses = np.array([int(j.qposadr[0]) for j in self.arm_joints])
        self.arm_dof_addresses = np.array([int(j.dofadr[0]) for j in self.arm_joints])
        self._renderer = None
        self._overhead_rgbd = None
        self.pixel_aligned_rgbd = True
        self.seed = scene_seed
        self.traces = []
        self._frame_index = 0
        self.scene_metadata = dict(generator='ring_peg_v1', **layout,
                                   geometry={k: getattr(g, k) for k in g.__dataclass_fields__})

    def reset(self, seed=None):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._set_freejoint_pose(RING, (*self.layout['ring_xy'], TABLE_HEIGHT+self.geometry.height/2),
                                 self.layout['ring_yaw'])
        mujoco.mj_forward(self.model, self.data)
        self._step_for(.5)
        return self.observation()

    def observation(self):
        # Truth for tracing stays private; public observations come from sensors.
        return {'time': float(self.data.time)}

    # Same body-ancestry grouping as ContactCompleteEnv, so unnamed robot meshes count.
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

    def private_ring_state(self):
        state = self._object_state(RING)
        body = self.model.body(RING+'/'+RING).id
        rotation = self.data.xmat[body].reshape(3, 3)
        return dict(state, rotation=rotation.copy(), tilt_deg=math.degrees(math.acos(np.clip(rotation[2, 2], -1, 1))))

    def peg_axis_xy(self):
        return np.asarray(self.data.xpos[self.model.body(PEG+'/'+PEG).id][:2]).copy()

    def ring_status(self):
        """Private success check: the peg axis passes through the hexagonal hole."""
        g = self.geometry
        state = self.private_ring_state()
        rotation, center = state['rotation'], state['position']
        # Where the peg axis meets the ring mid-plane, in ring coordinates.
        axis_xy = self.peg_axis_xy()
        normal = rotation[:, 2]
        reasons = []
        if abs(normal[2]) < 1e-3:
            reasons.append('ring is vertical')
            local = np.full(3, np.inf)
        else:
            z = center[2]+(normal[:2]@(center[:2]-axis_xy))/normal[2]
            local = rotation.T@(np.r_[axis_xy, z]-center)
        # Hexagon inradius test on the three flat directions.
        flats = np.array([[math.cos(a), math.sin(a)] for a in np.arange(3)*math.pi/3])
        margin = g.hole_inradius-g.peg_radius-float(np.max(np.abs(flats@local[:2])))
        if margin < 0:
            reasons.append('peg axis is not inside the hole clearance')
        peg_top = TABLE_HEIGHT+g.peg_height
        if center[2] > peg_top:
            reasons.append('ring is above the peg top')
        if self._objects_in_contact(RING, 'robot'):
            reasons.append('ring touches robot')
        if np.linalg.norm(state['linear_velocity']) > .015 or np.linalg.norm(state['angular_velocity']) > .10:
            reasons.append('ring is moving')
        return {'on_peg': not reasons, 'reasons': reasons, 'clearance_margin_m': margin,
                'ring_center_m': center.tolist(), 'tilt_deg': state['tilt_deg'],
                'ring_table_contact': self._objects_in_contact(RING, 'table'),
                'ring_peg_contact': self._objects_in_contact(RING, PEG)}

    def evaluate_on_peg(self, settle_s=.8, hold_s=2., drift_m=.003):
        self._step_for(settle_s)
        start = self.private_ring_state()['position']
        checks = []
        for _ in range(int(round(hold_s/.1))):
            self._step_for(.1)
            status = self.ring_status()
            checks.append(status['on_peg'])
        final = self.ring_status()
        drift = float(np.linalg.norm(np.asarray(final['ring_center_m'])-start))
        return dict(final, success=bool(all(checks) and drift <= drift_m), drift_m=drift,
                    hold_s=hold_s, drift_threshold_m=drift_m)
