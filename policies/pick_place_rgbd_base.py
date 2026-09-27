"""Reusable RGB-D pick-and-place base, extracted from the historical 4 cm policy.

No task goal, object order, tower height, cross-object drift compensation, or
simulator object state is used. The caller supplies a unique color and a world
XYZ point on a horizontal destination surface derived from its observations.
The default observer assumes an initially upright object with a visible flat top.
It is a simple surface estimate, not an explicit rigid-vertex abstraction.
"""
import numpy as np
from scipy import ndimage


class RGBDPickPlace:
    def __init__(self, api, grasp_fraction=0.30, release_clearance=0.003, position_tolerance=0.004):
        self.api = api
        self.models = {}
        self.held = None
        values = np.asarray([grasp_fraction, release_clearance, position_tolerance], float)
        if not np.isfinite(values).all() or not 0 <= grasp_fraction <= 1 or not 0 <= release_clearance <= .02 or not 0 < position_tolerance <= .02:
            raise ValueError('Invalid generic grasp/release parameters')
        self.grasp_fraction = float(grasp_fraction)
        self.release_clearance = float(release_clearance)
        self.position_tolerance = float(position_tolerance)
        rgb, xyz, valid = self.capture()
        p = xyz[self.masks(rgb)['table'] & valid]
        if len(p) < 100:
            raise RuntimeError('Insufficient tabletop depth.')
        p = p[::max(1, len(p) // 4000)]
        A = np.column_stack((p[:, :2], np.ones(len(p))))
        keep = np.abs(p[:, 2] - np.median(p[:, 2])) < 0.025
        for _ in range(4):
            if np.count_nonzero(keep) < 50:
                raise RuntimeError('Table fit unreliable.')
            self.plane = np.linalg.lstsq(A[keep], p[keep, 2], rcond=None)[0]
            keep = np.abs(p[:, 2] - A @ self.plane) < 0.003

    def tool(self):
        return np.asarray(self.api.proprioception()['tool_position'], dtype=float)

    def move(self, p, seconds=1.5):
        self.api.move_tool(np.asarray(p, dtype=float).tolist(), seconds=seconds)

    def capture(self):
        s = self.api.camera()
        rgb = np.asarray(s['rgb'], dtype=float)
        d = np.asarray(s['depth_m'], dtype=float)
        valid = np.asarray(s['valid_depth'], dtype=bool) & np.isfinite(d) & (d > 0)
        v, u = np.indices(d.shape)
        rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(np.asarray(s['intrinsics'])).T
        T = np.asarray(s['camera_to_world'])
        xyz = rays * d[..., None] @ T[:3, :3].T + T[:3, 3]
        return (rgb, xyz, valid)

    def masks(self, rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        return {
            'red': (r > 75) & (r > 1.65*g) & (r > 1.55*b),
            'green': (g > 65) & (g > 1.45*r) & (g > 1.3*b),
            'blue': (b > 65) & (b > 1.55*r) & (b > 1.2*g),
            'yellow': (r > 100) & (g > 90) & (r < 1.5*g) & (g > 2*b),
            'table': (r > 95) & (g > 65) & (r > 1.15*g) & (r < 1.65*g) & (g > 1.35*b),
        }

    def table_z(self, xy):
        return float(np.dot(self.plane[:2], xy) + self.plane[2])

    def locate(self, name, model=None, near=None):
        rgb, xyz, valid = self.capture()
        labels, count = ndimage.label(self.masks(rgb)[name] & valid)
        choices = []
        for k in range(1, count + 1):
            m = labels == k
            q = xyz[m]
            if len(q) < 8:
                continue
            h = q[:, 2] - q[:, :2] @ self.plane[:2] - self.plane[2]
            if np.median(h) < 0.004 or np.median(h) > 0.32:
                continue
            distance = 0.0 if near is None else np.linalg.norm(np.median(q, axis=0) - near)
            if distance > 0.12:
                continue
            choices.append((len(q) / (1 + 60 * distance), m))
        if not choices:
            raise RuntimeError('No visible ' + name + ' block.')
        m = max(choices, key=lambda item: item[0])[1]
        q = xyz[m]
        dx = xyz[1:-1, 2:] - xyz[1:-1, :-2]
        dy = xyz[2:, 1:-1] - xyz[:-2, 1:-1]
        n = np.cross(dx, dy)
        lengths = np.linalg.norm(n, axis=2)
        n /= np.maximum(lengths[..., None], 1e-12)
        n = np.where(n[..., 2:3] < 0, -n, n)
        interior = m[1:-1, 1:-1] & m[1:-1, 2:] & m[1:-1, :-2] & m[2:, 1:-1] & m[:-2, 1:-1]
        good = interior & (lengths > 1e-10) & (n[..., 2] > 0.78)
        seeds = xyz[1:-1, 1:-1][good]
        if len(seeds) >= 5:
            normal = np.median(n[good], axis=0)
            normal /= np.linalg.norm(normal)
            level = float(np.median(seeds @ normal))
            face = q[np.abs(q @ normal - level) < 0.0022]
            for _ in range(2):
                if len(face) < 6:
                    break
                mean = np.mean(face, axis=0)
                _, _, vh = np.linalg.svd(face - mean, full_matrices=False)
                fitted = vh[-1]
                if fitted[2] < 0:
                    fitted = -fitted
                if fitted[2] < 0.78:
                    break
                normal = fitted
                level = float(mean @ normal)
                face = q[np.abs(q @ normal - level) < 0.0022]
        else:
            z = np.percentile(q[:, 2], 90)
            face = q[np.abs(q[:, 2] - z) < 0.0025]
            normal = np.array([0.0, 0.0, 1.0])
            level = float(np.median(face[:, 2])) if len(face) else float(z)
        if len(face) < 5:
            raise RuntimeError('Top face occluded for ' + name)
        ex = np.array([1.0, 0.0, 0.0]) - normal * normal[0]
        ex /= np.linalg.norm(ex)
        ey = np.cross(normal, ex)
        uv = np.column_stack((face @ ex, face @ ey))
        lo, hi = np.percentile(uv, [2, 98], axis=0)
        middle = (lo + hi) / 2
        top_center = middle[0] * ex + middle[1] * ey + level * normal
        if model is None:
            height = float(top_center[2] - self.table_z(top_center[:2]))
            dims = hi - lo
            if not 0.005 < height < 0.15 or np.min(dims) < 0.004:
                raise RuntimeError('Unreliable dimensions for ' + name)
            model = {'height': height, 'dims': dims}
        height = model['height']
        center = top_center - normal * height / 2
        extent = float((abs(ex[2]) * model['dims'][0] + abs(ey[2]) * model['dims'][1] + normal[2] * height) / 2)
        return {'center': center, 'top': float(top_center[2]), 'normal': normal, 'bottom': float(center[2] - extent), 'extent_z': extent, 'height': height, 'dims': model['dims'], 'model': model, 'source': 'rgbd_visible_top_estimate', 'hidden_extent_inferred': True}

    def observe(self, color, near=None):
        """Cache dimensions only from the first upright on-table observation."""
        if color not in ('red', 'yellow', 'blue', 'green'):
            raise ValueError('Expected a supported unique color')
        measured = self.locate(color, self.models.get(color), near)
        self.models[color] = measured['model']
        return measured

    def pick(self, color, transit_z=None):
        """Pick one object; lifting is checked visually, not with a contact oracle."""
        if self.held is not None:
            raise RuntimeError('Finish or recover the held object before another pick')
        source = self.observe(color)
        park = self.tool().copy()
        high_z = max(park[2], source['top']+.12)
        if transit_z is not None:
            if not np.isfinite(transit_z):
                raise ValueError('Transit height must be finite')
            high_z = max(high_z, float(transit_z))
        park[2] = high_z
        grasp = source['center'].copy()
        grasp[2] += self.grasp_fraction*source['height']
        above = grasp.copy()
        above[2] = high_z
        self.api.gripper(.8, seconds=.6)
        self.move(park)
        self.move(above, 1.8)
        pre = grasp.copy()
        pre[2] += .025
        self.move(pre, 1.1)
        self.move(grasp, 1.2)
        self.api.gripper(-.174, seconds=1.)
        # A closed gripper is an uncertain held state until visual confirmation.
        self.held = {'color': color, 'source': source, 'park': park, 'confirmed': False}
        self.api.wait(.35)
        lift = grasp.copy()
        lift[2] += .06
        self.move(lift, 1.7)
        measured = self.observe(color, self.tool())
        if measured['center'][2]-source['center'][2] < .025:
            self.move(grasp)
            self.api.gripper(.8, seconds=.5)
            self.held = None
            self.move(above)
            self.move(park)
            return {'status': 'grasp_unconfirmed', 'held': False}
        self.held['confirmed'] = True
        return {'status': 'lift_visually_confirmed', 'held': True, 'observation': measured}

    def place(self, surface_point):
        """Place at caller-supplied XY and support Z; no support/goal inference.

        An uncertain final alignment leaves the gripper closed and returns an
        explicit status. The caller owns recovery and task-level decisions.
        """
        point = np.asarray(surface_point, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Destination must be finite world XYZ in meters')
        if self.held is None or not self.held['confirmed']:
            raise RuntimeError('A visually confirmed pick is required')
        color = self.held['color']
        current = self.tool()
        measured = self.observe(color, current)
        offset = measured['center']-current
        target_center = np.r_[point[:2], point[2]+measured['extent_z']+.03]
        command = target_center-offset
        high = current.copy()
        high[2] = max(self.held['park'][2], command[2]+.06)
        self.move(high)
        high[:2] = command[:2]
        self.move(high, 2.)
        self.move(command, 1.6)
        for clearance, attempts in ((.024, 3), (.011, 3), (self.release_clearance, 4)):
            for _ in range(attempts):
                self.api.wait(.15)
                current = self.tool()
                measured = self.observe(color, current)
                lateral = point[:2]-measured['center'][:2]
                gap = measured['bottom']-point[2]
                if np.linalg.norm(lateral) < self.position_tolerance and abs(clearance-gap) < self.position_tolerance:
                    break
                delta = np.r_[lateral, clearance-gap]
                length = np.linalg.norm(delta[:2])
                if length > .015:
                    delta[:2] *= .015/length
                delta[2] = np.clip(delta[2], -.014, .014)
                if length > .005 and gap < .012:
                    delta[:2] = 0.
                    delta[2] = min(.014, .014-gap)
                self.move(current+delta, 1.3)
        measured = self.observe(color, self.tool())
        xy_error = float(np.linalg.norm(measured['center'][:2]-point[:2]))
        gap = float(measured['bottom']-point[2])
        if xy_error > self.position_tolerance or abs(gap-self.release_clearance) > self.position_tolerance:
            return {'status': 'placement_unconfirmed', 'held': True,
                    'xy_error_m': xy_error, 'surface_gap_m': gap}
        park = self.held['park'].copy()
        self.api.gripper(.8, seconds=1.5)
        self.held = None
        self.api.wait(.8)
        self.api.gripper(1.4, seconds=1.)
        self.api.wait(1.2)
        retreat = self.tool()
        retreat[2] += .025
        self.move(retreat, 2.)
        self.api.wait(.5)
        retreat[2] += .06
        self.move(retreat, 2.)
        park[2] = max(park[2], retreat[2])
        self.move(park, 1.8)
        result = {'status': 'released', 'held': False, 'xy_error_m': xy_error,
                  'surface_gap_m': gap, 'task_success_certified': False}
        try:
            result['observation'] = self.observe(color)
        except RuntimeError as error:
            result['observation_unavailable'] = str(error)
        return result

    def pick_and_place(self, color, surface_point):
        """Compose one pick and one place; caller supplies all task semantics."""
        point = np.asarray(surface_point, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Destination must be finite world XYZ in meters')
        picked = self.pick(color, transit_z=float(point[2]+.15))
        if not picked['held']:
            return picked
        return self.place(point)
