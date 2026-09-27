"""Opt-in skill parameter binding for development, without inferred destinations.

Existing node-only calls retain their behavior. An explicitly bound world surface
point is an action intent, not a measured pose or a task-success certificate.
"""
from contextlib import contextmanager
import numpy as np
from online_event_skills import EventSkills, drive
from online_bridge_skills import append


def surface_point(value):
    """Accept only three finite numeric world coordinates in meters."""
    if isinstance(value, (list, tuple)) and any(isinstance(x, (bool, np.bool_)) for x in value):
        raise ValueError('surface_point_world_m requires numbers, not booleans')
    point = np.asarray(value)
    if point.shape != (3,) or point.dtype.kind not in 'iuf' or not np.isfinite(point).all():
        raise ValueError('surface_point_world_m requires three finite numbers')
    return tuple(float(x) for x in point)


class BoundTargetSkills(EventSkills):
    @contextmanager
    def bind_target(self, skill, key, point):
        if getattr(self, '_bound_target', None) is not None:
            raise RuntimeError('A target binding is already active')
        if key not in self.nodes or skill not in ('bridge', 'stack'):
            raise ValueError('Unknown skill/node')
        if (len(self.nodes[key]['supports']) == 1) != (skill == 'stack'):
            raise ValueError('Skill does not match the public support graph')
        binding = (key, surface_point(point))
        self._bound_target = binding
        try:
            append(self.api.output/'target_bindings.jsonl', {
                'node':key, 'skill':skill, 'surface_point_world_m':binding[1],
                'time':self.session.time(), 'semantics':'Caller action intent; no destination inference.'})
            yield self
        finally:
            self._bound_target = None

    def _attempt(self, node):
        binding = getattr(self, '_bound_target', None)
        if binding is None:
            return (yield from super()._attempt(node))
        key, coordinates = binding
        if self.nodes[key] is not node:
            raise ValueError('Target binding belongs to another object')
        target = np.asarray(coordinates, dtype=float)
        result = yield from self._attempt_bound(node, target)
        result.update(surface_point_world_m=list(coordinates), target_source='caller',
                      task_success_certified=False)
        return result

    def _attempt_bound(self, node, target):
        b, api, session = self.bridge, self.api, self.session
        held = self.holding()
        if held is None:
            good, _, _ = b.settled_check(api, session, node, target, self.scale)
            if good:
                return {'status':'visually_consistent'}
            current = b.observe(session, api, node['color'], attempts=1)
            if current is None:
                return {'status':'object_unavailable'}
            if b.orientation_evidence(node, b.fit_geometry(current)) is False:
                return {'status':'orientation_incompatible'}
            picked = yield from session.pick_steps(node['color'],
                transit_z=float(target[2]+current['height']+.085))
            if not picked['held'] or picked['status'] != 'lift_visually_confirmed':
                return {'status':picked['status']}
            yield {'phase':'lift_confirmed', 'boundary':'caller_target_retained_before_transfer'}
        elif held['color'] != node['color'] or not held['lift_confirmed']:
            return {'status':'holding_uncertain' if held['color']==node['color'] else 'another_object_held'}
        # Support changes may trigger cancellation by the common boundary guard.
        # Neither an old nor a new observed support silently rewrites this intent.
        placed = yield from session.place_steps(target.tolist())
        if placed['held']:
            return {'status':placed['status']}
        yield {'phase':'released', 'boundary':'settle_and_reobserve'}
        api.wait(.3)
        good, _, _ = b.settled_check(api, session, node, target, self.scale)
        return {'status':'visually_consistent' if good else 'unconfirmed'}


def drive_bound(skills, operation, on_boundary=None):
    """Run one explicitly parameterized placement through existing motion boundaries."""
    if operation.get('action') != 'execute':
        raise ValueError('A bound target is valid only for execute')
    with skills.bind_target(operation['skill'], operation['node'], operation['surface_point_world_m']):
        result = drive(skills, operation, on_boundary)
        result.update(surface_point_world_m=list(skills._bound_target[1]),
                      target_source='caller', task_success_certified=False)
        return result
