"""Action-boundary logging over the unchanged shared sensor and PnP session.

Bundle after GeometryStepState and the existing mixed/surface session classes.
No global snapshot is taken. A refresh is explicit and limited to the focus object.
"""


class GeometryStepSession(SymbolicSession):
    step_state = None
    placement_intent = None

    def pick(self, color, transit_z=None):
        if self.pnp.held is None:
            self.placement_intent = None
        return MixedSession.pick(self, color, transit_z)

    def place(self, surface_point):
        point = step_vector(surface_point)
        if self.pnp.held is not None and self.pnp.held['confirmed']:
            self.placement_intent = {'object': self.pnp.held['color'], 'point': point}
        return MixedSession.place(self, point)

    def pick_and_place(self, color, surface_point):
        # Preserve the common executor's transit height and failure behavior.
        point = step_vector(surface_point)
        picked = self.pick(color, transit_z=float(point[2]+.085))
        if not picked['held']:
            return picked
        return self.place(point)

    def checkpoint(self, color, step, refresh=False, action_status=None,
                   goal_center=None, reference_names=(), surface_only=False):
        if color not in self.colors:
            raise ValueError('Checkpoint requires a known object')
        if self.step_state is None or self.step_state.object_id != color:
            self.step_state = GeometryStepState(color)
        if goal_center is not None:
            self.target(color, goal_center)
        desired = self.ledger.targets.get(color)
        if desired is None:
            self.step_state.clear_goal()
        else:
            self.step_state.set_goal(desired['center'])
        try:
            fit = self.observe(color) if refresh else self.get(color)
        except (ValueError, RuntimeError):
            fit = None
        now = self.time()
        if surface_only:
            record = {'object': color, 'step': step, 'time': now,
                      'frame': 'world', 'units': 'meters',
                      'status': 'unavailable' if fit is None else 'current_estimate'}
            if fit is not None:
                record['geometry'] = metric_copy(surface_fit(fit), plain=True)
            if action_status is not None:
                record['action_status'] = action_status
        else:
            references = []
            if fit is not None:
                references = [self.reference(name) for name in reference_names]
            record = self.step_state.update(step, fit, now, references, action_status)
            intent = self.placement_intent
            if intent is not None and intent['object'] == color:
                # This is the executor's bounding-extent target, not a face point,
                # a servo command, or evidence of physical contact/stability.
                placement = {'reference': 'center_xy_min_z',
                             'point_m': step_display(intent['point']),
                             'remaining_translation_m': None}
                if record['status'] == 'current_estimate' and 'bottom' in fit:
                    actual = step_vector([fit['center'][0], fit['center'][1], fit['bottom']])
                    placement['remaining_translation_m'] = step_display(intent['point']-actual)
                record['placement_goal'] = placement
        self.api.record_geometry({'kind': 'geometry_step', 'record': record})
        return record


class SurfaceStepSession(SurfaceProjectionSession):
    def checkpoint(self, color, step, refresh=False, action_status=None):
        return self.backend.checkpoint(color, step, refresh, action_status, surface_only=True)
