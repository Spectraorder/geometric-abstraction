"""Matched surface interface over an unchanged mixed rigid session.

Bundle after the shared library providing metric_copy. This development prototype
changes public information only, not sensing, dimension models or motion control.
"""

SURFACE_FIT_FIELDS = ('center', 'normal', 'height', 'dims', 'top', 'bottom', 'extent_z',
                      'coordinate_frame', 'units', 'observation_time', 'source', 'view',
                      'position_fully_observed', 'inferred_orientation')
SURFACE_STATE_FIELDS = ('status', 'last_observed', 'reason', 'center', 'normal',
                        'top', 'bottom', 'extent_z', 'source', 'view',
                        'position_fully_observed', 'inferred_orientation')


def surface_fit(fit):
    return metric_copy({key: fit[key] for key in SURFACE_FIT_FIELDS if key in fit})


def surface_action_result(result):
    # Action statuses and residuals are common execution evidence. A returned
    # observation uses the same projection as direct observe/get calls.
    allowed = ('status', 'held', 'xy_error_m', 'surface_gap_m',
               'task_success_certified', 'observation_unavailable')
    output = metric_copy({key: result[key] for key in allowed if key in result})
    if 'observation' in result:
        output['observation'] = surface_fit(result['observation'])
    return output


def surface_packet(packet):
    return metric_copy({'schema': 'shared_surface_scene_v1',
                        'coordinate_frame': packet['coordinate_frame'], 'units': packet['units'],
                        'time': packet['time'],
                        'objects': {name: {key: row[key] for key in SURFACE_STATE_FIELDS if key in row}
                                    for name, row in packet['objects'].items()},
                        'temporal_pose_fusion': False, 'contact_or_stability_certified': False}, plain=True)


class SurfaceProjectionSession:
    def __init__(self, session):
        self.backend = session

    def observe(self, color):
        return surface_fit(self.backend.observe(color))

    def get(self, color):
        return surface_fit(self.backend.get(color))

    def snapshot(self):
        return surface_packet(self.backend.snapshot())

    def state(self):
        return surface_packet(self.backend.state())

    def time(self):
        return self.backend.time()

    def table_z(self, xy):
        return self.backend.table_z(xy)

    def tool(self):
        return self.backend.tool()

    def pick(self, color, transit_z=None):
        return surface_action_result(self.backend.pick(color, transit_z))

    def place(self, surface_point):
        return surface_action_result(self.backend.place(surface_point))

    def pick_and_place(self, color, surface_point):
        return surface_action_result(self.backend.pick_and_place(color, surface_point))
