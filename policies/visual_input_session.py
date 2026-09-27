"""Raw-visual planning interface over the identical hidden RGB-D PnP actuator.

Object coordinates, geometric residuals and perception results never cross this
interface. Policies may compute their own geometry from api.camera() pixels and
calibration. Robot-only proprioception and public task labels remain available.
"""


def visual_action_result(result):
    return {key: result[key] for key in ('status', 'held') if key in result}


class VisualInputSession:
    def __init__(self, session):
        self.backend = session

    def time(self):
        return self.backend.time()

    def tool(self):
        return self.backend.tool()

    def pick(self, color, transit_z=None):
        return visual_action_result(self.backend.pick(color, transit_z))

    def place(self, surface_point):
        return visual_action_result(self.backend.place(surface_point))

    def pick_and_place(self, color, surface_point):
        return visual_action_result(self.backend.pick_and_place(color, surface_point))

    def checkpoint(self, color, step, refresh=False, action_status=None):
        if color not in self.backend.colors:
            raise ValueError('Checkpoint requires a known object')
        if not isinstance(step, str) or not step:
            raise ValueError('An explicit action-step label is required')
        if refresh:
            raise ValueError('Use api.camera() for visual evidence; no supplied object observation')
        record = {'object': color, 'step': step, 'time': self.time(),
                  'status': 'visual_input_only'}
        if action_status is not None:
            if not isinstance(action_status, str):
                raise ValueError('Action status must be a string')
            record['action_status'] = action_status
        self.backend.api.record_geometry({'kind': 'geometry_step', 'record': record})
        return record
