"""Task-free mixed RGB-D callback for the shared PnP observation interface."""
import numpy as np
from aspire_so100.mixed_rigid_state import observe_mixed
from aspire_so100.cuboid_hypotheses import observe_cuboid_correspondence


class MixedPnPObservation:
    def __init__(self, objects, correspondence=False):
        """Public unique-color shape labels and optional initial orientation hints.

        Dimensions are learned from the initial on-table image. The observer has
        no robot API, action selection, task graph, pose history or target state.
        Partially observed coordinates retain the estimator's explicit flags;
        this callback does not promote them to fully observed measurements.
        """
        self.correspondence = bool(correspondence)
        self.objects = {}
        for item in objects:
            color, shape = item['color'], item['shape']
            orientation = item.get('orientation', 'upright')
            if color in self.objects:
                raise ValueError('Unique colors are required')
            if color not in ('red', 'yellow', 'blue', 'green'):
                raise ValueError('Unsupported color')
            if shape not in ('cube', 'prism', 'thin_prism', 'cylinder'):
                raise ValueError('Unsupported rigid shape')
            if orientation not in ('upright', 'flat_x') or (orientation == 'flat_x' and shape != 'thin_prism'):
                raise ValueError('Unsupported initial orientation hint')
            self.objects[color] = {'shape': shape, 'orientation': orientation}

    def observe(self, frame, color, model, table_plane):
        item = self.objects[color]
        measured = observe_mixed(frame, color, item['shape'], model, table_plane, item['orientation'])
        return dict(measured, coordinate_frame='world', units='meters',
                    observation_time=float(frame['time']), source='mixed_rigid_rgbd_observation')

    def observe_views(self, frames, color, model, table_plane):
        """Use the first usable same-time view; never average incompatible poses.

        Frames are an ordered list of (view name, calibrated RGB-D frame) pairs.
        This pure function does not capture cameras or propagate a hidden pose.
        Caller acquisition policy must be common across experimental arms.
        """
        if not frames:
            raise ValueError('At least one current view is required')
        stamps = np.asarray([float(frame['time']) for _, frame in frames])
        if not np.isfinite(stamps).all() or np.ptp(stamps) > 1e-6:
            raise ValueError('All views must have the same finite timestamp')
        failures = []
        for view, frame in frames:
            try:
                measured = self.observe(frame, color, model, table_plane)
                return dict(measured, view=view, rejected_views=failures)
            except (ValueError, RuntimeError) as error:
                failures.append({'view': view, 'reason': str(error)})
        item = self.objects[color]
        if self.correspondence and model is not None and item['shape'] in ('prism', 'thin_prism'):
            try:
                measured = observe_cuboid_correspondence(frames, color, item['shape'], model, table_plane)
                return dict(measured, rejected_views=failures)
            except ValueError as error:
                failures.append({'view': 'joint_correspondence', 'reason': str(error)})
        raise ValueError('No usable current view: '+str(failures))
