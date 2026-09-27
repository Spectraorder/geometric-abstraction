"""Instrument an existing cube controller with sensor-only rigid state records.

The new wrist estimates are diagnostic in this first experiment. They do not
silently replace the mature controller's perception or trigger new corrections.
"""
import numpy as np
from aspire_so100.ablation_geometry import observe_cube
from aspire_so100.instance_geometry import inventory_table
from aspire_so100.rigid_state import stack_targets, state_error, support_relation, geometry_json


class RigidStateObserver:
    def __init__(self, api):
        self.api = api
        self.order = [item['color'] for item in api.task()['levels']]
        self.models = {}
        self.plane = None
        self.anchor = None
        self.anchor_label = None
        self.sequence = 0

    def snapshot(self, label, reanchor=False):
        side, wrist = self.api.camera('side'), self.api.camera('wrist')
        if self.plane is None:
            self.plane = inventory_table(side)
        views = {'side': side}
        if 'depth_m' in wrist:
            views['wrist'] = wrist
        objects, diagnostics = {}, {}
        for color in self.order:
            observations = {}
            for view, frame in views.items():
                try:
                    if color not in self.models and view != 'side':
                        raise ValueError('Initial size must be measured in the global tabletop view')
                    measured = observe_cube(frame, color, self.models.get(color), self.plane)
                    self.models[color] = measured['model']
                    observations[view] = measured
                except ValueError as error:
                    observations[view] = {'unavailable': str(error)}
            available = [(view, value) for view, value in observations.items() if 'unavailable' not in value]
            if available:
                view, value = max(available, key=lambda pair: (
                    pair[1]['geometry_reliable'], pair[1]['observed_face_count'], pair[1]['inlier_fraction']))
                objects[color] = dict(value, selected_view=view)
            diagnostics[color] = observations
        if self.anchor is None or reanchor:
            if self.order[0] in objects:
                self.anchor = objects[self.order[0]]
                self.anchor_label = label
        targets = {}
        if self.anchor is not None and all(color in self.models for color in self.order):
            sizes = {color: {'height': self.models[color]['height']} for color in self.order}
            targets = stack_targets(sizes, self.order, self.plane, anchor=self.anchor)
        errors = {color: state_error(value, targets[color]) for color, value in objects.items() if color in targets}
        relations = {}
        for lower, upper in zip(self.order[:-1], self.order[1:]):
            if upper in objects and lower in objects:
                relations[upper + '_on_' + lower] = support_relation(objects[upper], objects[lower])
        record = {'schema_version': 1, 'sequence': self.sequence, 'label': label,
                  'time': float(side['time']), 'coordinate_frame': 'world', 'units': 'meters_radians',
                  'table_plane': self.plane, 'objects': objects, 'per_view': diagnostics,
                  'targets': targets, 'target_anchor_label': self.anchor_label,
                  'errors': errors, 'relations': relations,
                  'gripper': self.api.gripper_geometry(),
                  'selection': 'best_supported_single_view_not_fused',
                  'wrist_depth_enabled': 'depth_m' in wrist,
                  'control_role': 'diagnostic_observer_existing_controller_executes_actions'}
        self.api.record_geometry(geometry_json(record))
        self.sequence += 1

    def camera(self, view='side'):
        return self.api.camera(view)

    def task(self):
        return self.api.task()

    def proprioception(self):
        return self.api.proprioception()

    def move_tool(self, position, seconds=1.5, orientation=None):
        return self.api.move_tool(position, seconds=seconds, orientation=orientation)

    def plan_tool(self, position, orientation=None):
        return self.api.plan_tool(position, orientation=orientation)

    def move_joints(self, angles, seconds=1.5):
        return self.api.move_joints(angles, seconds=seconds)

    def gripper(self, angle, seconds=.6):
        return self.api.gripper(angle, seconds=seconds)

    def wait(self, seconds):
        return self.api.wait(seconds)

    def note(self, text):
        self.api.note(text)
        if any(marker in text for marker in ('payload tilt=', 'release offset=', 'after withdrawal center=', 'LEVEL confirmed')):
            base_withdrawn = text.startswith(self.order[0] + '_') and 'after withdrawal center=' in text
            self.snapshot(text, reanchor=base_withdrawn)


def run(api):
    observer = RigidStateObserver(api)
    observer.snapshot('initial')
    execute_column(observer)
    observer.snapshot('final')
