"""Shared sensor/execution adapter; task decisions stay in the calling planner.

Bundle after the numeric scene and shared observed PnP classes. Both experiment
interfaces use these same sensors, observer, current-state ledger and actuator.
"""


class MixedSession:
    def __init__(self, api, objects):
        self.api = api
        self.backend = MixedPnPObservation(objects, correspondence=True)
        self.ledger = MetricScene()
        self.colors = [item['color'] for item in objects]
        self.pnp = ObservedRGBDPickPlace(api)
        self.pnp.set_observer(self.current_observation)
        # Dimensions must be established while every source is initially on the
        # table. Never initialize a missing model from an airborne observation.
        for color in self.colors:
            self.pnp.observe(color)

    def current_observation(self, frame, color, model, plane):
        robot = self.api.gripper_geometry()
        side, side_exclusion = filter_robot_visual_geometry(frame, robot)
        try:
            try:
                measured = self.backend.observe_views([('side', side)], color, model, plane)
                exclusions = {'side': side_exclusion}
            except ValueError:
                wrist, wrist_exclusion = filter_robot_visual_geometry(self.api.camera('wrist'), robot)
                measured = self.backend.observe_views([('side', side), ('wrist', wrist)], color, model, plane)
                exclusions = {'side': side_exclusion, 'wrist': wrist_exclusion}
        except (ValueError, RuntimeError) as error:
            self.ledger.unavailable(color, float(frame['time']), str(error))
            raise
        measured = dict(measured, robot_exclusions=exclusions)
        # Expose the same validated schema through observe(), get() and state().
        # Shape-specific fitters may omit fields such as axis_dimensions.
        return self.ledger.observe(color, measured)

    def observe(self, color):
        return metric_copy(self.pnp.observe(color))

    def snapshot(self):
        """Explicitly refresh every object; unavailable observations stay missing."""
        for color in self.colors:
            try:
                self.pnp.observe(color)
            except (ValueError, RuntimeError) as error:
                self.ledger.unavailable(color, self.time(), str(error))
        return self.state()

    def time(self):
        return float(self.api.proprioception()['time'])

    def get(self, color):
        return self.ledger.get(color, self.time())

    def state(self):
        return self.ledger.packet(self.time())

    def history(self, color):
        return self.ledger.history(color)

    def target(self, color, center, rotation=None):
        return self.ledger.target(color, center, rotation, self.time())

    def goal_error(self, color):
        return self.ledger.goal_error(color, self.time())

    def table_z(self, xy):
        return self.pnp.table_z(xy)

    def tool(self):
        return self.pnp.tool()

    def pick(self, color, transit_z=None):
        return metric_copy(self.pnp.pick(color, transit_z))

    def place(self, surface_point):
        return metric_copy(self.pnp.place(surface_point))

    def pick_and_place(self, color, surface_point):
        return metric_copy(self.pnp.pick_and_place(color, surface_point))


class SymbolicSession(MixedSession):
    def reference(self, name):
        return SymbolicScene(self.ledger).reference(name, self.time())

    def relation(self, predicate, first, second, tolerance_m=.004):
        return SymbolicScene(self.ledger).relation(predicate, first, second, stamp=self.time(), tolerance_m=tolerance_m)

    def align(self, source_reference, target_point, target_normal):
        return SymbolicScene(self.ledger).align(source_reference, target_point, target_normal, self.time())

    def state(self):
        return SymbolicScene(self.ledger).packet(self.time())

    def text(self):
        return SymbolicScene(self.ledger).text(self.time())
