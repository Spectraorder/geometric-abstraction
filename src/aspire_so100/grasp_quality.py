"""Private grasp-quality diagnostics; never supplied as runtime observations."""
import math
import numpy as np

GRASP_QUALITY_VERSION = 'grasp_quality_v1'


class GraspQuality:
    """Measure free-air nut motion relative to the gripper, not world motion.

    Thresholds are experimental diagnostics, not calibrated hardware tolerances.
    Deliberate thread rotation and supported/released objects are excluded.
    """
    def __init__(self):
        self.anchor = None
        self.seconds = 0.
        self.max_seconds = 0.
        self.max_travel = 0.
        self.stable_grasp = False
        self.stable_transport = False
        self.slip_events = 0
        self.max_slip = 0.
        self.max_rotation = 0.
        self.air_seconds = 0.
        self.dual_seconds = 0.
        self.slipping = False

    def update(self, dt, nut_position, nut_rotation, tool_position, tool_rotation,
               *, airborne, touching, both_jaws, unsupported, engaged):
        if not engaged and airborne and touching:
            self.air_seconds += dt
            self.dual_seconds += dt*bool(both_jaws)
        eligible = airborne and both_jaws and unsupported and not engaged
        relative_position = tool_rotation.T @ (nut_position-tool_position)
        relative_rotation = tool_rotation.T @ nut_rotation
        if not eligible:
            self.anchor = None
            self.seconds = 0.
            self.slipping = False
            return
        if self.anchor is None:
            self.anchor = (relative_position.copy(), relative_rotation.copy(), tool_position.copy())
        p, rotation, origin = self.anchor
        slip = float(np.linalg.norm(relative_position-p))
        angle = math.acos(float(np.clip((np.trace(rotation.T@relative_rotation)-1)/2,-1.,1.)))
        self.max_slip = max(self.max_slip, slip)
        self.max_rotation = max(self.max_rotation, angle)
        upright = nut_rotation[2,2] >= math.cos(math.radians(15))
        slipped = slip > .003 or angle > math.radians(5)
        if slipped or not upright:
            if slipped and not self.slipping:
                self.slip_events += 1
            self.slipping = slipped
            self.seconds = 0.
            # Keep the old pose while badly tilted so a sustained flip is not
            # forgotten on each frame; restart a window once upright again.
            if upright:
                self.anchor = (relative_position.copy(), relative_rotation.copy(), tool_position.copy())
            return
        self.slipping = False
        self.seconds += dt
        self.max_seconds = max(self.max_seconds,self.seconds)
        travel = float(np.linalg.norm(tool_position-origin))
        self.max_travel = max(self.max_travel,travel)
        self.stable_grasp |= self.seconds >= 1.
        self.stable_transport |= self.seconds >= 1. and travel >= .03

    def status(self):
        return dict(stable_grasp=bool(self.stable_grasp), stable_transport=bool(self.stable_transport),
                    current_stable_seconds=self.seconds, max_stable_seconds=self.max_seconds,
                    max_stable_travel_m=self.max_travel, slip_events=self.slip_events,
                    max_relative_slip_m=self.max_slip, max_relative_rotation_deg=math.degrees(self.max_rotation),
                    airborne_contact_seconds=self.air_seconds,
                    airborne_two_jaw_fraction=self.dual_seconds/self.air_seconds if self.air_seconds else 0.)
