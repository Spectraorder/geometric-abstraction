"""Development-only explicit opening primitive shared by both planning interfaces.

Opening clears the executor's held bookkeeping after the command succeeds. It
certifies no object pose, contact, release, stable support or task completion.
No perception, task-specific trigger, retreat motion or automatic recovery is added.
"""
import argparse
from pathlib import Path
from run_vision_trial import validate_source


def library(source):
    replacements = [
        ('    def pick(self, color, transit_z=None):\n        """Pick one object; lifting is checked visually, not with a contact oracle."""',
         '''    def open_gripper(self):
        """Open explicitly; held=False describes bookkeeping, not object truth."""
        self.api.gripper(.8, seconds=.6)
        self.held = None
        return {'status': 'opened_unverified', 'held': False}

    def pick(self, color, transit_z=None):
        """Pick one object; lifting is checked visually, not with a contact oracle."""'''),
        ('    def pick(self, color, transit_z=None):\n        return metric_copy(self.pnp.pick(color, transit_z))',
         '''    def open_gripper(self):
        return metric_copy(self.pnp.open_gripper())

    def pick(self, color, transit_z=None):
        return metric_copy(self.pnp.pick(color, transit_z))'''),
        ('    def pick(self, color, transit_z=None):\n        return visual_action_result(self.backend.pick(color, transit_z))',
         '''    def open_gripper(self):
        return visual_action_result(self.backend.open_gripper())

    def pick(self, color, transit_z=None):
        return visual_action_result(self.backend.pick(color, transit_z))''')]
    for old,new in replacements:
        if source.count(old)!=1:raise ValueError('Expected one method insertion site')
        source=source.replace(old,new,1)
    validate_source(source)
    return source


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-library',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:stream.write(library(args.frozen_library.read_text()))
