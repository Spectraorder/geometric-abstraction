"""Collect initial visual evidence without a task-solving policy or motion."""


def run(api):
    api.camera('side')
    api.camera('wrist')
    api.note('Observation-only baseline: no task planning or manipulation was attempted.')
