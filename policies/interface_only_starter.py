"""Observation-only baseline with a shared, task-independent logging adapter."""


def checkpoint_action(session, color, step, result):
    """Forward an action's status string without changing its outcome or geometry."""
    status = result['status']
    if not isinstance(status, str):
        raise ValueError('Action result status must be a string')
    return session.checkpoint(color, step, action_status=status)


def run(api):
    api.camera('side')
    api.camera('wrist')
    api.note('Observation-only baseline: no task planning or manipulation was attempted.')
