# Shared policy interface contract

This contract describes syntax, types and units. It supplies no task solution,
object sequence, workspace, perception algorithm or placement-target calculation.
The same contract and checkpoint adapter are supplied to both experimental arms.

## Valid source structure

Return the complete policy source, including `def run(api):` and every helper it
uses. The host provides `Session`; do not import it, redefine it or access its
internals. Construct it as `Session(api, objects)` using the complete public inventory.

At module level, only function definitions, permitted imports and string docstrings
are allowed. **No module-level assignments**, including constants such as
`HYPOTHESIS = "..."`, dictionaries or arrays. Put constants and mutable state inside
`run` or a helper function. Do not use `global`, `nonlocal`, classes, attribute
assignment, files, tools or dynamic execution. Imports are restricted to numpy,
scipy and math. The 30000-character source limit includes comments and helpers.

A valid way to define a diagnostic string is:

```python
def hypothesis_text():
    return 'Describe the current hypothesis here.'
```

## Action results and checkpoints

| Call or value | Contract |
| --- | --- |
| `session.pick(color, transit_z=None)` | Returns an action-result dictionary. |
| `session.place(surface_xyz)` | Returns an action-result dictionary. |
| `session.pick_and_place(color, surface_xyz)` | Returns the final action-result dictionary, including a failed pick if applicable. |
| `result['status']` | A string describing the executor outcome. |
| `result['held']` | A boolean describing the executor's holding state. |
| `session.checkpoint(color, step, action_status=None)` | `color` is a known object color, `step` a nonempty string, `action_status` a string or `None`. |
| `surface_xyz` | Finite length-three world coordinates in meters: desired object center X/Y and destination support-surface Z. |
| `transit_z` | Optional world Z in meters. |

The complete action-result dictionary is **not** a valid `action_status` argument.
Do not pass it directly and do not use `str(result)` to conceal a type mismatch.
Use the shared helper from the starter, copying it into your returned source:

```python
def checkpoint_action(session, color, step, result):
    status = result['status']
    if not isinstance(status, str):
        raise ValueError('Action result status must be a string')
    return session.checkpoint(color, step, action_status=status)
```

After an action result has been obtained, logging can be written as:

```python
checkpoint_action(session, color, 'pick', picked_result)
checkpoint_action(session, color, 'place', placed_result)
```

These examples do not prescribe which action to take or calculate any target.
The helper is ordinary policy source, not an additional injected runtime method.
It performs no observation refresh or motion and does not catch an action error.
Base's checkpoint returns only metadata; Full's retains its existing geometry
behavior. No Full-only fields are required by the shared helper.

Current executor statuses include `lift_visually_confirmed` and
`grasp_unconfirmed` after pick, and `released` and `placement_unconfirmed` after
place. Read these strings from the result rather than reconstructing them.
`held=True` and `released` are not task-success or stability certificates.
An unconfirmed placement can leave `held=True`. Preserve the executor's existing
observation, association and release checks; the helper changes none of them.

## Observation boundaries and submission check

Base action results expose only status/held. Full may additionally expose visual
estimates and residuals. A checkpoint's own `status` describes its observation
availability and is distinct from an action result's status. Use the returned
action dictionary for `action_status`.

Before submitting, check that constants are inside functions, all referenced
helpers are included, `checkpoint` receives a string or `None`, and all targets
have the documented shape and units. This is a source-review checklist, not a
claim that code has been executed. No simulator, hidden evaluator feedback or
free model-repair turn is available during submission.
