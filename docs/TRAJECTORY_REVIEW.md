# Human trajectory review in Viser

Every saved vision-policy trial now overlays its policy plan and physical tool
motion in Viser. This is an offline review of the exact recorded execution; it
does not rerun the policy or expose evaluator state to it.

```bash
uv run --no-sync python view_scene.py --vision-trial PATH_TO_TRIAL
```

Open **Trajectory review** in the Viser GUI. Purple is the executed tool-center
path reconstructed from saved MuJoCo states up through the current playback
time; future motion is not drawn. Orange segments appear only after their
`move_tool` request was issued and connect that moment's tool position to the
requested target. The green point shows the current replay position. IK previews
and floating timestamp labels are intentionally omitted to keep the 3D view
uncluttered.

The translucent cyan SO-100 is a single look-ahead phantom. By default it shows
the arm's recorded pose one second after the current frame, including wrist and
gripper orientation. **Phantom look-ahead (s)** adjusts that horizon from 0.1 to
3 seconds. It disappears at the end of the rollout and does not draw a trail of
future robot poses.

Pause playback in **Controls** before dragging **Review time (s)**. The previous
and next-event buttons jump among policy events such as motions, gripper
commands, waits, notes and reports. The status panel names the last and next
policy event.

The clock begins at the first recorded rollout state. API timestamps use the
same simulator clock and are converted to seconds from that start. The policy-end
mark comes from `policy_sim_seconds`; time after that mark is the evaluator's
stability hold, rather than policy-authored motion. Generated policy source does
not need a global clock or timestamp annotations.

The overlay distinguishes requested and actual motion but is not a collision
certificate. Human reviewers should inspect the executed purple path, object
motion and contacts, especially around orange motion targets and gripper events.
The raw sources remain `api_calls.json` and `rollout.npz` inside the trial
directory.
