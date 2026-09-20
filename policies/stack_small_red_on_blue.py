"""Baseline: top-down grasp, lift, transport, descend, and release."""

import numpy as np


def run(api) -> None:
    observation = api.get_observation()
    target = np.asarray(observation["objects"]["small_red_block"]["position"])
    base = np.asarray(observation["objects"]["blue_block"]["position"])

    api.open_gripper()
    # wrist_roll=0 turns the jaws across the block's short axis (4 cm).
    api.move_end_effector(
        [target[0], target[1], target[2] + 0.10], wrist_roll=0.0
    )
    api.move_end_effector(
        [target[0], target[1], target[2] + 0.012], wrist_roll=0.0
    )
    api.close_gripper()
    api.wait(0.3)

    transit_height = max(target[2], base[2]) + 0.13
    api.move_end_effector([target[0], target[1], transit_height], wrist_roll=0.0)
    api.move_end_effector([base[0], base[1], transit_height], wrist_roll=0.0)
    api.move_end_effector(
        [base[0], base[1], base[2] + 0.050], wrist_roll=0.0
    )
    api.open_gripper()
    api.wait(0.8)
    api.move_end_effector([base[0], base[1], transit_height], wrist_roll=0.0)
