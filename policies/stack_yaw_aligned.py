"""Candidate: align the gripper with the target block's observed yaw."""

import math

import numpy as np


def run(api) -> None:
    observation = api.get_observation()
    target_state = observation["objects"]["small_red_block"]
    target = np.asarray(target_state["position"])
    base = np.asarray(observation["objects"]["blue_block"]["position"])
    quaternion = np.asarray(target_state["quaternion_wxyz"])
    target_yaw = 2.0 * math.atan2(quaternion[3], quaternion[0])

    api.open_gripper()
    api.move_end_effector(
        [target[0], target[1], target[2] + 0.10], wrist_roll=target_yaw
    )
    api.move_end_effector(
        [target[0], target[1], target[2] + 0.012], wrist_roll=target_yaw
    )
    api.close_gripper()
    api.wait(0.3)

    transit_height = max(target[2], base[2]) + 0.13
    api.move_end_effector(
        [target[0], target[1], transit_height], wrist_roll=target_yaw
    )
    api.move_end_effector([base[0], base[1], transit_height], wrist_roll=target_yaw)
    api.move_end_effector(
        [base[0], base[1], base[2] + 0.050], wrist_roll=target_yaw
    )
    api.open_gripper()
    api.wait(0.8)
    api.move_end_effector([base[0], base[1], transit_height], wrist_roll=target_yaw)
