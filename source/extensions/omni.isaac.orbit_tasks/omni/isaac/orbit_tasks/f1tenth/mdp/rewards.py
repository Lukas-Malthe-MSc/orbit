# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from rich import print
import numpy as np

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import wrap_to_pi
from omni.isaac.orbit.sensors import Lidar


if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def joint_pos_target_l2(env: RLTaskEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    # print(f"joint_pos_target_l2: {torch.sum(torch.square(joint_pos - target), dim=1)}")
    return torch.sum(torch.square(joint_pos - target), dim=1)

def action_target_log(env: RLTaskEnv, action_idx, lambda_1, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize action from a target value."""
    # extract the used quantities (to enable type-hinting)
    action = env.action_manager.action[:, action_idx]

    # f(x) = log10(lambda*|x - target| + 1)
    return torch.log10(lambda_1 * torch.abs(action - target) + 1)
    

def smooth_action_penalty(env: RLTaskEnv, action_idx, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize action from a target value."""
    action = env.action_manager.action[:, action_idx]
    prev_action = env.action_manager.prev_action[:, action_idx]

    # Ensure action and prev_action are correctly shaped
    assert action.shape == prev_action.shape, "Action and previous action must have the same shape"
    
    # Compute the penalty for each action individually
    penalty = ((action - prev_action) ** 2)
    
    return penalty


def forward_velocity(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.max(asset.data.root_lin_vel_b[:, 0], torch.zeros(asset.data.root_lin_vel_b[:, 0].shape, device=asset.device))

def distance_traveled_reward(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Calculates the reward based on the distance traveled by the asset in the forward direction.

    Args:
        env (RLTaskEnv): The environment containing the simulation and assets.
        asset_cfg (SceneEntityCfg): Configuration for the asset whose distance traveled is to be calculated.

    Returns:
        torch.Tensor: The distance traveled by the asset since the last timestep.
    """
    # Access the asset
    asset: RigidObject = env.scene[asset_cfg.name]

    # Check if the previous position is stored
    if 'prev_position' not in env.custom_data:
        # Initialize previous position if not present
        env.custom_data['prev_position'] = asset.data.root_pos_w[:, :2].clone()

    # Calculate the difference in position from the last step
    current_position = asset.data.root_pos_w[:, :2]
    position_difference = current_position - env.custom_data['prev_position']

    # Calculate the Euclidean distance traveled since the last step
    distance_traveled = torch.norm(position_difference, dim=1)

    # Update the previous position for the next timestep
    env.custom_data['prev_position'] = current_position.clone()

    return distance_traveled



def lidar_min_distance(env: RLTaskEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The min distance of the lidar scans."""
    sensor: Lidar = env.scene[sensor_cfg.name]
    lidar_ranges = sensor.data.output
    min_distances = torch.min(lidar_ranges, dim=1).values
    return 1/min_distances


def passed_starting_location(
                            env:            RLTaskEnv, 
                            asset_cfg:      SceneEntityCfg, 
                            threshold:      float, 
                            ) -> torch.Tensor:
    """Checks if assets have passed their starting locations within some threshold."""
    # Access the asset
    asset: Articulation = env.scene[asset_cfg.name]

    # Check if we've already captured the starting positions for this asset
    if asset_cfg.name not in env.starting_positions:
        # Capture and store the starting positions
        env.starting_positions[asset_cfg.name] = asset.data.root_pos_w[:, :2].clone()

    if env.common_step_counter < 100:
        return torch.zeros(asset.data.root_pos_w.shape[0], dtype=torch.bool, device=asset.device)
    # Retrieve the starting positions
    starting_positions = env.starting_positions[asset_cfg.name]
    
    # print(f"starting_positions: {starting_positions}")
    
    # Compute the difference in positions
    position_differences = asset.data.root_pos_w[:, :2] - starting_positions
    
    # print(f"position_differences: {position_differences}")
    
    # Calculate the distance moved from the starting positions
    distance_moved = torch.norm(position_differences, dim=1)
    
    # Check if the assets have moved beyond the threshold from their starting positions
    passed_threshold = distance_moved > threshold
    # print(f"passed_starting_location: {passed_threshold}")
    return passed_threshold


def timed_lap_time(
                    env:            RLTaskEnv, 
                    asset_cfg:      SceneEntityCfg, 
                    threshold:      float, 
                    lap_threshold:  float = 5.0
                    ) -> torch.Tensor:
    """Checks if assets are within their starting locations +- a threshold and prints time elapsed since leaving and returning for each asset."""
    # Access the asset
    asset: Articulation = env.scene[asset_cfg.name]
    num_assets = asset.data.root_pos_w.shape[0]  # Assuming this gives the number of individual assets
    
    if(env.common_step_counter < 5):
        return torch.zeros(num_assets, device=asset.data.root_pos_w.device) #indicating not started yet
    
    # Initialize tracking if not already done or if there's a timeout event
    if asset_cfg.name not in env.starting_positions:
        env.starting_positions[asset_cfg.name] = {
            "position": asset.data.root_pos_w[:, :2].clone(),
            "left_at_step": torch.full((num_assets,), -1, dtype=torch.int64),
            "returned_at_step": torch.full((num_assets,), -1, dtype=torch.int64),
            "is_lap_completed": torch.full((num_assets,), -1, dtype=torch.int64),
            "lap_times": []  # Buffer to store lap times
        }
    
    tracking_info = env.starting_positions[asset_cfg.name]
    position_differences = asset.data.root_pos_w[:, :2] - tracking_info["position"]
    distance_moved = torch.norm(position_differences, dim=1)
    within_threshold = distance_moved <= threshold

    for i in range(num_assets):
        if tracking_info["is_lap_completed"][i] == -1 and distance_moved[i] > lap_threshold:
            tracking_info["is_lap_completed"][i] = 1
            print(f"Car {i} has left limit")

        if within_threshold[i] and tracking_info["left_at_step"][i] >= 0 and tracking_info["is_lap_completed"][i]:
            tracking_info["returned_at_step"][i] = env.common_step_counter
            time_elapsed = (tracking_info["returned_at_step"][i] - tracking_info["left_at_step"][i]) * env.physics_dt
            print(f"Asset {i}: Time elapsed since leaving and returning: {time_elapsed} seconds, dt={env.physics_dt}")
            tracking_info["lap_times"].append(time_elapsed.item())
            tracking_info["left_at_step"][i] = -1
            tracking_info["is_lap_completed"][i] = -1

        elif not within_threshold[i] and tracking_info["left_at_step"][i] < 0:
            tracking_info["left_at_step"][i] = env.common_step_counter

    # Save lap times to .npy file when simulation ends
    # if env.common_step_counter == 24000/2:
    #     for key, value in env.starting_positions.items():
    #         lap_times_array = np.array(value["lap_times"])
    #         map = "test_track_3_with_fixed_obstacles"
    #         model = "CNN"
    #         np.save(f"data-analysis/data/{map}_{model}_rewards.npy", env.reward_buffer)
    #         np.save(f"data-analysis/data/{map}_{model}_lap_times.npy", lap_times_array)
    #         print(f"Lap times for {key} saved to {key}_lap_times.npy")
            
    # env.reward_buffer.append(env.reward_buf.item())

    return within_threshold