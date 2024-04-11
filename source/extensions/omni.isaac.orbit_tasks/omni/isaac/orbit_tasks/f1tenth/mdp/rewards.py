# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from rich import print

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

def forward_velocity(
    env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # print(f"forward_velocity: {asset.data.root_lin_vel_b[:, 0]}")
    return torch.max(asset.data.root_lin_vel_b[:, 0], torch.zeros(asset.data.root_lin_vel_b[:, 0].shape, device=asset.device))


def lidar_mean_absolute_deviation(env: RLTaskEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The mean absolute deviation of the lidar readings."""
    sensor: Lidar = env.scene[sensor_cfg.name]
    lidar_ranges = sensor.data.output
    
    mean = torch.mean(lidar_ranges, dim=1)
    absolute_deviation = torch.abs(lidar_ranges - mean.unsqueeze(1))
    return torch.mean(absolute_deviation, dim=1)

def lidar_min_distance(env: RLTaskEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The min distance of the lidar scans."""
    sensor: Lidar = env.scene[sensor_cfg.name]
    lidar_ranges = sensor.data.output
    min_distances = torch.min(lidar_ranges, dim=1).values
    return 1/min_distances


def passed_starting_location(env: RLTaskEnv, asset_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
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
    
    print(f"starting_positions: {starting_positions}")
    
    # Compute the difference in positions
    position_differences = asset.data.root_pos_w[:, :2] - starting_positions
    
    print(f"position_differences: {position_differences}")
    
    # Calculate the distance moved from the starting positions
    distance_moved = torch.norm(position_differences, dim=1)
    
    # Check if the assets have moved beyond the threshold from their starting positions
    passed_threshold = distance_moved > threshold
    # print(f"passed_starting_location: {passed_threshold}")
    return passed_threshold
    

def update_pass_counters(env: RLTaskEnv, asset_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Updates counters for each asset based on whether they've passed their starting location."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Ensure we have starting positions
    if asset_cfg.name not in env.starting_positions:
        env.starting_positions[asset_cfg.name] = asset.data.root_pos_w[:, :2].clone()
        env.pass_counters[asset_cfg.name] = torch.zeros(asset.data.root_pos_w.shape[0], dtype=torch.int64, device=asset.device)

    # Retrieve starting positions and pass counters
    starting_positions = env.starting_positions[asset_cfg.name]
    pass_counters = env.pass_counters[asset_cfg.name]

    # Compute the difference in positions from the starting point
    position_differences = asset.data.root_pos_w[:, :2] - starting_positions
    
    # Calculate the distance moved from the starting positions
    distance_moved = torch.norm(position_differences, dim=1)
    
    # Identify assets that have moved beyond the threshold
    passed_threshold = distance_moved > threshold

    # Update the pass counters for assets that have passed the threshold
    # This simplistic approach increments the counter every time the asset is beyond the threshold
    # A more sophisticated approach might track entering and exiting the threshold region
    pass_counters += passed_threshold.int()

    # Save the updated pass counters back to the dictionary
    env.pass_counters[asset_cfg.name] = pass_counters

    return pass_counters


def timed_lap_time(env: RLTaskEnv, asset_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Checks if assets are within their starting locations +- a threshold and prints time elapsed since leaving and returning for each asset."""
    # Access the asset
    asset: Articulation = env.scene[asset_cfg.name]

    num_assets = asset.data.root_pos_w.shape[0]  # Assuming this gives the number of individual assets

    # Initialize tracking if not already done or if there's a timeout event
    if asset_cfg.name not in env.starting_positions: #any(env.termination_manager.time_outs)
        env.starting_positions[asset_cfg.name] = {
            "position": asset.data.root_pos_w[:, :2].clone(),
            "left_at_step": torch.full((num_assets,), -1, dtype=torch.int64),  # -1 indicates that the asset hasn't left yet
            "returned_at_step": torch.full((num_assets,), -1, dtype=torch.int64)  # Step counter when the asset last returned within the threshold
        }

    tracking_info = env.starting_positions[asset_cfg.name]

    # Compute the difference in positions and the distance moved for each asset
    position_differences = asset.data.root_pos_w[:, :2] - tracking_info["position"]
    distance_moved = torch.norm(position_differences, dim=1)

    # Check if the assets are within the threshold of their starting positions
    within_threshold = distance_moved <= threshold

    for i in range(num_assets):
        if within_threshold[i] and tracking_info["left_at_step"][i] >= 0:
            # Calculate and print the elapsed time for assets that have returned
            tracking_info["returned_at_step"][i] = env.common_step_counter
            time_elapsed = (tracking_info["returned_at_step"][i] - tracking_info["left_at_step"][i]) * env.step_dt
            print(f"Asset {i}: Time elapsed since leaving and returning: {time_elapsed} seconds")
            tracking_info["left_at_step"][i] = -1  # Reset after calculating time

        elif not within_threshold[i] and tracking_info["left_at_step"][i] < 0:
            # Mark the step counter for assets that just left the threshold
            tracking_info["left_at_step"][i] = env.common_step_counter

    return within_threshold
