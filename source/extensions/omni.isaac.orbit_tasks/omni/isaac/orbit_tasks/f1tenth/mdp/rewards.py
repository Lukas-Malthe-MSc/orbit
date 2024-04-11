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

def speed_scaled_distance_reward(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Calculates a reward based on the distance traveled scaled by the time taken, rewarding faster travel over a given distance.

    Args:
        env (RLTaskEnv): The environment containing the simulation and assets.
        asset_cfg (SceneEntityCfg): Configuration for the asset whose performance is being evaluated.

    Returns:
        torch.Tensor: The scaled reward for the asset's travel over the last timestep.
    """
    # Access the asset
    asset: RigidObject = env.scene[asset_cfg.name]

    # Initialize or retrieve previous position for distance calculation
    if 'prev_position' not in env.custom_data:
        env.custom_data['prev_position'] = asset.data.root_pos_w[:, :2].clone()
        return torch.tensor(0.0)  # No reward on the first step

    # Calculate the difference in position from the last step
    current_position = asset.data.root_pos_w[:, :2]
    position_difference = current_position - env.custom_data['prev_position']

    # Calculate the Euclidean distance traveled since the last step
    distance_traveled = torch.norm(position_difference, dim=1)

    # Assuming env.step_dt represents the time elapsed between steps
    time_elapsed = env.step_dt

    # Calculate the reward as distance divided by time (speed)
    # In environments with consistent time steps, you could simply use `distance_traveled` as the reward.
    reward = distance_traveled / time_elapsed

    # Update the previous position for the next timestep
    env.custom_data['prev_position'] = current_position.clone()

    return reward



def lidar_min_distance(env: RLTaskEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The min distance of the lidar scans."""
    sensor: Lidar = env.scene[sensor_cfg.name]
    lidar_ranges = sensor.data.output
    min_distances = torch.min(lidar_ranges, dim=1).values
    return 1/min_distances

# move to position x, y
def move_to_position(env: RLTaskEnv, target: torch.Tensor, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the distance between the asset's root position and the target position."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    return torch.sqrt(torch.sum(torch.square(asset.data.root_pos_w[:, :2] - target), dim=1))

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


def within_starting_location(env: RLTaskEnv, asset_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Checks if assets are within their starting locations +- a threshold."""
    # Access the asset
    asset: Articulation = env.scene[asset_cfg.name]

    # Check if we've already captured the starting positions for this asset
    if asset_cfg.name not in env.starting_positions or any(env.termination_manager.time_outs):
        # Capture and store the starting positions
        env.starting_positions[asset_cfg.name] = asset.data.root_pos_w[:, :2].clone()
        print(f"starting_positions: {env.starting_positions[asset_cfg.name]}")

    # If it's too early in the simulation, assume we're within the starting location
    if env.common_step_counter < 100:
        return torch.zeros(asset.data.root_pos_w.shape[0], dtype=torch.bool, device=asset.device)
    
    # Retrieve the starting positions
    starting_positions = env.starting_positions[asset_cfg.name]

    # Compute the difference in positions
    position_differences = asset.data.root_pos_w[:, :2] - starting_positions

    # Calculate the distance moved from the starting positions
    distance_moved = torch.norm(position_differences, dim=1)

    # Check if the assets are within the threshold of their starting positions
    within_threshold = distance_moved <= threshold
    # print(f"within_starting_location: {within_threshold}")
    return within_threshold
