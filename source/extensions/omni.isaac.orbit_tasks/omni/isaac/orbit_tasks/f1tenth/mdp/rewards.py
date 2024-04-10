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

def lidar_distance_sum(env: RLTaskEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside of the soft joint limits."""
    """The ranges from the given lidar sensor."""
    # extract the used quantities (to enable type-hinting)
    sensor: Lidar = env.scene[sensor_cfg.name]
    lidar_ranges = sensor.data.output
 
    return torch.sum(lidar_ranges, dim=1)

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
    
    print(f"passed_starting_location: {asset.data.root_pos_w[:, :2]}")

    # Check if we've already captured the starting positions for this asset
    if asset_cfg.name not in env.starting_positions:
        # Capture and store the starting positions
        env.starting_positions[asset_cfg.name] = asset.data.root_pos_w[:, :2].clone()

    # Retrieve the starting positions
    starting_positions = env.starting_positions[asset_cfg.name]
    
    # Compute the difference in positions
    position_differences = asset.data.root_pos_w[:, :2] - starting_positions
    
    # Calculate the distance moved from the starting positions
    distance_moved = torch.norm(position_differences, dim=1)
    
    # Check if the assets have moved beyond the threshold from their starting positions
    passed_threshold = distance_moved > threshold
    print(f"passed_starting_location: {passed_threshold}")
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


# Reward for touching target
def touch_target(env: RLTaskEnv, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Reward for touching the target."""
    asset: Articulation = env.scene[asset_cfg.name]
    target: Articulation = env.scene[target_cfg.name]
    
    print(f"target_contact: {target.cfg.collision_group}")
    
    # Check if there is collision between the asset and the target
    collision_threshold = asset.data.root_pos_w[:, :2]
    
    if collision_threshold:
        print("THROAT GOAT")
        return torch.tensor([1.0], device=asset.device)
    else:
        return torch.tensor([0.0], device=asset.device)