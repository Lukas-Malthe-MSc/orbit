# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

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
    # print(f"forward_velocity: {asset.data.root_lin_vel_b[:, 0]}")
    # return asset.data.root_lin_vel_b[:, 0]

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