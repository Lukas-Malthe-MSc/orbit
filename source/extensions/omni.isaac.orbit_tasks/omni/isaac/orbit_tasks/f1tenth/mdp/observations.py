# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from rich import print

import torch
from typing import TYPE_CHECKING

import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import Lidar
from omni.isaac.orbit.assets import RigidObject

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv
    
def lidar_ranges(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The ranges from the given lidar sensor."""
    # extract the used quantities (to enable type-hinting)
    sensor: Lidar = env.scene.sensors[sensor_cfg.name]
    lidar_ranges = sensor.data.output
    # print(f", num_sensors: {lidar_ranges.shape}")
    return lidar_ranges

def lidar_ranges_normalized(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return normalized lidar ranges with Gaussian noise."""
    # Extract the lidar sensor from the scene
    sensor: Lidar = env.scene.sensors[sensor_cfg.name]
    lidar_ranges = sensor.data.output  # Original lidar ranges

    # Get the min and max range from the sensor configuration
    min_range = sensor.cfg.min_range  # Minimum possible range
    max_range = sensor.cfg.max_range  # Maximum possible range

    # Normalize the lidar data
    lidar_ranges_normalized = (lidar_ranges - min_range) / (max_range - min_range)

    # Generate Gaussian noise with the same shape as the lidar data
    mean = 0.0  # Mean of the Gaussian distribution
    std = 0.1  # Standard deviation of the Gaussian distribution
    gaussian_noise = torch.normal(mean=mean, std=std, size=lidar_ranges_normalized.shape, device=lidar_ranges_normalized.device)

    # Apply noise to the normalized lidar data
    lidar_ranges_normalized_noisy = lidar_ranges_normalized + gaussian_noise

    # Clip values to maintain the [0, 1] range
    lidar_ranges_normalized_noisy = torch.clip(lidar_ranges_normalized_noisy, 0.0, 1.0)

    return lidar_ranges_normalized_noisy



def base_pos(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root position in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w

def base_rot(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root rotation in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_quat_w


def base_lin_vel(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, :2]


def base_ang_vel(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, :2]