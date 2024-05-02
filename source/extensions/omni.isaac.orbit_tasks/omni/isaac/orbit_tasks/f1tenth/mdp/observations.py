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

from threading import Lock
import numpy as np
import copy

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

    # Generate Gaussian noise with the same shape as the lidar data
    mean = 0.0  # Mean of the Gaussian distribution
    std = 0.1  # Standard deviation of the Gaussian distribution
    gaussian_noise = torch.normal(mean=mean, std=std, size=lidar_ranges.shape, device=lidar_ranges.device)

    lidar_ranges_noisy = lidar_ranges + gaussian_noise # Add noise to the lidar data
    
    lidar_ranges_noisy = torch.clip(lidar_ranges_noisy, min=min_range, max=max_range) # Clip the noisy lidar data to the min and max range

    # Normalize the lidar data
    lidar_ranges_normalized = (lidar_ranges - min_range) / (max_range - min_range)

    return lidar_ranges_normalized 



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


def base_lin_vel_xy_dot(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # For logging purposes
    # data_logger = DataLogger()
    # data_logger.log_data(asset.data.root_pos_w)
    # data_logger._counter += 1
    
    # if data_logger._counter % 100 == 0:
    #     data_logger.write_data(filename="data-analysis/data/pos_data_log_pen.npy")
    return asset.data.root_lin_vel_b[:, :2]


def base_ang_vel_yaw_dot(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 2:]




class DataLogger:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._data = []
                cls._instance._counter = 0
            return cls._instance
        
    def log_data(self, data):
        # Create a deep copy of the data
        data_copy = copy.deepcopy(data)
        self._instance._data.append(data_copy)
        
    def get_data(self):
        return self._instance._data
    
    def write_data(self, filename):
        np.save(filename, torch.stack(self._instance._data).cpu().numpy())