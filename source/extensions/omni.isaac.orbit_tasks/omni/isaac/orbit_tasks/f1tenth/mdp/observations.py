from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import Lidar

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv
    
def lidar_ranges(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The ranges from the given lidar sensor."""
    # extract the used quantities (to enable type-hinting)
    sensor: Lidar = env.scene.sensors[sensor_cfg.name]

    lidar_ranges = sensor.data.output["linear_depth"]

    return lidar_ranges

def lidar_ranges_normalized(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return normalized lidar ranges with Gaussian noise."""
    # Extract the lidar sensor from the scene
    sensor: Lidar = env.scene.sensors[sensor_cfg.name]

    lidar_ranges = sensor.data.output["linear_depth"]

    # Get the min and max range from the sensor configuration
    min_range = sensor.cfg.min_range  # Minimum possible range
    max_range = sensor.cfg.max_range  # Maximum possible range

    # Generate Gaussian noise with the same shape as the lidar data
    mean = 0.0  # Mean of the Gaussian distribution
    std = 0.1  # Standard deviation of the Gaussian distribution
    gaussian_noise = torch.normal(mean=mean, std=std, size=lidar_ranges.shape, device=lidar_ranges.device)

    # Add noise to the lidar data
    lidar_ranges_noisy = lidar_ranges + gaussian_noise

    # Clip the noisy lidar data to the min and max range
    lidar_ranges_noisy = torch.clip(lidar_ranges_noisy, min=min_range, max=max_range)

    # Normalize the noisy lidar data
    lidar_ranges_normalized = (lidar_ranges_noisy - min_range) / (max_range - min_range)

    return lidar_ranges_normalized


def base_lin_vel_xy_dot(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Linear velocies (x_dot, y_dot) in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, :2]


def base_ang_vel_yaw_dot(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Angular velocity (psi_dot) in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 2:]


def last_processed_action(env: BaseEnv, bounding_strategy: str | None, scale, offset) -> torch.Tensor:
    """
    The last input action to the environment.

    Parameters:
    ----------
    env : BaseEnv
        The environment containing the action manager and actions.
    bounding_strategy : str | None
        The strategy used to bound the actions. Can be 'clip' or 'tanh'. If None, no bounding is applied.
    scale : tuple[float, float]
        The scaling factor applied to the input action.
    offset : tuple[float, float]
        The offset applied to the input action.

    Returns:
    -------
    torch.Tensor
        The processed actions.
    """
    _scale = torch.tensor(scale, device=env.action_manager.action.device, dtype=torch.float32)
    _offset = torch.tensor(offset, device=env.action_manager.action.device, dtype=torch.float32)

    actions = env.action_manager.action

    if bounding_strategy == 'clip':
        _processed_actions = torch.clip(actions, min=-1.0, max=1.0) * _scale + _offset
        
    elif bounding_strategy == 'tanh':
        _processed_actions = torch.tanh(actions) * _scale + _offset
        
    else:
        _processed_actions = actions * _scale + _offset
        
    return _processed_actions