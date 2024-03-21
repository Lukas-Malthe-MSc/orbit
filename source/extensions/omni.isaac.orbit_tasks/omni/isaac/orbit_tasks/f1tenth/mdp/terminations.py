from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import wrap_to_pi
from omni.isaac.orbit.sensors import Lidar

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def lidar_distance_limit(env: RLTaskEnv, distance_threshold, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside of the soft joint limits."""
    """The ranges from the given lidar sensor."""
    # extract the used quantities (to enable type-hinting)
    sensor: Lidar = env.scene[sensor_cfg.name]
    lidar_ranges = sensor.data.output
    # print(sensor.)
    # if torch.any(lidar_ranges < distance_threshold, dim=1):
    #     print("TERMINATING!!!")
    return torch.any(lidar_ranges < distance_threshold, dim=1)

def bad_orientation(
    env: RLTaskEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # if (torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle):
    #     print(f"bad orientation: {torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle}")
    return torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle