from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import wrap_to_pi, euler_xyz_from_quat
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


def flipped_over(
    env: RLTaskEnv,
    asset_cfg: SceneEntityCfg,
    roll_threshold=2.0,
    pitch_threshold=2.0
) -> torch.Tensor:
    """Determine which environments contain flipped robots based on Euler angles."""

    # Get the RigidObject instance for the given asset config
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get the quaternion from the environment
    quaternions = asset.data.root_quat_w

    # Convert the quaternion to Euler angles
    roll, pitch, yaw = euler_xyz_from_quat(quaternions)

    # Wrap Euler angles to [-pi, pi] to avoid unexpected results
    roll = wrap_to_pi(roll)
    pitch = wrap_to_pi(pitch)

    # Determine if roll or pitch exceeds given thresholds
    is_flipped = (
        (roll > roll_threshold) | (roll < -roll_threshold) |
        (pitch > pitch_threshold) | (pitch < -pitch_threshold)
    )

    # Return tensor indicating which environments contain flipped robots
    return is_flipped