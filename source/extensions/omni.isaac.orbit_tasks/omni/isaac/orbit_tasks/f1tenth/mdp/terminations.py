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

    if env.collision_beams is None:
        env.collision_beams = get_scale_vector()
        env.collision_beams = env.collision_beams.to(lidar_ranges.device)
        print("AYYYYYYY")

    below_limit = lidar_ranges < env.collision_beams
    
    result = torch.any(below_limit, dim=1)
    return result

    # return torch.any(lidar_ranges < distance_threshold, dim=1)


# def get_scale_vector(env: RLTaskEnv,
#                      width: float, 
#                      length: float, 
#                      num_beams: int, 
#                      fov: float
#                      ):
def get_scale_vector(width=0.145, length=0.18, num_beams=1081, fov=1.5*torch.pi):
    # Rotate beams
    shift = -(2*torch.pi - fov) / 2
    angles = torch.linspace(shift, fov + shift, steps=num_beams)

    # Angle to the corner of the car
    # angle_radians = torch.atan2(length, width)
    angle_radians = torch.atan2(torch.tensor(length), torch.tensor(width))

    scaled_vec = []

    for theta in angles:
        if abs(theta) > angle_radians and abs(theta) < torch.pi - angle_radians:
            # Beam points to the front of the car
            scale = length * torch.sqrt(1 + (torch.cos(theta) / torch.sin(theta))**2)
        else:
            # Beam points to the side or rear of the car
            scale = width * torch.sqrt(1 + (torch.sin(theta) / torch.cos(theta))**2)

        scaled_vec.append(scale)

    return torch.tensor(scaled_vec)

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




