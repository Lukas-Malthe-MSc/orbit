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
    return torch.sum(torch.square(joint_pos - target), dim=1)

def action_target_log_l1(env: RLTaskEnv, action_idx, lambda_1: float, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize action deviation from a target value."""
    action = env.action_manager.action[:, action_idx]
    
    return torch.log10(lambda_1 * torch.abs(action - target) + 1)
    

def smooth_action_l2(env: RLTaskEnv, action_idx, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize current action for being different from previous action."""
    action = env.action_manager.action[:, action_idx]
    prev_action = env.action_manager.prev_action[:, action_idx]
        
    return ((action - prev_action) ** 2)


def forward_velocity(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.max(asset.data.root_lin_vel_b[:, 0], torch.zeros(asset.data.root_lin_vel_b[:, 0].shape, device=asset.device))


def lidar_min_distance(env: RLTaskEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The min distance of the lidar scans."""
    sensor: Lidar = env.scene[sensor_cfg.name]
    lidar_ranges = sensor.data.output
    min_distances = torch.min(lidar_ranges, dim=1).values
    return 1/min_distances



def timed_lap_time( env:            RLTaskEnv, 
                    asset_cfg:      SceneEntityCfg, 
                    threshold:      float, 
                    lap_threshold:  float = 5.0
                    ) -> torch.Tensor:
    """Checks if assets are within their starting locations +- a threshold and prints time elapsed since leaving and returning for each asset."""
    # Access the asset
    asset: Articulation = env.scene[asset_cfg.name]
    num_assets = asset.data.root_pos_w.shape[0]  # Assuming this gives the number of individual assets
    
    # If the environment has not started yet, return zeros
    if(env.common_step_counter < 5):
        return torch.zeros(num_assets, device=asset.data.root_pos_w.device) #indicating not started yet
    
    # Initialize tracking if not already done
    if asset_cfg.name not in env.starting_positions:
        env.starting_positions[asset_cfg.name] = {
            "position": asset.data.root_pos_w[:, :2].clone(),
            "left_at_step": torch.full((num_assets,), -1, dtype=torch.int64),
            "returned_at_step": torch.full((num_assets,), -1, dtype=torch.int64),
            "is_lap_completed": torch.full((num_assets,), -1, dtype=torch.int64),
            "lap_times": []  # Buffer to store lap times
        }
    
    tracking_info = env.starting_positions[asset_cfg.name]
    position_differences = asset.data.root_pos_w[:, :2] - tracking_info["position"]
    distance_moved = torch.norm(position_differences, dim=1)
    within_threshold = distance_moved <= threshold

    for i in range(num_assets):
        if tracking_info["is_lap_completed"][i] == -1 and distance_moved[i] > lap_threshold:
            tracking_info["is_lap_completed"][i] = 1
            print(f"Car {i} has left limit")

        if within_threshold[i] and tracking_info["left_at_step"][i] >= 0 and tracking_info["is_lap_completed"][i]:
            tracking_info["returned_at_step"][i] = env.common_step_counter
            time_elapsed = (tracking_info["returned_at_step"][i] - tracking_info["left_at_step"][i]) * env.physics_dt
            print(f"Asset {i}: Time elapsed since leaving and returning: {time_elapsed} seconds, dt={env.physics_dt}")
            tracking_info["lap_times"].append(time_elapsed.item())
            tracking_info["left_at_step"][i] = -1
            tracking_info["is_lap_completed"][i] = -1

        elif not within_threshold[i] and tracking_info["left_at_step"][i] < 0:
            tracking_info["left_at_step"][i] = env.common_step_counter

    return within_threshold