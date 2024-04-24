from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import AssetBaseCfg, RigidObjectCfg, Articulation, RigidObject, AssetBase
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.core.utils.prims import delete_prim, create_prim
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.orbit.utils.math import quat_from_euler_xyz, sample_uniform
from omni.physx.scripts import utils as physx_utils

import random
from typing import Tuple, Dict


if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv, BaseEnv

    
def randomize_map(
    env: RLTaskEnv,
    env_ids: torch.Tensor,
    maps_paths: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the maps for multiple environments by selecting random ones from a list of maps."""
    # print(env.common_step_counter)
    if not env.maps_is_randomized:
        for i in range (env.num_envs):

            # num_envs = env.num_envs
            # select random environemt id
            # env_id = int(random.uniform(0, num_envs))
            env_id = i

            print(f"Randomizing map for environment {env_id}")
            # Construct the root prim path for the current environment
            map_root_prim = f"/World/envs/env_{env_id}/RaceTrack"  # Adjust based on your scene structure

            # Clear existing maps for the current environment
            delete_prim(map_root_prim)  # Remove existing map objects to avoid conflicts

            # Select a random map path from the provided list
            random_map_path = random.choice(maps_paths)

            # Create a new prim with the selected map
            create_prim(
                prim_path=map_root_prim,  # Adjust to your map root path
                prim_type="Xform",  # You can change this depending on your object type
                usd_path=random_map_path,  # The USD file to reference
                scale=(0.015, 0.015, 0.015),  # Scale of the object
            )
        env.maps_is_randomized = True
    
    

def reset_root_state_uniform(
    env: BaseEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # positions
    pos_offset = torch.zeros_like(root_states[:, 0:3])
    pos_offset[:, 0].uniform_(*pose_range.get("x", (0.0, 0.0)))
    pos_offset[:, 1].uniform_(*pose_range.get("y", (0.0, 0.0)))
    pos_offset[:, 2].uniform_(*pose_range.get("z", (0.0, 0.0)))
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pos_offset
    # orientations
    euler_angles = torch.zeros_like(positions)
    euler_angles[:, 0].uniform_(*pose_range.get("roll", (0.0, 0.0)))
    euler_angles[:, 1].uniform_(*pose_range.get("pitch", (0.0, 0.0)))
    euler_angles[:, 2].uniform_(*pose_range.get("yaw", (0.0, 0.0)))
    orientations = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
    # velocities
    velocities = root_states[:, 7:13]
    velocities[:, 0].uniform_(*velocity_range.get("x", (0.0, 0.0)))
    velocities[:, 1].uniform_(*velocity_range.get("y", (0.0, 0.0)))
    velocities[:, 2].uniform_(*velocity_range.get("z", (0.0, 0.0)))
    velocities[:, 3].uniform_(*velocity_range.get("roll", (0.0, 0.0)))
    velocities[:, 4].uniform_(*velocity_range.get("pitch", (0.0, 0.0)))
    velocities[:, 5].uniform_(*velocity_range.get("yaw", (0.0, 0.0)))
    # randomize_map(env, env_ids, maps_paths=[f"omniverse://localhost/Projects/f1tenth/maps/track_1.usd",
    #                                         f"omniverse://localhost/Projects/f1tenth/maps/track_2.usd",
    #                                         f"omniverse://localhost/Projects/f1tenth/maps/track_3.usd",
    #                                         f"omniverse://localhost/Projects/f1tenth/maps/track_4.usd",
    #                                         f"omniverse://localhost/Projects/f1tenth/maps/track_5.usd"
    #                                         ])
    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


