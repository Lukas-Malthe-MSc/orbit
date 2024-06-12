from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.core.utils.prims import delete_prim, create_prim

import random


if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv

    
def randomize_map(
    env: RLTaskEnv,
    env_ids: torch.Tensor, 
    maps_paths: list[str],
    map_prim: str = "map",
    scale: tuple[float, float, float] = (0.015, 0.015, 0.015),
):
    """
    Randomize the maps for each environment by uniform sampling from a list of maps.
    
    NOTE: This only works when mode="startup". Also, there seems to be issues when there is only 1 environment.
    """

    for i in range (env.num_envs):

        print(f"Randomizing map for environment {i}")
        # Construct the root prim path for the current environment
        map_root_prim = f"/World/envs/env_{i}/{map_prim}"  # Adjust based on your scene structure

        # Clear existing maps for the current environment
        delete_prim(map_root_prim)  # Remove existing map objects to avoid conflicts

        # Select a random map path from the provided list
        random_map_path = random.choice(maps_paths)

        # Create a new prim with the selected map
        create_prim(
            prim_path=map_root_prim,  # Adjust to your map root path
            prim_type="Xform",  # You can change this depending on your object type
            usd_path=random_map_path,  # The USD file to reference
            scale=scale,  # Scale of the object
        )

    