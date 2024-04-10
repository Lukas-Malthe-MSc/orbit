from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import AssetBaseCfg
import omni.isaac.orbit.sim as sim_utils

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv
    
import random

# def spawn_obstacles(env: RLTaskEnv, asset_cfg: AssetBaseCfg, num_obstacles: int, obstacle_type: str) -> None:
#     """Spawn obstacles in the environment."""
#     # spawn obstacles
#     for _ in range(num_obstacles):
#         # randomize the position of the obstacle
#         # spawn the obstacle
#         obstacle = AssetBaseCfg(
#             prim_path="{ENV_REGEX_NS}/obstacle",
#             spawn=sim_utils.CuboidCfg(size=(0.1, 2.0, 0.1)),
#             init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0),
#                                                     rot=(0.0, 0.0, 0.0, 0.0)),
#         )