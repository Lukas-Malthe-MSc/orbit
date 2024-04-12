from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import AssetBaseCfg, RigidObjectCfg, Articulation, RigidObject, AssetBase
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv, BaseEnv
    
import random

# randomize obstacle position
def randomize_obstacle_position(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("cone1")):
    obstacle: RigidObjectCfg = env.scene[asset_cfg.name]

    obstacle.InitialStateCfg().pos = (random.uniform(-5, 5), random.uniform(-5, 5), 0)
    