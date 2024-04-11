# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""

from __future__ import annotations

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.assets import ArticulationCfg
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR

from pathlib import Path

current_working_directory = Path.cwd()

# F1TENTH_PROJECT_DIR = "omniverse://localhost/Projects/f1tenth"
F1TENTH_PROJECT_DIR = current_working_directory

F1TENTH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{F1TENTH_PROJECT_DIR}/f1tenth_assets/omniverse/robot/instanceable/f1tenth.usd",
        # usd_path=f"{F1TENTH_PROJECT_DIR}/robot/instanceable/f1tenth.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5)
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=['wheel_.*'],
            effort_limit=None, #50,
            velocity_limit=None, #25,
            stiffness=None, #1000,
            damping=None#175
        ),
        "rotators": ImplicitActuatorCfg(
            joint_names_expr=["rotator_.*"],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None, # 0,
            damping=None, #175
        ),
    },
)
