# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

from __future__ import annotations

import omni.isaac.orbit.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

CONE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(18.0, 18.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.0,
    vertical_scale=0.00,
    slope_threshold=0.0,
    difficulty_choices=(0.5, 0.75, 0.9),
    use_cache=False,
    sub_terrains={
        # "plane": terrain_gen.MeshPlaneTerrainCfg(
        #     proportion=0.0,
        #     size=(100, 100),
        # ),
        "cones": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            proportion=0.2, object_type="cone", size=(0.05, 0.2), platform_width=0.0,
            object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(num_objects=20, height=0.5, radius=0.1),
            object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(num_objects=20, height=0.5, radius=0.1),
        ),
    },
)
"""Rough terrains configuration."""
    # ground terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/Terrain",
    #     terrain_type="generator",
    #     terrain_generator=CONE_TERRAINS_CFG,
    #     max_init_terrain_level=None,
    #     collision_group=0, #-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     # visual_material=sim_utils.MdlFileCfg(
    #     #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #     #     project_uvw=True,
    #     # ),
    #     debug_vis=False,
    # )