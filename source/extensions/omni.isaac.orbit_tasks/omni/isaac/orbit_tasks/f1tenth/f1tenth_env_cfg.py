# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
from omni.isaac.orbit.sensors.lidar.lidar_cfg import LidarCfg
from rich import print

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.orbit.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from omni.isaac.orbit.terrains import TerrainImporterCfg
import omni.isaac.orbit_tasks.f1tenth.mdp as mdp
from omni.isaac.orbit.managers import CurriculumTermCfg as CurrTerm
##
# Pre-defined configs
##
from omni.isaac.orbit_assets.f1tenth import F1TENTH_CFG  # isort:skip
from omni.isaac.orbit.terrains.config.cones import CONE_TERRAINS_CFG  # isort: skip
##
# Scene definition
##

from pathlib import Path

current_working_directory = Path.cwd()                                                                                              

"""
Train commmand:
$ ./orbit.sh -p source/standalone/workflows/rsl_rl/train.py --task F1tenth-v0 --headless --offscreen_render --num_envs 4096

Play command:
$ ./orbit.sh -p source/standalone/workflows/rsl_rl/play.py --task F1tenth-v0 --num_envs 1 --load_run 2024-04-24_11-11-10 --checkpoint model_49.pt

"""

@configclass
class F1tenthSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0),
                                       color=(0.5, 0.5, 0.5)),   
    )
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )
    
    # f1tenth
    robot: ArticulationCfg = F1TENTH_CFG.replace(prim_path="{ENV_REGEX_NS}/f1tenth")

    # TODO: Ensure that lidar sensor is correctly configured
    # sensors
    lidar: LidarCfg = LidarCfg(
        prim_path="{ENV_REGEX_NS}/f1tenth/hokuyo_1/Lidar",
        # update_period=0.025,  # Update rate of 40Hz
        # data_types=["point_cloud"],  # Assuming the LiDAR generates point cloud data
        horizontal_fov=270.0,  # Horizontal field of view of 270 degrees
        horizontal_resolution=0.2497,  # Horizontal resolution of 0.5 degrees
        max_range=30.0,  # Maximum range of 30 meters
        min_range=0.02,  # Minimum range of 0.1 meters
        rotation_rate=0.0,  # Rotation rate of 0.0 radians per second
        offset=LidarCfg.OffsetCfg(
            pos=(0.11749, 0.0, 0.1),  # Example position offset from the robot base
            
            rot=(1.0, 0.0, 0.0, 0.0),  # Example rotation offset; no rotation in this case
            convention="ros"  # Frame convention
        ),
        draw_lines=False,
        draw_points=False,
    )

    race_track: AssetBaseCfg = AssetBaseCfg( 
        prim_path="{ENV_REGEX_NS}/RaceTrack",
        collision_group=0,
        spawn=sim_utils.UsdFileCfg(
            # usd_path="omniverse://localhost/Projects/f1tenth/box.usd",
            usd_path= f"{current_working_directory}/f1tenth_assets/omniverse/maps/track_2.usd",
            # usd_path="omniverse://localhost/Projects/f1tenth/maps/track_2.usd",
            scale=(.015, .015, .015),
        )
    )
    
    # box1: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/box1",
    #     spawn=sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5), 
    #                             rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
    #                             collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25),
    #                                             rot=(1.0, 0.0, 0.0, 0.0)),
    # )
    
    # box2: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/box2",
    #     spawn=sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5),
    #                             rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
    #                             collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25),
    #                                             rot=(1.0, 0.0, 0.0, 0.0)),
    # )
    
    # box3: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/box3",
    #     spawn=sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5),
    #                             rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
    #                             collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25),
    #                                             rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    # box4: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/box4",
    #     spawn=sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5),
    #                             rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
    #                             collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25),
    #                                             rot=(1.0, 0.0, 0.0, 0.0)),
    # )
    
    # box5: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/box5",
    #     spawn=sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5),
    #                             rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
    #                             collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25),
    #                                             rot=(1.0, 0.0, 0.0, 0.0)),
    # )
    
    # box6: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/box6",
    #     spawn=sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5),
    #                             rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
    #                             collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25),
    #                                             rot=(1.0, 0.0, 0.0, 0.0)),
    # )
    
    # box7: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/box7",
    #     spawn=sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5),
    #                             rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
    #                             collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25),
    #                                             rot=(1.0, 0.0, 0.0, 0.0)),
    # )
    
    # box8: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/box8",
    #     spawn=sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5),
    #                             rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
    #                             collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25),
    #                                             rot=(1.0, 0.0, 0.0, 0.0)),
    # )
    
    # box9: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/box9",
    #     spawn=sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5),
    #                             rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
    #                             collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25),
    #                                             rot=(1.0, 0.0, 0.0, 0.0)),
    # )
    
    # box10: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/box10",
    #     spawn=sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5),
    #                             rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
    #                             collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25),
    #                                             rot=(1.0, 0.0, 0.0, 0.0)),
    # )

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    #   
    ackermann_action = mdp.AckermannActionCfg(asset_name="robot", 
                                  wheel_joint_names=["wheel_back_left", "wheel_back_right", "wheel_front_left", "wheel_front_right"], 
                                  steering_joint_names=["rotator_left", "rotator_right"], 
                                  base_width=0.25, base_length=0.35, wheel_radius=0.05, scale=(2.5, torch.pi/4), offset=(0.0, 0.0)) #TODO: adjust max speed

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # lidar_ranges = ObsTerm(func=mdp.lidar_ranges, noise=Gnoise(mean=0.0, std=0.1), params={"sensor_cfg": SceneEntityCfg("lidar")})
        lidar_ranges_normalized = ObsTerm(func=mdp.lidar_ranges_normalized, params={"sensor_cfg": SceneEntityCfg("lidar")})
        # observation terms (order preserved)
        # base_pos = ObsTerm(func=mdp.base_pos, noise=Unoise(n_min=-0.1, n_max=0.1))
        # base_rot = ObsTerm(func=mdp.base_rot, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Gnoise(mean=0.0, std=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Gnoise(mean=0.0, std=0.1))
        
        last_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RandomizationCfg:
    """Configuration for randomization."""
    
    # startup
    add_base_mass = RandTerm(
        func=mdp.add_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"), "mass_range": (-5.0, 5.0)},
    )
    
    randomize_map = RandTerm(
        func=mdp.randomize_map,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "maps_paths": [f"{current_working_directory}/f1tenth_assets/omniverse/maps/track_1.usd",
                           f"{current_working_directory}/f1tenth_assets/omniverse/maps/track_2.usd",
                           f"{current_working_directory}/f1tenth_assets/omniverse/maps/track_3.usd",
                           f"{current_working_directory}/f1tenth_assets/omniverse/maps/track_4.usd",
                           f"{current_working_directory}/f1tenth_assets/omniverse/maps/track_5.usd"],
            },
            # "maps_paths": [f"omniverse://localhost/Projects/f1tenth/maps/track_1.usd",
            #                f"omniverse://localhost/Projects/f1tenth/maps/track_2.usd",
            #                f"omniverse://localhost/Projects/f1tenth/maps/track_3.usd",
            #                f"omniverse://localhost/Projects/f1tenth/maps/track_4.usd",
            #                f"omniverse://localhost/Projects/f1tenth/maps/track_5.usd"],
            # },
    )
    
    physics_material = RandTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
    # reset
    reset_root_state = RandTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "pose_range": {
                "x": (-5.0, 5.0),  # X position range from -5 to 5
                "y": (-5.0, 5.0),  # Y position range from -5 to 5
                "z": (0.0, 0.75),   # Z position range from 0 to 2 (assuming starting on the ground)
                "roll": (0.0, 0.0),  # Roll orientation range from -pi to pi
                "pitch": (0.0, 0.0), # Pitch orientation range from -pi to pi
                "yaw": (-3.14, 3.14),   # Yaw orientation range from -pi to pi
            }, 
            "velocity_range": {
                "x": (-1.0, 1.0),  # X linear velocity range from -1 to 1
                "y": (-1.0, 1.0),  # Y linear velocity range from -1 to 1
                "z": (0.0, 0.0),  # Z linear velocity range from -1 to 1 (upwards/downwards movement)
                "roll": (-0.5, 0.5),  # Roll angular velocity range
                "pitch": (-0.5, 0.5), # Pitch angular velocity range
                "yaw": (-0.5, 0.5),   # Yaw angular velocity range
            }     
        },
    )
    
    # randomize_obstacle1_position = RandTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("box1"), 
    #         "pose_range": {
    #             "x": (-8.0, 8.0),  # X position range from -5 to 5
    #             "y": (-8.0, 8.0),  # Y position range from -5 to 5
    #             "z": (0.0, 0.75),   # Z position range from 0 to 2 (assuming starting on the ground)
    #             "roll": (0.0, 0.0),  # Roll orientation range from -pi to pi
    #             "pitch": (0.0, 0.0), # Pitch orientation range from -pi to pi
    #             "yaw": (-3.14, 3.14),   # Yaw orientation range from -pi to pi
    #         }, 
    #         "velocity_range": {
    #             "x": (0.0, 0.0),  # X linear velocity range from -1 to 1
    #             "y": (0.0, 0.0),  # Y linear velocity range from -1 to 1
    #             "z": (0.0, 0.0),  # Z linear velocity range from -1 to 1 (upwards/downwards movement)
    #             "roll": (0.0, 0.0),  # Roll angular velocity range
    #             "pitch": (0.0, 0.0), # Pitch angular velocity range
    #             "yaw": (0.0, 0.0),   # Yaw angular velocity range
    #         }     
    #     },
    # )
    
    # randomize_obstacle2_position = RandTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("box2"), 
    #         "pose_range": {
    #             "x": (-8.0, 8.0),  # X position range from -5 to 5
    #             "y": (-8.0, 8.0),  # Y position range from -5 to 5
    #             "z": (0.0, 0.75),   # Z position range from 0 to 2 (assuming starting on the ground)
    #             "roll": (0.0, 0.0),  # Roll orientation range from -pi to pi
    #             "pitch": (0.0, 0.0), # Pitch orientation range from -pi to pi
    #             "yaw": (-3.14, 3.14),   # Yaw orientation range from -pi to pi
    #         }, 
    #         "velocity_range": {
    #             "x": (0.0, 0.0),  # X linear velocity range from -1 to 1
    #             "y": (0.0, 0.0),  # Y linear velocity range from -1 to 1
    #             "z": (0.0, 0.0),  # Z linear velocity range from -1 to 1 (upwards/downwards movement)
    #             "roll": (0.0, 0.0),  # Roll angular velocity range
    #             "pitch": (0.0, 0.0), # Pitch angular velocity range
    #             "yaw": (0.0, 0.0),   # Yaw angular velocity range
    #         }     
    #     },
    # )

    # randomize_obstacle3_position = RandTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("box3"), 
    #         "pose_range": {
    #             "x": (-8.0, 8.0),  # X position range from -5 to 5
    #             "y": (-8.0, 8.0),  # Y position range from -5 to 5
    #             "z": (0.0, 0.75),   # Z position range from 0 to 2 (assuming starting on the ground)
    #             "roll": (0.0, 0.0),  # Roll orientation range from -pi to pi
    #             "pitch": (0.0, 0.0), # Pitch orientation range from -pi to pi
    #             "yaw": (-3.14, 3.14),   # Yaw orientation range from -pi to pi
    #         }, 
    #         "velocity_range": {
    #             "x": (0.0, 0.0),  # X linear velocity range from -1 to 1
    #             "y": (0.0, 0.0),  # Y linear velocity range from -1 to 1
    #             "z": (0.0, 0.0),  # Z linear velocity range from -1 to 1 (upwards/downwards movement)
    #             "roll": (0.0, 0.0),  # Roll angular velocity range
    #             "pitch": (0.0, 0.0), # Pitch angular velocity range
    #             "yaw": (0.0, 0.0),   # Yaw angular velocity range
    #         }     
    #     },
    # )

    # randomize_obstacle4_position = RandTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("box4"), 
    #         "pose_range": {
    #             "x": (-8.0, 8.0),  # X position range from -5 to 5
    #             "y": (-8.0, 8.0),  # Y position range from -5 to 5
    #             "z": (0.0, 0.75),   # Z position range from 0 to 2 (assuming starting on the ground)
    #             "roll": (0.0, 0.0),  # Roll orientation range from -pi to pi
    #             "pitch": (0.0, 0.0), # Pitch orientation range from -pi to pi
    #             "yaw": (-3.14, 3.14),   # Yaw orientation range from -pi to pi
    #         }, 
    #         "velocity_range": {
    #             "x": (0.0, 0.0),  # X linear velocity range from -1 to 1
    #             "y": (0.0, 0.0),  # Y linear velocity range from -1 to 1
    #             "z": (0.0, 0.0),  # Z linear velocity range from -1 to 1 (upwards/downwards movement)
    #             "roll": (0.0, 0.0),  # Roll angular velocity range
    #             "pitch": (0.0, 0.0), # Pitch angular velocity range
    #             "yaw": (0.0, 0.0),   # Yaw angular velocity range
    #         }     
    #     },
    # )

    # randomize_obstacle5_position = RandTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("box5"), 
    #         "pose_range": {
    #             "x": (-8.0, 8.0),  # X position range from -5 to 5
    #             "y": (-8.0, 8.0),  # Y position range from -5 to 5
    #             "z": (0.0, 0.75),   # Z position range from 0 to 2 (assuming starting on the ground)
    #             "roll": (0.0, 0.0),  # Roll orientation range from -pi to pi
    #             "pitch": (0.0, 0.0), # Pitch orientation range from -pi to pi
    #             "yaw": (-3.14, 3.14),   # Yaw orientation range from -pi to pi
    #         }, 
    #         "velocity_range": {
    #             "x": (0.0, 0.0),  # X linear velocity range from -1 to 1
    #             "y": (0.0, 0.0),  # Y linear velocity range from -1 to 1
    #             "z": (0.0, 0.0),  # Z linear velocity range from -1 to 1 (upwards/downwards movement)
    #             "roll": (0.0, 0.0),  # Roll angular velocity range
    #             "pitch": (0.0, 0.0), # Pitch angular velocity range
    #             "yaw": (0.0, 0.0),   # Yaw angular velocity range
    #         }     
    #     },
    # )

    # randomize_obstacle6_position = RandTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("box6"), 
    #         "pose_range": {
    #             "x": (-8.0, 8.0),  # X position range from -5 to 5
    #             "y": (-8.0, 8.0),  # Y position range from -5 to 5
    #             "z": (0.0, 0.75),   # Z position range from 0 to 2 (assuming starting on the ground)
    #             "roll": (0.0, 0.0),  # Roll orientation range from -pi to pi
    #             "pitch": (0.0, 0.0), # Pitch orientation range from -pi to pi
    #             "yaw": (-3.14, 3.14),   # Yaw orientation range from -pi to pi
    #         }, 
    #         "velocity_range": {
    #             "x": (0.0, 0.0),  # X linear velocity range from -1 to 1
    #             "y": (0.0, 0.0),  # Y linear velocity range from -1 to 1
    #             "z": (0.0, 0.0),  # Z linear velocity range from -1 to 1 (upwards/downwards movement)
    #             "roll": (0.0, 0.0),  # Roll angular velocity range
    #             "pitch": (0.0, 0.0), # Pitch angular velocity range
    #             "yaw": (0.0, 0.0),   # Yaw angular velocity range
    #         }     
    #     },
    # )
    
    # randomize_obstacle7_position = RandTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("box7"), 
    #         "pose_range": {
    #             "x": (-8.0, 8.0),  # X position range from -5 to 5
    #             "y": (-8.0, 8.0),  # Y position range from -5 to 5
    #             "z": (0.0, 0.75),   # Z position range from 0 to 2 (assuming starting on the ground)
    #             "roll": (0.0, 0.0),  # Roll orientation range from -pi to pi
    #             "pitch": (0.0, 0.0), # Pitch orientation range from -pi to pi
    #             "yaw": (-3.14, 3.14),   # Yaw orientation range from -pi to pi
    #         }, 
    #         "velocity_range": {
    #             "x": (0.0, 0.0),  # X linear velocity range from -1 to 1
    #             "y": (0.0, 0.0),  # Y linear velocity range from -1 to 1
    #             "z": (0.0, 0.0),  # Z linear velocity range from -1 to 1 (upwards/downwards movement)
    #             "roll": (0.0, 0.0),  # Roll angular velocity range
    #             "pitch": (0.0, 0.0), # Pitch angular velocity range
    #             "yaw": (0.0, 0.0),   # Yaw angular velocity range
    #         }     
    #     },
    # )

    # randomize_obstacle8_position = RandTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("box8"), 
    #         "pose_range": {
    #             "x": (-8.0, 8.0),  # X position range from -5 to 5
    #             "y": (-8.0, 8.0),  # Y position range from -5 to 5
    #             "z": (0.0, 0.75),   # Z position range from 0 to 2 (assuming starting on the ground)
    #             "roll": (0.0, 0.0),  # Roll orientation range from -pi to pi
    #             "pitch": (0.0, 0.0), # Pitch orientation range from -pi to pi
    #             "yaw": (-3.14, 3.14),   # Yaw orientation range from -pi to pi
    #         }, 
    #         "velocity_range": {
    #             "x": (0.0, 0.0),  # X linear velocity range from -1 to 1
    #             "y": (0.0, 0.0),  # Y linear velocity range from -1 to 1
    #             "z": (0.0, 0.0),  # Z linear velocity range from -1 to 1 (upwards/downwards movement)
    #             "roll": (0.0, 0.0),  # Roll angular velocity range
    #             "pitch": (0.0, 0.0), # Pitch angular velocity range
    #             "yaw": (0.0, 0.0),   # Yaw angular velocity range
    #         }     
    #     },
    # )

    # randomize_obstacle9_position = RandTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("box9"), 
    #         "pose_range": {
    #             "x": (-8.0, 8.0),  # X position range from -5 to 5
    #             "y": (-8.0, 8.0),  # Y position range from -5 to 5
    #             "z": (0.0, 0.75),   # Z position range from 0 to 2 (assuming starting on the ground)
    #             "roll": (0.0, 0.0),  # Roll orientation range from -pi to pi
    #             "pitch": (0.0, 0.0), # Pitch orientation range from -pi to pi
    #             "yaw": (-3.14, 3.14),   # Yaw orientation range from -pi to pi
    #         }, 
    #         "velocity_range": {
    #             "x": (0.0, 0.0),  # X linear velocity range from -1 to 1
    #             "y": (0.0, 0.0),  # Y linear velocity range from -1 to 1
    #             "z": (0.0, 0.0),  # Z linear velocity range from -1 to 1 (upwards/downwards movement)
    #             "roll": (0.0, 0.0),  # Roll angular velocity range
    #             "pitch": (0.0, 0.0), # Pitch angular velocity range
    #             "yaw": (0.0, 0.0),   # Yaw angular velocity range
    #         }     
    #     },
    # )

    # randomize_obstacle10_position = RandTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("box10"), 
    #         "pose_range": {
    #             "x": (-8.0, 8.0),  # X position range from -5 to 5
    #             "y": (-8.0, 8.0),  # Y position range from -5 to 5
    #             "z": (0.0, 0.75),   # Z position range from 0 to 2 (assuming starting on the ground)
    #             "roll": (0.0, 0.0),  # Roll orientation range from -pi to pi
    #             "pitch": (0.0, 0.0), # Pitch orientation range from -pi to pi
    #             "yaw": (-3.14, 3.14),   # Yaw orientation range from -pi to pi
    #         }, 
    #         "velocity_range": {
    #             "x": (0.0, 0.0),  # X linear velocity range from -1 to 1
    #             "y": (0.0, 0.0),  # Y linear velocity range from -1 to 1
    #             "z": (0.0, 0.0),  # Z linear velocity range from -1 to 1 (upwards/downwards movement)
    #             "roll": (0.0, 0.0),  # Roll angular velocity range
    #             "pitch": (0.0, 0.0), # Pitch angular velocity range
    #             "yaw": (0.0, 0.0),   # Yaw angular velocity range
    #         }     
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    # alive = RewTerm(func=mdp.is_alive, weight=0.5)
    # # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-10.0)
    
    # -- Task: Drive forward
    forward_velocity_reward = RewTerm(func=mdp.forward_velocity, weight=1.0)
    # distance_traveled_reward = RewTerm(func=mdp.distance_traveled_reward, weight=1.0, params={"asset_cfg": SceneEntityCfg("robot")})

    # -- Penalty
    steering_angle_position = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=['rotator_left', 'rotator_right']), "target": 0.0}
    )
    
    # -- Penalty
    min_lidar_distance = RewTerm(
        func=mdp.lidar_min_distance,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("lidar")})
    
    # too_close_to_obstacle = RewTerm(
    #     func=mdp.lidar_distance_limit,
    #     weight=-100.0,
    #     params={"sensor_cfg": SceneEntityCfg("lidar"), "distance_threshold": 0.35},
    # )
    
    # timed_lap_time = RewTerm(
    #     func=mdp.timed_lap_time,
    #     weight=1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "threshold":0.1, "lap_threshold":2.0}
    # )
    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )
    
    # too_close_to_obstacle = DoneTerm(
    #     func=mdp.lidar_distance_limit,
    #     params={"sensor_cfg": SceneEntityCfg("lidar"), "distance_threshold": 0.35},
    # )
    
    is_flipped = DoneTerm(func=mdp.flipped_over, params={"asset_cfg": SceneEntityCfg("robot")})
    
    


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    pass

##
# Environment configuration
##


@configclass
class F1tenthEnvCfg(RLTaskEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: F1tenthSceneCfg = F1tenthSceneCfg(num_envs=4096, env_spacing=18.0, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    randomization: RandomizationCfg = RandomizationCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # No command generator
    commands: CommandsCfg = CommandsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 60
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 40  # 120
