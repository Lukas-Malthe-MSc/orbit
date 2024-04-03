# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from omni.isaac.orbit.sensors.lidar.lidar_cfg import LidarCfg
from omni.isaac.orbit.sim.spawners.sensors.sensors_cfg import LidarSensorCfg
from rich import print

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.sensors import RayCasterCfg, patterns, CameraData, RayCaster
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.orbit_tasks.f1tenth.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.orbit_assets.f1tenth import F1TENTH_CFG  # isort:skip


##
# Scene definition
##


@configclass
class F1tenthSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
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
        min_range=None,  # Minimum range of 0.1 meters
        rotation_rate=0.0,  # Rotation rate of 0.0 radians per second
        offset=LidarCfg.OffsetCfg(
            pos=(0.11749, 0.0, 0.1),  # Example position offset from the robot base
            rot=(1.0, 0.0, 0.0, 0.0),  # Example rotation offset; no rotation in this case
            convention="ros"  # Frame convention
        )
    )
    
    race_track = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RaceTrack",
        collision_group=-1,
        spawn=sim_utils.UsdFileCfg(
            # usd_path="omniverse://localhost/Projects/f1tenth/box.usd",
            usd_path="omniverse://localhost/Projects/f1tenth/racetrack_square.usd",
            scale=(.01, .01, .01),
        )
    )

    
    # TODO: Add touch sensor that can register collisions with the walls
    # Check ant_env_cfg.py for an example of how to add a touch sensor




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
                                  base_width=0.24, base_length=0.33, wheel_radius=0.062, max_speed=2.0, max_steering_angle=math.pi/4, scale=(1.0, 1.0), offset=(0.0, 0.0)) #TODO: adjust max speed

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_pos = ObsTerm(func=mdp.base_pos, noise=Unoise(n_min=-0.1, n_max=0.1))
        # base_rot = ObsTerm(func=mdp.base_rot, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        
        lidar_ranges = ObsTerm(func=mdp.lidar_ranges, params={"sensor_cfg": SceneEntityCfg("lidar")})
        
        last_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RandomizationCfg:
    """Configuration for randomization."""

    reset_root_state = RandTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "pose_range": {
                # "x": (-0.3, 0.3),  # X position range from -5 to 5
                "x": (3.0, 3.5),  # X position range from -5 to 5
                "y": (-0.3, 0.3),  # Y position range from -5 to 5
                "z": (0.0, 0.5),   # Z position range from 0 to 2 (assuming starting on the ground)
                "roll": (-0.2, 0.2),  # Roll orientation range from -pi to pi
                "pitch": (-0.2, 0.2), # Pitch orientation range from -pi to pi
                "yaw": (-0.1, 0.1),   # Yaw orientation range from -pi to pi
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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    # # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-10.0)
    
    # -- Task: Drive forward
    velocity = RewTerm(func=mdp.forward_velocity, weight=1.0)
    
    # -- Task: Move to center of track
    # lidar_deviation = RewTerm(func=mdp.lidar_mean_absolute_deviation, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("lidar")})
    
    # -- Task: Move to position
    # move_to_position = RewTerm(func=mdp.move_to_position, weight=-1.0, params={"target": (5.0, 4.0), "asset_cfg": SceneEntityCfg("robot")})
    
    # -- Penalty
    steering_angle_position = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=['rotator_left', 'rotator_right']), "target": 0.0}
    )
    
    # # -- Penalty
    min_lidar_distance = RewTerm(
        func=mdp.lidar_min_distance,
        weight=-0.01,
        params={"sensor_cfg": SceneEntityCfg("lidar")})
    
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
    


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass

##
# Environment configuration
##


@configclass
class F1tenthEnvCfg(RLTaskEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: F1tenthSceneCfg = F1tenthSceneCfg(num_envs=4096, env_spacing=15.0, replicate_physics=True)
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
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 40  # 120
