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
        update_period=0.025,  # Update rate of 40Hz
        # data_types=["point_cloud"],  # Assuming the LiDAR generates point cloud data
        spawn=None,
        offset=LidarCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.2),  # Example position offset from the robot base
            rot=(1.0, 0.0, 0.0, 0.0),  # Example rotation offset; no rotation in this case
            convention="ros"  # Frame convention
        )
    )
    
    
    
    race_track = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RaceTrack",
        collision_group=-1,
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/Projects/f1tenth/race_track.usd",
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
                                  base_width=0.24, base_length=0.33, wheel_radius=0.062, max_speed=10.0, max_steering_angle=math.pi/4, scale=(1.0, 1.0), offset=(0.0, 0.0))

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_pos = ObsTerm(func=mdp.base_pos, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_rot = ObsTerm(func=mdp.base_rot, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        
        lidar_ranges = ObsTerm(func=mdp.lidar_ranges, params={"sensor_cfg": SceneEntityCfg("lidar")})

        print(f"lidar_ranges: {lidar_ranges}")
        
        last_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RandomizationCfg:
    """Configuration for randomization."""

    # On reset
    reset_root_state = RandTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "pose_range": {
                "x": (-2.5, 2.5),  # X position range from -5 to 5
                "y": (-2.5, 2.5),  # Y position range from -5 to 5
                "z": (0.0, 0.5),   # Z position range from 0 to 2 (assuming starting on the ground)
                "roll": (-0.2, 0.2),  # Roll orientation range from -π to π
                "pitch": (-0.2, 0.2), # Pitch orientation range from -π to π
                "yaw": (-3.14, 3.14),   # Yaw orientation range from -π to π
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
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    
    # -- Task
    velocity = RewTerm(func=mdp.forward_velocity, weight=2.0)
    
    # -- Penalty
    steering_angle_position = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=['rotator_left', 'rotator_right']), "target": 0.0}
    )

    # # (3) Primary task: keep pole upright
    # pole_pos = RewTerm(
    #     func=mdp.joint_pos_target_l2,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    # )
    # # (4) Shaping tasks: lower cart velocity
    # car_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=0.05,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=['wheel_back_left', 
    #                                                           'wheel_back_right', 'wheel_front_left', 'wheel_front_right'])},
    # )
    # # (5) Shaping tasks: lower pole angular velocity
    # pole_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    # )
    
    ## (6 MY REWARD) Drive forward
    # upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})
    # (4) Reward for moving in the right direction
    # move_to_target = RewTerm(
    #     func=mdp.move_to_target_bonus, weight=0.5, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)}
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
    
    too_close_to_obstacle = DoneTerm(
        func=mdp.lidar_distance_limit,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "distance_threshold": 0.3},
    )
    # bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1})

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
    scene: F1tenthSceneCfg = F1tenthSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)
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
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
