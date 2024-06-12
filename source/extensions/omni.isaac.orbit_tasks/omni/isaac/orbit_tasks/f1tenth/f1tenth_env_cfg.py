import torch
from omni.isaac.orbit.sensors.lidar.lidar_cfg import LidarCfg
from .env_utils.boxes_utils import create_box_configs, create_randomize_box_pos, create_box_configs_with_positions

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
import omni.isaac.orbit_tasks.f1tenth.mdp as mdp
from omni.isaac.orbit_assets.f1tenth import F1TENTH_CFG
from pathlib import Path

current_working_directory = Path.cwd()                                                                                              

"""
Train commmand:
./orbit.sh -p source/standalone/workflows/rsl_rl/train.py --task F1tenth-v0 --headless --offscreen_render --num_envs 4096

Play command:
CNN
./orbit.sh -p source/standalone/workflows/rsl_rl/play.py --task F1tenth-v0 --num_envs 1 --load_run 2024-05-16_14-38-55 --checkpoint model_999.pt

MLP
./orbit.sh -p source/standalone/workflows/rsl_rl/play.py --task F1tenth-v0 --num_envs 1 --load_run 2024-05-16_19-22-56 --checkpoint model_999.pt

LSTM
./orbit.sh -p source/standalone/workflows/rsl_rl/play.py --task F1tenth-v0 --num_envs 1 --load_run 2024-05-16_17-58-21 --checkpoint model_999.pt

TODO: Remember to change the neural network type in the rsl_rl_ppo_cfg.py file
"""

# For now this has to be done manually
is_inference = False
race_track_str = "test_track_1" #Used for infrence only. 
#Options: Testing [test_track_1, test_track_2, test_track_3], Training [track_1, track_2, track_3, track_4, track_5]
use_obstacles = True

@configclass
class F1tenthSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(1600.0, 1200.0),
                                       color=(3, 3, 3),
                                       physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, 
                                                                                       dynamic_friction=0.5, 
                                                                                       restitution=0.0)),   
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
        min_range=0.020,  # Minimum range of 0.1 meters
        rotation_rate=0.0,  # Rotation rate of 0.0 radians per second
        offset=LidarCfg.OffsetCfg(
            pos=(2.11749, 0.0, -2.1),  # Example position offset from the robot base
            rot=(1.0, 0.0, 0.0, 0.0),  # Example rotation offset; no rotation in this case
            convention="ros"  # Frame convention
        ),
        draw_lines=False,
        draw_points=True,
    )

    # race_track: AssetBaseCfg = AssetBaseCfg( 
    #     prim_path="{ENV_REGEX_NS}/RaceTrack",
    #     collision_group=0,
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path= f"{current_working_directory}/f1tenth_assets/omniverse/maps/validation_track.usd",
    #         scale=(.01, .01, .01),
    #     )
    # )

    
    race_track: AssetBaseCfg = AssetBaseCfg( 
        prim_path="{ENV_REGEX_NS}/RaceTrack",
        collision_group=0,
        spawn=sim_utils.UsdFileCfg(
            usd_path= f"{current_working_directory}/f1tenth_assets/omniverse/maps/{race_track_str}.usd",
            scale=(.015, .015, .015),
        )
    )
    
    # Create boxes for training mode
    if not is_inference:
        box0, box1, box2, box3, box4, box5, box6, box7, box8, box9 = create_box_configs(num_boxes=10, size=(0.35, 0.32, 0.30))
        
    # Create boxes for testing mode 
    # Note: only creates boxes during experiments
    else:
        # Experiment on test track 1 with obstacles
        if race_track_str == "test_track_1" and use_obstacles:
            box1, box2, box3, box4, box5, box6 = create_box_configs_with_positions(
                positions = [
                    (-5.2, -2.7, 0.25),
                    (4.3, 3.8, 0.25),
                    (-1.4, 5.1, 0.25),
                    (2.7, -5.6, 0.25),
                    (-3.3, 1.2, 0.25),
                    (0.5, -4.9, 0.25)
                ]
            )
        # Experiment on test track 2 with obstacles
        if race_track_str == "test_track_2" and use_obstacles:
            box1, box2, box3, box4, box5, box6 = create_box_configs_with_positions(
                positions = [
                    (-4.3, -3.2, 0.25),
                    (2.3, 5.0, 0.25),
                    (-2.1, 2.9, 0.25),
                    (3.0, -4.2, 0.25),
                    (-3.8, 0.5, 0.25),
                    (1.4, -4.1, 0.25)
                ]
            )

        # Experiment on test track 3 with obstacles
        if race_track_str == "test_track_3" and use_obstacles:
            box1, box2, box3, box4, box5, box6 = create_box_configs_with_positions(
                positions = [
                    (-5.1, 4.4, 0.25),
                    (4.2, -3.7, 0.25),
                    (-2.8, 5.6, 0.25),
                    (5.5, 4.5, 0.25),
                    (0.3, -5.5, 0.25),
                    (-4.7, -1.6, 0.25)
                ]
            )


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
    ackermann_action = mdp.AckermannActionCfg(asset_name="robot", 
                                  wheel_joint_names=["wheel_back_left", "wheel_back_right", "wheel_front_left", "wheel_front_right"], 
                                  steering_joint_names=["rotator_left", "rotator_right"], 
                                  base_width=0.24, base_length=0.32, wheel_radius=0.056, bounding_strategy="tanh", scale=(5., 0.36), offset=(0.0, 0.0)) #TODO: adjust max speed

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # observation terms (order preserved)
        # lidar_ranges = ObsTerm(func=mdp.lidar_ranges, noise=Gnoise(mean=0.0, std=0.1), params={"sensor_cfg": SceneEntityCfg("lidar")})
        lidar_ranges_normalized = ObsTerm(func=mdp.lidar_ranges_normalized, params={"sensor_cfg": SceneEntityCfg("lidar")})
        base_lin_vel_xy_dot = ObsTerm(func=mdp.base_lin_vel_xy_dot, noise=Gnoise(mean=0.0, std=0.1))
        base_ang_vel_yaw_dot = ObsTerm(func=mdp.base_ang_vel_yaw_dot, noise=Gnoise(mean=0.0, std=0.1))
        
        last_actions = ObsTerm(func=mdp.last_processed_action, params={"bounding_strategy": "tanh", "scale": (5., 0.36), "offset": (0.0, 0.0)})

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RandomizationCfg:
    """Configuration for randomization."""
    
    # Training randomization configuration
    if not is_inference:
        # startup
        add_base_mass = RandTerm(
            func=mdp.add_body_mass,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"), "mass_range": (0.1, 0.5)},
        )
        
        maps = RandTerm(
            func=mdp.randomize_map,
            mode="startup",
            params={
                "maps_paths": [
                    f"{current_working_directory}/f1tenth_assets/omniverse/maps/track_1.usd",
                    f"{current_working_directory}/f1tenth_assets/omniverse/maps/track_2.usd",
                    f"{current_working_directory}/f1tenth_assets/omniverse/maps/track_3.usd",
                    f"{current_working_directory}/f1tenth_assets/omniverse/maps/track_4.usd",
                    f"{current_working_directory}/f1tenth_assets/omniverse/maps/track_5.usd",
                ],
                "map_prim": "RaceTrack",
                "scale": (0.015, 0.015, 0.015),
            },
        )
        
        physics_material = RandTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"),
                "static_friction_range": (0.9, 1),
                "dynamic_friction_range": (0.9, 1),
                "restitution_range": (0.0, 0.1),
                "num_buckets": 64,
            },
        )
        

        # randomize box positions at each reset
        randomize_box1_pos, randomize_box2_pos, randomize_box3_pos, \
        randomize_box4_pos, randomize_box5_pos, randomize_box6_pos, \
        randomize_box7_pos, randomize_box8_pos, randomize_box9_pos, \
        randomize_box0_pos = create_randomize_box_pos(
            num_boxes=10, 
            pose_range = {
                "x": (-6.5, 6.5),
                "y": (-6.5, 6.5),
                "z": (0.0, 0.75),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            }, velocity_range = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
        })
        
        # reset
        reset_root_state = RandTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"), 
                "pose_range": {
                    "x": (5.0, 5.0),  # X position range from 5 to 5
                    "y": (5.0, 5.0),  # Y position range from 5 to 5
                    "z": (0.01, 0.01),   # Z position range
                    "roll": (0.0, 0.0),  # Roll orientation range
                    "pitch": (0.0, 0.0), # Pitch orientation range
                    "yaw": (-3.14, 3.14),   # Yaw orientation range from -pi to pi
                }, 
                "velocity_range": {
                    "x": (0.0, 0.0),  # X linear velocity range
                    "y": (0.0, 0.0),  # Y linear velocity range
                    "z": (0.0, 0.0),  # Z linear velocity range
                    "roll": (0.0, 0.0),  # Roll angular velocity range
                    "pitch": (0.0, 0.0), # Pitch angular velocity range
                    "yaw": (0.0, 0.0),   # Yaw angular velocity range
                }     
            },
        )
    
    # Inference randomization configuration 
    else:
        reset_root_state_non_random = RandTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"), 
                "pose_range": {
                    "x": (5.0, 5.0),  # X position range from 5 to 5
                    "y": (5.0, 5.0),  # Y position range from 5 to 5
                    "z": (0.01, 0.01),   # Z position range
                    "roll": (0.0, 0.0),  # Roll orientation range
                    "pitch": (0.0, 0.0), # Pitch orientation range
                    "yaw": (0, 0),   # NOTE: Changed yaw orientation range from -pi to pi to 0 to 0
                }, 
                "velocity_range": {
                    "x": (0.0, 0.0),  # X linear velocity range
                    "y": (0.0, 0.0),  # Y linear velocity range
                    "z": (0.0, 0.0),  # Z linear velocity range
                    "roll": (0.0, 0.0),  # Roll angular velocity range
                    "pitch": (0.0, 0.0), # Pitch angular velocity range
                    "yaw": (0.0, 0.0),   # Yaw angular velocity range
                }     
            },
        )
        physics_material_non_random = RandTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"),
                "static_friction_range": (0.95, 0.95),
                "dynamic_friction_range": (0.95, 0.95),
                "restitution_range": (0.05, 0.05),
                "num_buckets": 64,
            },
        )

    reset_robot_joints = RandTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Collision penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-10.0)
    
    # -- Task: Forward velocity
    forward_velocity_reward = RewTerm(func=mdp.forward_velocity, weight=1.0)

    # -- Penalty: Steering angle
    steering_angle_position = RewTerm(
        func=mdp.action_target_log_l1,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot"), "action_idx": 1 ,"lambda_1": 180/torch.pi, "target": 0.0}
    )
    
    # Using timed lap time for inference to measure performance
    if is_inference:
        timed_lap_time = RewTerm(
            func=mdp.timed_lap_time,
            weight=.0001,
            params={"asset_cfg": SceneEntityCfg("robot"), "threshold":0.7, "lap_threshold":0.8}
        )
    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    # For training use collision termination
    if not is_inference:
        collision = DoneTerm(
            func=mdp.lidar_distance_limit,
            params={"sensor_cfg": SceneEntityCfg("lidar"), "distance_threshold": 0.4},
        )
    
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    is_flipped = DoneTerm(
        func=mdp.flipped_over, 
        params={"asset_cfg": SceneEntityCfg("robot")})
    

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
        # general settings for training
        self.decimation = 2
        self.sim.dt = 1 / 80
        self.episode_length_s = 60
        
        # Inference settings - smaller sim.dt means faster simulation rendering
        if is_inference:
            self.decimation = 1
            self.sim.dt = 1 / 40
            self.episode_length_s *= 10 # 10 minutes
            
        # viewer settings
        self.viewer.eye = (0.0, 0.0, 30.0)
        # simulation settings
        
        # self.sim.use_fabric = False

        
