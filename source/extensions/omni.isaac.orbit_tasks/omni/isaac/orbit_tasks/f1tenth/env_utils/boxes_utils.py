import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import RigidObjectCfg
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
import omni.isaac.orbit_tasks.f1tenth.mdp as mdp
from omni.isaac.orbit.managers import SceneEntityCfg
import random

def create_box_configs(num_boxes=10, size=(0.35, 0.32, 0.30), 
                       pos_ranges=((5.0, -5.0), (5.0, -5.0), (0.0, 1.0)), 
                       rot_ranges=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))):
    """Create box configurations for the environment with random positions and rotations"""
    box_configs = []
    for i in range(0, num_boxes):
        pos = (
            random.uniform(pos_ranges[0][0], pos_ranges[0][1]),
            random.uniform(pos_ranges[1][0], pos_ranges[1][1]),
            random.uniform(pos_ranges[2][0], pos_ranges[2][1])
        )
        rot = (
            random.uniform(rot_ranges[0][0], rot_ranges[0][1]),
            random.uniform(rot_ranges[1][0], rot_ranges[1][1]),
            random.uniform(rot_ranges[2][0], rot_ranges[2][1]),
            random.uniform(rot_ranges[3][0], rot_ranges[3][1])
        )
        box_configs.append(
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}" + f"/box{i}",
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg()
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=pos,
                    rot=rot
                )
            )
        )
    return box_configs



def create_randomize_box_pos(num_boxes, pose_range, velocity_range):
    """Create randomize box positions for the environment"""
    randomize_boxes_positions = []
    for i in range(0, num_boxes):
        randomize_boxes_positions.append(
            RandTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(f"box{i}"),
                    "pose_range": pose_range,
                    "velocity_range": velocity_range
                }
            )
        )
    return randomize_boxes_positions


def create_box_configs_with_positions(positions, size=(0.35, 0.32, 0.30), rot=(1.0, 0.0, 0.0, 0.0)):
    """Create box configurations for the environment with given positions"""
    box_configs = []
    for i, pos in enumerate(positions, start=1):
        box_configs.append(
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}" + f"/box{i}",
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg()
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=pos,
                    rot=rot
                )
            )
        )
    return box_configs