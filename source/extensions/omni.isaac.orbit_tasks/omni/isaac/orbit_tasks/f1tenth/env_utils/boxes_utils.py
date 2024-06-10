import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import RigidObjectCfg
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
import omni.isaac.orbit_tasks.f1tenth.mdp as mdp
from omni.isaac.orbit.managers import SceneEntityCfg

def create_box_configs(num_boxes=10, size=(0.35, 0.32, 0.30), pos=(0.0, 0.0, 0.25), rot=(1.0, 0.0, 0.0, 0.0)):
    box_configs = []
    for i in range(0, num_boxes):
        box_configs.append(
            RigidObjectCfg(
                prim_path= "{ENV_REGEX_NS}" + f"/box{i}",
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
    randomize_boxes_positions = []
    for i in range(1, num_boxes + 1):
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