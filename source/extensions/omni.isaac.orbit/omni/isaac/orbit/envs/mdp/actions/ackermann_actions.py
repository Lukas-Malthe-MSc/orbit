# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets.articulation import Articulation
from omni.isaac.orbit.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv

    from . import actions_cfg


class AckermannAction(ActionTerm):
    r"""Non-holonomic action that maps a two dimensional action to the velocity of the robot in
    the x, y and yaw directions.

    This action term helps model a skid-steer robot base. The action is a 2D vector which comprises of the
    forward velocity :math:`v_{B,x}` and the turning rate :\omega_{B,z}: in the base frame. Using the current
    base orientation, the commands are transformed into dummy joint velocity targets as:

    .. math::

        \dot{q}_{0, des} &= v_{B,x} \cos(\theta) \\
        \dot{q}_{1, des} &= v_{B,x} \sin(\theta) \\
        \dot{q}_{2, des} &= \omega_{B,z}

    where :math:`\theta` is the yaw of the 2-D base. Since the base is simulated as a dummy joint, the yaw is directly
    the value of the revolute joint along z, i.e., :math:`q_2 = \theta`.

    .. note::
        The current implementation assumes that the base is simulated with three dummy joints (prismatic joints along x
        and y, and revolute joint along z). This is because it is easier to consider the mobile base as a floating link
        controlled by three dummy joints, in comparison to simulating wheels which is at times is tricky because of
        friction settings.

        However, the action term can be extended to support other base configurations as well.

    .. tip::
        For velocity control of the base with dummy mechanism, we recommend setting high damping gains to the joints.
        This ensures that the base remains unperturbed from external disturbances, such as an arm mounted on the base.
    """

    cfg: actions_cfg.AckermannActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, 2)."""
    _offset: torch.Tensor
    """The offset applied to the input action. Shape is (1, 2)."""

    def __init__(self, cfg: actions_cfg.AckermannActionCfg, env: BaseEnv):
        super().__init__(cfg, env)

        wheel_ids, wheel_names = self._asset.find_joints(cfg.wheel_joint_names)
        self._wheel_ids = wheel_ids
        self._wheel_names = wheel_names

        steering_ids, steering_names = self._asset.find_joints(cfg.steering_joint_names)
        self._steering_ids = steering_ids
        self._steering_names = steering_names
        
        # Action scaling and offset
        self._scale = torch.tensor(cfg.scale, device=self.device, dtype=torch.float32)
        self._offset = torch.tensor(cfg.offset, device=self.device, dtype=torch.float32)

        # Initialize tensors for actions
        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=self.device)  # Placeholder for [velocity, steering_angle]
        
        self.base_length = torch.tensor(cfg.base_length, device=self.device)
        self.base_width = torch.tensor(cfg.base_width, device=self.device)
        self.wheel_rad = torch.tensor(cfg.wheel_radius, device=self.device)
        self.max_speed = torch.tensor(cfg.max_speed, device=self.device)
        self.max_steering_angle = torch.tensor(cfg.max_steering_angle, device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """



    def process_actions(self, actions):
        # store the raw actions
        # actions[:, 0] = torch.clamp(actions[:, 0], min=2., max=2.)
        # actions[:, 1] = torch.clamp(actions[:, 1], min=0.1, max=0.1)
        
        self._raw_actions[:] = actions
        
        self._processed_actions = self.raw_actions * self._scale + self._offset
        self._processed_actions[:, 0] = torch.clamp(self._processed_actions[:, 0], min=-self.max_speed, max=self.max_speed)
        self._processed_actions[:, 1] = torch.clamp(self._processed_actions[:, 1], min=-self.max_steering_angle, max=self.max_steering_angle)

    def apply_actions(self):

        left_rotator_angle, right_rotator_angle, wheel_speeds = self. calculate_ackermann_angles_and_velocities(
            target_velocity=self.processed_actions[:, 0],  # Velocity for all cars
            target_steering_angle_rad=self.processed_actions[:, 1] # Steering angle for all cars
        )
        wheel_angles = torch.stack([left_rotator_angle, right_rotator_angle], dim=1)

        # wheel_angles = torch.zeros(wheel_speeds.shape[0], 2, device=self.device)
        # wheel_speeds = torch.ones(wheel_speeds.shape[0], 4, device=self.device)

        self._asset.set_joint_velocity_target(wheel_speeds, joint_ids=self._wheel_ids)
        self._asset.set_joint_position_target(wheel_angles, joint_ids=self._steering_ids)

     
        
    def calculate_ackermann_angles_and_velocities(self, target_steering_angle_rad, target_velocity):
        L = self.base_length
        W = self.base_width
        wheel_radius = self.wheel_rad
        
        # Ensure inputs are PyTorch tensors
        target_steering_angle_rad = target_steering_angle_rad.float()
        target_velocity = target_velocity.float()
        
        # Calculating the turn radius from the steering angle
        R = L / torch.tan(target_steering_angle_rad)
        
        # Calculate the steering angles for the left and right front wheels in radians
        delta_left = torch.atan(L / (R - W / 2))
        delta_right = torch.atan(L / (R + W / 2))
        
        # Calculate target rotation for each wheel
        target_rotation = target_velocity / wheel_radius
        
        # TODO: Implement proper wheel speeds for each wheel. This code is inspired from the Ackermann OmniGraph node.
        front_wheel_left_speed = target_rotation 
        front_wheel_right_speed = target_rotation 
        back_wheel_left_speed = target_rotation 
        back_wheel_right_speed = target_rotation
        
        # Assign calculated speeds to each wheel: [front_left, front_right, rear_left, rear_right]
        wheel_speeds = torch.stack([front_wheel_left_speed, front_wheel_right_speed, back_wheel_left_speed, back_wheel_right_speed], dim=1)
        
        return delta_left, delta_right, wheel_speeds
