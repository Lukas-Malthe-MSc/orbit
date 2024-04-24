# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from omni.isaac.orbit.utils import configclass

@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""
    
@configclass
class RslRlActorCriticRecurrentCfg:
    """Configuration for the recurrent PPO actor-critic networks."""
    
    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Defaults to ActorCriticRecurrent."""
    
    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""
    
    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""
    
    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""
    
    activation: str = MISSING
    """The activation function for the actor and critic networks."""
    
    rnn_type: str = "lstm"
    """The type of RNN to use ('lstm' or 'gru')."""
    
    rnn_hidden_size: int = 256
    """The hidden state size of the RNN layer."""
    
    rnn_num_layers: int = 1
    """The number of RNN layers."""
    
    
@configclass
class RslRlActorCriticTransformerCfg:
    """Configuration for the transformer PPO actor-critic networks."""
    
    class_name: str = "ActorCriticTransformer"
    """The policy class name. Defaults to ActorCriticTransformer."""
    
    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""
    
    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""
    
    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""
    
    activation: str = MISSING
    """The activation function for the actor and critic networks."""
    
    num_heads: int = 4
    """The number of attention heads."""
    
    transformer_layers: int = 2
    """The number of transformer layers."""
    
    transformer_dim: int = 256
    """The hidden dimension of the transformer."""
    
@configclass
class RslRlActorCriticSelfAttentionCfg:
    """Configuration for the self-attention PPO actor-critic networks."""
    
    class_name: str = "ActorCriticSelfAttention"
    """The policy class name. Defaults to ActorCriticSelfAttention."""
    
    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""
    
    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""
    
    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""
    
    activation: str = MISSING
    """The activation function for the actor and critic networks."""
    
    attention_size: int = 512
    """The hidden size of the attention layer."""
    
@configclass
class RslRlActorCriticLidarCnnCfg:
    """Configuration for the Lidar CNN PPO actor-critic networks."""
    
    class_name: str = "ActorCriticLidarCnn"
    """The policy class name. Defaults to ActorCriticLidarCnn."""
    
    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""
    
    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""
    
    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""
    
    activation: str = MISSING
    """The activation function for the actor and critic networks."""
    
    num_lidar_scans: int = 1081
    """The number of lidar scans."""
    
    kernel_size: int = 3
    """The kernel size for the CNN."""
    
    out_channels: int = 32
    """The number of output channels for the CNN."""


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""


@configclass
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda"
    """The device for the rl-agent. Default is cuda."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg | RslRlActorCriticRecurrentCfg | RslRlActorCriticLidarCnnCfg | RslRlActorCriticSelfAttentionCfg | RslRlActorCriticTransformerCfg= MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "orbit"
    """The neptune project name. Default is "orbit"."""

    wandb_project: str = "orbit"
    """The wandb project name. Default is "orbit"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
