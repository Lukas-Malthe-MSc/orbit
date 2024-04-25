# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.utils import configclass

from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlActorCriticRecurrentCfg,
    RslRlActorCriticTransformerCfg,
    RslRlActorCriticSelfAttentionCfg,
    RslRlActorCriticLidarCnnCfg,
    RslRlActorCriticViTCfg,
)

@configclass
class F1tenthPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50
    save_interval = 10
    experiment_name = "f1tenth"
    empirical_normalization = False
    device = "cuda:0"
    
    """MLP"""
    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     activation="elu",
    # )
    
    """Recurrent"""
    # policy = RslRlActorCriticRecurrentCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     activation="elu",
    #     rnn_type="lstm",
    #     rnn_hidden_size=512,
    #     rnn_num_layers=1,
    # )
        
    # policy = RslRlActorCriticTransformerCfg(
    #     actor_hidden_dims=[256, 128],
    #     critic_hidden_dims=[256, 128],
    #     activation="elu",
    #     init_noise_std=1.0,
    #     num_heads=4,
    #     transformer_layers=2,
    #     transformer_dim=512
    # )
    
    """Self-Attention"""
    policy = RslRlActorCriticSelfAttentionCfg(
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        attention_size=512,
    )
    
    """LiDAR CNN"""
    # policy = RslRlActorCriticLidarCnnCfg(
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     activation="elu",
    #     init_noise_std=1.0,
    #     num_lidar_scans=1081,
    #     kernel_size=5,
    #     out_channels=16,
    # )
    
    """ViT (Vision Transformer)"""
    # policy = RslRlActorCriticViTCfg(
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     activation="elu",
    #     init_noise_std=1.0,
    #     num_heads=4,
    #     transformer_layers=2,
    #     transformer_dim=512,
    #     patch_size=16,
    # )
        
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# @configclass
# class AnymalCFlatPPORunnerCfg(AnymalCRoughPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         self.max_iterations = 300
#         self.experiment_name = "anymal_c_flat"
#         self.policy.actor_hidden_dims = [128, 128, 128]
#         self.policy.critic_hidden_dims = [128, 128, 128]

"""
from omni.isaac.orbit.utils import configclass

from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class F1tenthPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "f1tenth"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
"""