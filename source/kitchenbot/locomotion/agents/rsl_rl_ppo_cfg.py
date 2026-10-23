# UNTESTED DRAFT (#46): written against Isaac Lab v3.0.0 source without a simulator; #5 runs and fixes it.  # noqa: E501
"""RSL-RL PPO for ``KitchenBot-Stand-G1-Dex3`` (docs/architecture.md section 6).

Actor and critic MLP [256, 128, 128], 24 steps per environment, 1,500 iterations as the first
budget. Built on the 3.0 model configs (``RslRlMLPModelCfg`` for ``actor`` and ``critic``);
``RslRlPpoActorCriticCfg`` is deprecated for rsl-rl >= 4.0 and we pin 5.5.1. Every PPO value is
copied from Isaac Lab's ``G1FlatPPORunnerCfg``
(``core/velocity/config/g1/agents/rsl_rl_ppo_cfg.py``, v3.0.0); see
docs/isaaclab_3_api_notes.md section 7.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

HIDDEN_DIMS = [256, 128, 128]


@configclass
class StandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "kitchenbot_stand_g1_dex3"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=list(HIDDEN_DIMS),
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=list(HIDDEN_DIMS),
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
