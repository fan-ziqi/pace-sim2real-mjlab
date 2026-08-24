"""RSL-RL PPO config matching PACE's original compact template runner."""

from dataclasses import dataclass, fields
from typing import Any

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(32, 32),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=(32, 32), activation="elu", obs_normalization=False),
        algorithm=RslRlPpoAlgorithmCfg(
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
        ),
        experiment_name="pace_sim2real",
        save_interval=50,
        num_steps_per_env=16,
        max_iterations=150,
        # PACE identification does not require cloud tracking.  Make the
        # offline-safe backend the default; W&B remains opt-in via CLI.
        logger="tensorboard",
    )


@dataclass(init=False)
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Source-compatible compact PACE runner config, not mjlab's generic default."""

    def __init__(self, **overrides: Any) -> None:
        template = ppo_runner_cfg()
        field_names = {item.name for item in fields(RslRlOnPolicyRunnerCfg)}
        unknown = set(overrides).difference(field_names)
        if unknown:
            raise TypeError(f"Unknown PPORunnerCfg fields: {sorted(unknown)}")
        for item in fields(RslRlOnPolicyRunnerCfg):
            setattr(self, item.name, overrides.get(item.name, getattr(template, item.name)))
