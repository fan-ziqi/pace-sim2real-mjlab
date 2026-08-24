"""ANYmal-D PACE configuration matching the original task name and parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.entity import EntityCfg

from pace_sim2real.assets.anymal_d_asset import (
    ANYDRIVE_PACE_ACTUATOR_CFG,
    JOINT_ORDER,
    get_anymal_d_robot_cfg,
)

from .pace_sim2real_env_cfg import (
    PaceCfg,
    PaceSim2realEnvCfg,
    PaceSim2realSceneCfg,
    _default_pace_manager_terms,
)

TASK_ID = "Isaac-Pace-Anymal-D-v0"


def _anymal_d_bounds() -> torch.Tensor:
    bounds = torch.zeros((49, 2), dtype=torch.float32)
    bounds[:12, 0] = 1.0e-5
    bounds[:12, 1] = 1.0
    bounds[12:24, 1] = 7.0
    bounds[24:36, 1] = 0.5
    bounds[36:48, 0] = -0.1
    bounds[36:48, 1] = 0.1
    bounds[48, 1] = 10.0
    return bounds


@dataclass(kw_only=True)
class AnymalDPaceCfg(PaceCfg):
    """PACE identification parameter ranges for ANYmal D."""

    robot_name: str = "anymal_d_sim"
    data_dir: str = "anymal_d_sim/chirp_data.pt"
    joint_order: list[str] = field(default_factory=lambda: list(JOINT_ORDER))
    bounds_params: torch.Tensor = field(default_factory=_anymal_d_bounds)


@dataclass(kw_only=True)
class ANYmalDPaceSceneCfg(PaceSim2realSceneCfg):
    """Compatibility scene config exposing the original ``scene.robot`` field."""

    robot: EntityCfg = field(default_factory=get_anymal_d_robot_cfg)

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass(kw_only=True)
class AnymalDPaceEnvCfg(PaceSim2realEnvCfg):
    """Native, inheritable mjlab counterpart of the upstream ANYmal config.

    Existing users can subclass this class and override ``__post_init__`` just
    as they did in Isaac Lab.  The manager-facing dictionaries additionally
    expose their original attribute spellings (for example
    ``cfg.actions.joint_pos``).
    """

    scene: ANYmalDPaceSceneCfg = field(default_factory=ANYmalDPaceSceneCfg)
    sim2real: AnymalDPaceCfg = field(default_factory=AnymalDPaceCfg)

    def __post_init__(self) -> None:
        super().__post_init__()
        # ANYmal's URDF includes two passive joints.  Keep the original PACE
        # 36-value policy observation over its 12 fitted actuator joints while
        # the generic base EnvCfg retains all-joint defaults for custom robots.
        observations, _, rewards, _ = _default_pace_manager_terms(tuple(self.sim2real.joint_order))
        self.observations = observations
        self.rewards = rewards
        # ``sim.dt`` remains writable through PaceSimulationCfg's alias.
        self.sim.dt = 0.0025


def anymal_d_pace_env_cfg(play: bool = False, num_envs: int = 4096) -> AnymalDPaceEnvCfg:
    """Create the registered ``Isaac-Pace-Anymal-D-v0`` mjlab environment."""
    cfg = AnymalDPaceEnvCfg(scene=ANYmalDPaceSceneCfg(num_envs=num_envs))
    if play:
        cfg.episode_length_s = 1.0e9
    return cfg


__all__ = [
    "ANYDRIVE_PACE_ACTUATOR_CFG",
    "AnymalDPaceCfg",
    "AnymalDPaceEnvCfg",
    "ANYmalDPaceSceneCfg",
    "TASK_ID",
    "anymal_d_pace_env_cfg",
]
