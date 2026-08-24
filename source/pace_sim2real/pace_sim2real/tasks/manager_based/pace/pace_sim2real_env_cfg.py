"""Framework-independent PACE configuration built on mjlab's manager API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig


class AttrDict(dict[str, Any]):
    """Dictionary used by mjlab managers with Isaac-style attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


@dataclass(kw_only=True)
class PaceSimulationCfg(SimulationCfg):
    """mjlab simulation config with the upstream ``sim.dt`` compatibility alias."""

    @property
    def dt(self) -> float:
        return self.mujoco.timestep

    @dt.setter
    def dt(self, value: float) -> None:
        self.mujoco.timestep = value


@dataclass(kw_only=True)
class PaceSim2realSceneCfg(SceneCfg):
    """Scene defaults corresponding to the Isaac Lab PACE scene config."""

    num_envs: int = 4096
    env_spacing: float = 2.5
    terrain: TerrainEntityCfg | None = field(
        default_factory=lambda: TerrainEntityCfg(terrain_type="plane")
    )
    robot: EntityCfg | None = None

    def __post_init__(self) -> None:
        if self.robot is not None:
            self.entities = {**self.entities, "robot": self.robot}


@dataclass(kw_only=True)
class CMAESOptimizerCfg:
    """CMA-ES optimizer settings."""

    max_iteration: int = 200
    epsilon: float | None = 1e-2
    sigma: float = 0.5
    save_interval: int = 10
    save_optimization_process: bool = False


@dataclass(kw_only=True)
class PaceCfg:
    """Robot-specific PACE fitting configuration."""

    cmaes: CMAESOptimizerCfg = field(default_factory=CMAESOptimizerCfg)
    robot_name: str = ""
    data_dir: str = ""
    joint_order: list[str] = field(default_factory=list)
    bounds_params: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 2), dtype=torch.float32)
    )


def _default_pace_manager_terms(
    fitted_joint_names: tuple[str, ...] | None = None,
) -> tuple[AttrDict, AttrDict, AttrDict, AttrDict]:
    """Build the manager defaults exposed by the upstream base EnvCfg.

    Like the Isaac Lab base configuration, the default joint-position action
    and observation terms cover every joint driven by the robot's actuators.
    ``sim2real.joint_order`` selects the fitted subset only; it must not make a
    custom robot with extra controlled joints accept a too-short action vector.
    """
    fitted_cfg = (
        SceneEntityCfg("robot", joint_names=fitted_joint_names)
        if fitted_joint_names is not None
        else None
    )
    joint_params = {"asset_cfg": fitted_cfg} if fitted_cfg is not None else {}
    observation_terms = {
        "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel, params=joint_params),
        "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel, params=joint_params),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
    }
    observations = AttrDict(
        {
            "actor": ObservationGroupCfg(dict(observation_terms), enable_corruption=False),
            "critic": ObservationGroupCfg(dict(observation_terms), enable_corruption=False),
            "policy": ObservationGroupCfg(dict(observation_terms), enable_corruption=False),
        }
    )
    actions = AttrDict(
        {
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=1.0,
                use_default_offset=False,
            )
        }
    )
    rewards = AttrDict(
        {
            "dof_pos_limits": RewardTermCfg(
                func=envs_mdp.joint_pos_limits, weight=0.0, params=joint_params
            )
        }
    )
    terminations = AttrDict(
        {"time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=False)}
    )
    return observations, actions, rewards, terminations


@dataclass(kw_only=True)
class PaceSim2realEnvCfg(ManagerBasedRlEnvCfg):
    """Base mjlab config retaining the PACE public configuration name."""

    decimation: int = 1
    scene: PaceSim2realSceneCfg = field(default_factory=PaceSim2realSceneCfg)
    observations: AttrDict = field(default_factory=AttrDict)
    actions: AttrDict = field(default_factory=AttrDict)
    rewards: AttrDict = field(default_factory=AttrDict)
    terminations: AttrDict = field(default_factory=AttrDict)
    sim2real: PaceCfg = field(default_factory=PaceCfg)
    episode_length_s: float = 99999.0
    sim: PaceSimulationCfg = field(
        default_factory=lambda: PaceSimulationCfg(mujoco=MujocoCfg(timestep=0.0025))
    )
    viewer: ViewerConfig = field(
        default_factory=lambda: ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base",
            distance=2.0,
            elevation=-18.0,
            azimuth=45.0,
        )
    )

    def __post_init__(self) -> None:
        """Fill the same usable manager defaults as the original base EnvCfg.

        A custom robot normally subclasses this class and replaces ``scene``
        and ``sim2real``.  Leaving its four manager dictionaries empty would
        construct an inert zero-action environment, so only user-supplied
        non-empty dictionaries are preserved.
        """
        if self.scene.robot is None:
            return
        observations, actions, rewards, terminations = _default_pace_manager_terms()
        if not self.observations:
            self.observations = observations
        if not self.actions:
            self.actions = actions
        if not self.rewards:
            self.rewards = rewards
        if not self.terminations:
            self.terminations = terminations


def make_pace_env_cfg(
    *, robot: EntityCfg, sim2real: PaceCfg, num_envs: int = 4096, play: bool = False
) -> PaceSim2realEnvCfg:
    """Create the common PACE environment and its upstream manager defaults."""
    observations, actions, rewards, terminations = _default_pace_manager_terms()
    cfg = PaceSim2realEnvCfg(
        scene=PaceSim2realSceneCfg(num_envs=num_envs, robot=robot),
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        sim2real=sim2real,
    )
    if play:
        cfg.episode_length_s = 1.0e9
    return cfg
