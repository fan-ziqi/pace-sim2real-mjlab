from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from pace_sim2real import CMAESOptimizer
from pace_sim2real.assets.anymal_d_asset import get_anymal_d_robot_cfg
from pace_sim2real.scripts._common import (
    extract_task_module,
    pace_position_action,
    validate_pace_trajectory_data,
)
from pace_sim2real.scripts._legacy_cli import translate_play_args, translate_train_args
from pace_sim2real.scripts.data_collection import chirp_trajectory
from pace_sim2real.tasks.manager_based.pace.agents.rsl_rl_ppo_cfg import PPORunnerCfg
from pace_sim2real.tasks.manager_based.pace.anymal_pace_env_cfg import (
    AnymalDPaceCfg,
    AnymalDPaceEnvCfg,
)
from pace_sim2real.tasks.manager_based.pace.pace_sim2real_env_cfg import (
    PaceSim2realEnvCfg,
    PaceSim2realSceneCfg,
)
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real.utils.delay import pace_delay_steps


def test_cmaes_bias_residual_uses_raw_simulated_position(tmp_path: Path):
    data = {
        "time": torch.tensor([0.0, 0.0025]),
        "dof_pos": torch.zeros(2, 1),
        "des_dof_pos": torch.zeros(2, 1),
    }
    optimizer = CMAESOptimizer(
        bounds=torch.tensor([[0.0, 1.0]] * 5),
        population_size=4,
        log_dir=tmp_path,
        joint_order=["joint"],
        max_iteration=1,
        data=data,
        device="cpu",
    )
    try:
        bias = torch.full((4, 1), 0.05)
        optimizer.sim_params[:, optimizer.bias_idx] = bias
        # Real data is encoder-frame q-bias.  Passing raw q to tell() must
        # produce zero loss at the true candidate instead of subtracting bias twice.
        optimizer.tell(bias, torch.zeros_like(bias))
        assert torch.equal(optimizer.scores, torch.zeros(4))
    finally:
        optimizer.close()


def test_timebase_validation_and_collector_grid():
    time, _ = chirp_trajectory(
        duration=0.02,
        dt=0.0025,
        joint_count=1,
        min_frequency=0.1,
        max_frequency=1.0,
        device="cpu",
    )
    data = {
        "time": time,
        "dof_pos": torch.zeros(len(time), 1),
        "des_dof_pos": torch.zeros(len(time), 1),
    }
    validated, _, _ = validate_pace_trajectory_data(data, physics_dt=0.0025, joint_count=1)
    assert torch.equal(validated, time)
    data["time"] = time * 2.0
    with pytest.raises(ValueError, match="sampling interval"):
        validate_pace_trajectory_data(data, physics_dt=0.0025, joint_count=1)


def test_legacy_config_and_python_interfaces_are_preserved():
    cfg = PaceDCMotorCfg(
        joint_names_expr=("left", "right"),
        saturation_effort={".*": 140.0},
        effort_limit={".*": 89.0},
        velocity_limit=8.5,
        stiffness={".*": 85.0},
        damping=[0.6, 0.7],
        encoder_bias=[0.01, -0.02],
        friction={".*": 0.1},
        dynamic_friction={".*": 0.1},
        viscous_friction={".*": 0.2},
        max_delay=10,
    )
    assert cfg.joint_names_expr == ("left", "right")
    assert cfg.max_delay == 10
    assert cfg.delay_max_lag == 0  # native command delay stays disabled

    env_cfg = AnymalDPaceEnvCfg()
    assert isinstance(env_cfg, AnymalDPaceEnvCfg)
    assert env_cfg.actions.joint_pos is env_cfg.actions["joint_pos"]
    assert env_cfg.observations.policy is env_cfg.observations["policy"]
    assert env_cfg.sim.dt == 0.0025
    runner = PPORunnerCfg()
    assert runner.experiment_name == "pace_sim2real"
    assert runner.num_steps_per_env == 16
    assert runner.max_iterations == 150
    assert runner.logger == "tensorboard"

    # The base class, which custom robots inherit, must be an immediately
    # usable environment rather than an empty manager shell.
    custom_base = PaceSim2realEnvCfg(
        scene=PaceSim2realSceneCfg(robot=get_anymal_d_robot_cfg()), sim2real=AnymalDPaceCfg()
    )
    assert set(custom_base.actions) == {"joint_pos"}
    assert {"actor", "critic", "policy"}.issubset(custom_base.observations)
    assert set(custom_base.rewards) == {"dof_pos_limits"}
    assert set(custom_base.terminations) == {"time_out"}

    serialized_actuator = asdict(env_cfg)["scene"]["robot"]["articulation"]["actuators"][0]
    assert serialized_actuator["encoder_bias"] == 0.0
    assert serialized_actuator["max_delay"] == 10
    assert serialized_actuator["pace_saturation_effort"] == 140.0


def test_legacy_rsl_cli_translation(tmp_path: Path):
    train = translate_train_args(
        [
            "--task",
            "Isaac-Pace-Anymal-D-v0",
            "--num_envs",
            "4",
            "--max_iterations",
            "1",
            "--seed",
            "7",
            "--logger",
            "tensorboard",
            "--device",
            "cpu",
            "--video",
            "--headless",
        ]
    )
    assert train == [
        "Isaac-Pace-Anymal-D-v0",
        "--env.scene.num-envs",
        "4",
        "--agent.max-iterations",
        "1",
        "--agent.seed",
        "7",
        "--agent.logger",
        "tensorboard",
        "--gpu-ids",
        "None",
        "--video",
        "True",
    ]
    checkpoint = tmp_path / "model_1.pt"
    checkpoint.touch()
    play = translate_play_args(
        [
            "--task=Isaac-Pace-Anymal-D-v0",
            "--num_envs=1",
            "--checkpoint",
            str(checkpoint),
            "--agent",
            "rsl_rl_cfg_entry_point",
            "--video",
            "--disable_fabric",
        ]
    )
    assert "--checkpoint-file" in play
    assert play[play.index("--agent") + 1] == "trained"


def test_delay_candidates_use_upstream_truncation() -> None:
    assert torch.equal(
        pace_delay_steps(torch.tensor([0.0, 5.2, 5.8, 10.0])),
        torch.tensor([0, 5, 5, 10]),
    )


def test_task_module_flag_is_removed_before_mjlab_cli_parsing() -> None:
    module, remaining = extract_task_module(
        ["--task", "Custom-Pace-v0", "--task-module=my_robot.tasks", "--num_envs", "4"]
    )
    assert module == "my_robot.tasks"
    assert remaining == ["--task", "Custom-Pace-v0", "--num_envs", "4"]


def test_fitted_targets_are_scattered_into_full_custom_robot_action() -> None:
    class ActionTerm:
        target_ids = torch.tensor([2, 0, 3, 1])
        action_dim = 4

    class ActionManager:
        active_terms = ["extra", "joint_pos"]
        total_action_dim = 5

        @staticmethod
        def get_term(name: str):
            return ActionTerm() if name == "joint_pos" else type("Extra", (), {"action_dim": 1})()

    class Env:
        num_envs = 2
        device = "cpu"
        action_manager = ActionManager()

    action = pace_position_action(
        Env(), None, torch.tensor([0, 1]), torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )
    assert torch.equal(action, torch.tensor([[0.0, 0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 3.0, 0.0, 4.0]]))
