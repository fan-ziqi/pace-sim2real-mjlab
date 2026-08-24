"""Regression tests for public API boundaries found in code review."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pace_sim2real import CMAESOptimizer
from pace_sim2real.scripts._legacy_cli import translate_play_args
from pace_sim2real.utils import PaceDCMotorCfg, load_pace_artifact


class UnsupportedPayload:
    """Pickle-only object used to assert that safe loading rejects globals."""


def _optimizer(tmp_path: Path) -> CMAESOptimizer:
    return CMAESOptimizer(
        bounds=torch.tensor([[0.0, 1.0]] * 5),
        population_size=4,
        log_dir=tmp_path,
        joint_order=["joint"],
        max_iteration=1,
        data={
            "time": torch.tensor([0.0]),
            "dof_pos": torch.zeros(1, 1),
            "des_dof_pos": torch.zeros(1, 1),
        },
        device="cpu",
    )


def test_cmaes_tell_rejects_bad_shape_without_mutating_state(tmp_path: Path) -> None:
    optimizer = _optimizer(tmp_path)
    try:
        scores_before = optimizer.scores.clone()
        with pytest.raises(ValueError, match="shape"):
            optimizer.tell(torch.zeros(4), torch.zeros(4))
        assert optimizer.scores_counter == 0
        assert torch.equal(optimizer.scores, scores_before)
    finally:
        optimizer.close()


def test_cmaes_runs_never_share_a_timestamp_directory(tmp_path: Path) -> None:
    first = _optimizer(tmp_path)
    second = _optimizer(tmp_path)
    try:
        assert first.writer.log_dir != second.writer.log_dir
        assert len(list(tmp_path.iterdir())) == 2
    finally:
        first.close()
        second.close()


def test_cmaes_rejects_invalid_public_constructor_boundaries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_iteration"):
        CMAESOptimizer(
            bounds=torch.tensor([[0.0, 1.0]] * 5),
            population_size=4,
            log_dir=tmp_path,
            joint_order=["joint"],
            max_iteration=0,
            data={
                "time": torch.tensor([0.0]),
                "dof_pos": torch.zeros(1, 1),
                "des_dof_pos": torch.zeros(1, 1),
            },
            device="cpu",
        )


def test_pace_actuator_alias_conflicts_are_rejected() -> None:
    common = {
        "saturation_effort": 1.0,
        "effort_limit": 1.0,
        "velocity_limit": 1.0,
        "stiffness": 1.0,
        "damping": 1.0,
    }
    with pytest.raises(ValueError, match="must agree"):
        PaceDCMotorCfg(joint_names_expr=("left",), target_names_expr=("right",), **common)
    with pytest.raises(ValueError, match="max_delay"):
        PaceDCMotorCfg(joint_names_expr=("joint",), max_delay=2, delay_max_lag=7, **common)
    with pytest.raises(ValueError, match="max_delay"):
        PaceDCMotorCfg(joint_names_expr=("joint",), delay_max_lag=-1, **common)


def test_play_translation_validates_ignored_training_argument_values() -> None:
    with pytest.raises(SystemExit, match="--seed requires a value"):
        translate_play_args(["--task", "Isaac-Pace-Anymal-D-v0", "--seed"])


def test_safe_loader_accepts_tensor_artifacts_and_rejects_pickle_globals(tmp_path: Path) -> None:
    valid = tmp_path / "valid.pt"
    torch.save({"value": torch.tensor([1.0])}, valid)
    assert torch.equal(load_pace_artifact(valid)["value"], torch.tensor([1.0]))

    unsafe = tmp_path / "unsafe.pt"
    torch.save(UnsupportedPayload(), unsafe)
    with pytest.raises(ValueError, match="Cannot safely load"):
        load_pace_artifact(unsafe)


def test_importing_pace_has_no_mjlab_global_constructor_side_effect() -> None:
    script = """
from mjlab.envs import ManagerBasedRlEnv
before = ManagerBasedRlEnv.__init__
import pace_sim2real
assert ManagerBasedRlEnv.__init__ is before
"""
    subprocess.run([sys.executable, "-c", script], check=True)
