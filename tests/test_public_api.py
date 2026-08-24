from __future__ import annotations

from pathlib import Path

import mujoco
import torch

from pace_sim2real import CMAESOptimizer, PaceCfg, PaceSim2realEnvCfg, PaceSim2realSceneCfg
from pace_sim2real.assets.anymal_d_asset import JOINT_ORDER, get_spec
from pace_sim2real.tasks.manager_based.pace.anymal_pace_env_cfg import AnymalDPaceCfg
from pace_sim2real.utils import PaceDCMotorCfg, project_root


def test_public_api_and_paths_are_available():
    assert PaceCfg is not None
    assert PaceSim2realEnvCfg is not None
    assert PaceSim2realSceneCfg is not None
    assert CMAESOptimizer is not None
    assert (project_root() / "pyproject.toml").exists()

    cfg = PaceDCMotorCfg(
        joint_names_expr=("joint",),
        saturation_effort=140.0,
        effort_limit=89.0,
        velocity_limit=8.5,
        stiffness=85.0,
        damping=0.6,
        max_delay=10,
    )
    assert cfg.target_names_expr == ("joint",)
    # PACE delay is a custom torque buffer; the native command-delay buffer is
    # intentionally disabled to preserve the upstream physical model.
    assert cfg.max_delay == 10
    assert cfg.delay_max_lag == 0
    assert cfg.delay_hold_prob == 0.0


def test_anymal_asset_and_original_parameter_layout():
    model = get_spec().compile()
    assert [model.joint(i).name for i in range(12)] == list(JOINT_ORDER)
    assert model.nv >= 12

    cfg = AnymalDPaceCfg()
    assert cfg.joint_order == list(JOINT_ORDER)
    assert cfg.bounds_params.shape == (49, 2)
    assert torch.all(cfg.bounds_params[:12, 0] > 0)
    assert torch.equal(cfg.bounds_params[36:48, 0], torch.full((12,), -0.1))


def test_cmaes_log_format_and_stopping(tmp_path: Path):
    joint_order = ["left", "right"]
    bounds = torch.tensor([[0.0, 1.0]] * 9)
    data = {
        "time": torch.linspace(0.0, 0.01, 3),
        "dof_pos": torch.zeros(3, 2),
        "des_dof_pos": torch.zeros(3, 2),
    }
    optimizer = CMAESOptimizer(
        bounds=bounds,
        population_size=4,
        log_dir=tmp_path,
        joint_order=joint_order,
        max_iteration=1,
        data=data,
        device="cpu",
        epsilon=None,
    )
    try:
        for _ in range(3):
            optimizer.tell(torch.zeros(4, 2), torch.zeros(4, 2))
        optimizer.evolve()
        assert optimizer.finished()
        run_dirs = list(tmp_path.iterdir())
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "config.pt").exists()
        assert (run_dirs[0] / "mean_000.pt").exists()
        assert (run_dirs[0] / "best_trajectory.pt").exists()
    finally:
        optimizer.close()


def test_packaged_urdf_compiles_without_visual_meshes():
    # This guards the self-contained MjSpec import path used at runtime.
    spec = get_spec()
    assert isinstance(spec, mujoco.MjSpec)
