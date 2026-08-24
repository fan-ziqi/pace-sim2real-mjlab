"""Shared mjlab utilities for PACE command-line workflows."""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from pace_sim2real.utils import apply_pace_parameters, bind_environment, project_root


def resolve_device(device: str | None) -> str:
    """Resolve PACE's old ``--device`` argument to an available torch device."""
    if device is not None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def import_task_module(task_module: str | None) -> None:
    """Import a user module that registers an mjlab PACE task, if supplied."""
    if task_module is None:
        return
    try:
        importlib.import_module(task_module)
    except ModuleNotFoundError as error:
        if error.name == task_module:
            raise ModuleNotFoundError(
                f"Cannot import PACE task module {task_module!r}. Ensure its parent directory is "
                "on PYTHONPATH or install the custom robot package."
            ) from error
        raise


def extract_task_module(args: list[str]) -> tuple[str | None, list[str]]:
    """Remove PACE's wrapper-only custom-task import option from CLI arguments."""
    task_module: str | None = None
    remaining: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in {"--task-module", "--task_module"}:
            if index + 1 >= len(args):
                raise SystemExit(f"{arg} requires an importable module name")
            module = args[index + 1]
            index += 2
        elif arg.startswith("--task-module=") or arg.startswith("--task_module="):
            module = arg.split("=", 1)[1]
            index += 1
        else:
            remaining.append(arg)
            index += 1
            continue
        if task_module is not None and task_module != module:
            raise SystemExit("only one --task-module may be supplied")
        task_module = module
    return task_module, remaining


def make_env(
    task: str, num_envs: int, device: str, *, play: bool = False, task_module: str | None = None
) -> ManagerBasedRlEnv:
    """Load a registered PACE task and construct its vectorized mjlab environment."""
    import pace_sim2real.tasks  # noqa: F401

    import_task_module(task_module)
    cfg = load_env_cfg(task, play=play)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    # Lets CMAESOptimizer accept the original PACE positional
    # update_simulator(articulation, joint_ids, initial_position) signature.
    bind_environment(env)
    return env


def pace_joint_ids(robot, joint_order: list[str] | tuple[str, ...], device: str) -> torch.Tensor:
    """Resolve PACE's canonical joint ordering against the loaded robot."""
    missing = [name for name in joint_order if name not in robot.joint_names]
    if missing:
        raise ValueError(f"robot is missing PACE joints: {missing}")
    return torch.tensor(
        [robot.joint_names.index(name) for name in joint_order], device=device, dtype=torch.long
    )


def pace_position_action(
    env: ManagerBasedRlEnv, robot, joint_ids: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Scatter PACE fitted-joint targets into mjlab's complete action vector.

    PACE excites ``sim2real.joint_order`` only, while the upstream-compatible
    default action term controls every actuator joint.  Additional joints on a
    custom robot are deliberately held at zero rather than making ``env.step``
    receive a too-short vector.
    """
    del robot  # Joint IDs are resolved against the same robot by the caller.
    targets = targets.to(device=env.device, dtype=torch.float32)
    if targets.shape != (env.num_envs, len(joint_ids)):
        raise ValueError(
            "PACE targets must have shape (num_envs, fitted_joints); got "
            f"{tuple(targets.shape)}, expected ({env.num_envs}, {len(joint_ids)})"
        )
    try:
        term_index = env.action_manager.active_terms.index("joint_pos")
        term = env.action_manager.get_term("joint_pos")
    except (AttributeError, ValueError, KeyError) as error:
        raise ValueError(
            "PACE requires actions.joint_pos to control every name in "
            "sim2real.joint_order; use make_pace_env_cfg or configure an equivalent term."
        ) from error
    target_to_action = {
        int(joint_id): index for index, joint_id in enumerate(term.target_ids.tolist())
    }
    missing = [
        int(joint_id) for joint_id in joint_ids.tolist() if int(joint_id) not in target_to_action
    ]
    if missing:
        raise ValueError(f"actions.joint_pos does not control PACE joint IDs: {missing}")
    action_offset = sum(
        env.action_manager.get_term(name).action_dim
        for name in env.action_manager.active_terms[:term_index]
    )
    action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    action_columns = [
        action_offset + target_to_action[int(joint_id)] for joint_id in joint_ids.tolist()
    ]
    action[:, action_columns] = targets
    return action


def prepare_pace_model(
    env: ManagerBasedRlEnv,
    robot,
    joint_ids: torch.Tensor,
    *,
    armature: torch.Tensor,
    damping: torch.Tensor,
    friction: torch.Tensor,
    bias: torch.Tensor,
    delay: torch.Tensor,
    initial_encoder_position: torch.Tensor | None = None,
) -> None:
    """Apply PACE parameters through the shared validated model adapter."""
    apply_pace_parameters(
        env,
        robot,
        joint_ids,
        armature=armature,
        damping=damping,
        friction=friction,
        bias=bias,
        delay=delay,
        initial_encoder_position=initial_encoder_position,
    )


def validate_pace_trajectory_data(
    data: dict[str, torch.Tensor], *, physics_dt: float, joint_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate PACE's measured encoder trajectory and its simulation timebase.

    PACE advances the simulator exactly once for every saved sample.  Accepting
    a trajectory recorded at another rate silently changes the physical meaning
    of fitted damping and command latency, so this is deliberately strict.
    """
    required = {"time", "dof_pos", "des_dof_pos"}
    if not isinstance(data, dict):
        raise ValueError("excitation data must be a dictionary of tensor arrays")
    missing = required.difference(data)
    if missing:
        raise KeyError(f"excitation data is missing keys: {sorted(missing)}")
    time = torch.as_tensor(data["time"], dtype=torch.float32).reshape(-1)
    measured = torch.as_tensor(data["dof_pos"], dtype=torch.float32)
    target = torch.as_tensor(data["des_dof_pos"], dtype=torch.float32)
    if measured.ndim != 2 or target.ndim != 2:
        raise ValueError("dof_pos and des_dof_pos must both have shape [samples, joints]")
    if measured.shape != target.shape or measured.shape != (len(time), joint_count):
        raise ValueError(
            "data must contain matching [samples, PACE joints] dof_pos/des_dof_pos arrays "
            f"and a same-length time array; got time={tuple(time.shape)}, "
            f"dof_pos={tuple(measured.shape)}, des_dof_pos={tuple(target.shape)}"
        )
    if len(time) == 0 or not (
        torch.isfinite(time).all()
        and torch.isfinite(measured).all()
        and torch.isfinite(target).all()
    ):
        raise ValueError(
            "PACE trajectory time, positions, and targets must be finite and non-empty"
        )
    if len(time) > 1:
        intervals = torch.diff(time)
        if torch.any(intervals <= 0):
            raise ValueError(
                "PACE trajectory time must be strictly increasing and expressed in seconds"
            )
        tolerance = max(1.0e-7, physics_dt * 1.0e-3)
        expected = torch.full_like(intervals, physics_dt)
        if not torch.allclose(intervals, expected, rtol=1.0e-3, atol=tolerance):
            raise ValueError(
                "PACE trajectory sampling interval must equal the simulator physics_dt "
                f"({physics_dt:g} s); resample real data before fitting. "
                f"Observed interval range: [{intervals.min().item():g}, {intervals.max().item():g}] s"
            )
    return time, measured, target


def estimate_cmaes_trajectory_memory(
    *, population_size: int, samples: int, joint_count: int, bytes_per_value: int = 4
) -> int:
    """Return the lower-bound bytes reserved only for CMA-ES position history."""
    return population_size * samples * joint_count * bytes_per_value


def data_path(data_dir: str) -> Path:
    """Resolve a PACE-relative data file path, keeping ``PACE_ROOT`` support."""
    return project_root() / "data" / data_dir
