"""Validated application of PACE physical parameters to an mjlab world."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from mjlab.managers.event_manager import RecomputeLevel

from .delay import pace_delay_steps
from .pace_actuator import PaceDCMotor, update_pace_encoder_bias


def _matrix(
    name: str,
    value: torch.Tensor,
    *,
    shape: tuple[int, int],
    device: str | torch.device,
    nonnegative: bool = False,
) -> torch.Tensor:
    """Convert and validate a per-world, per-joint physical parameter."""
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and torch.any(tensor < 0):
        raise ValueError(f"{name} must be non-negative")
    return tensor


def _delay_steps(
    delay: torch.Tensor,
    *,
    num_envs: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Validate the continuous PACE delay coordinate before upstream truncation."""
    delay_tensor = torch.as_tensor(delay, dtype=torch.float32, device=device)
    if tuple(delay_tensor.shape) == (num_envs, 1):
        delay_tensor = delay_tensor[:, 0]
    elif tuple(delay_tensor.shape) != (num_envs,):
        raise ValueError(
            f"delay must have shape (num_envs,) or (num_envs, 1); got {tuple(delay_tensor.shape)}"
        )
    if not torch.isfinite(delay_tensor).all() or torch.any(delay_tensor < 0):
        raise ValueError("delay must contain finite, non-negative values")
    # The explicit conversion deliberately preserves Isaac Lab's integer cast:
    # 5.8 physics steps means five steps, rather than six.
    return pace_delay_steps(delay_tensor, device=torch.device(device))


def _validate_pace_actuators(
    robot: Any, joint_ids: torch.Tensor, lags: torch.Tensor
) -> list[PaceDCMotor]:
    """Ensure every fitted joint has a PACE torque-delay implementation."""
    requested = {int(joint_id) for joint_id in joint_ids.tolist()}
    covered: set[int] = set()
    actuators: list[PaceDCMotor] = []
    max_lag = int(lags.max().item())
    for actuator in robot.actuators:
        targets = {int(target_id) for target_id in actuator.target_ids}
        if not requested.intersection(targets):
            continue
        if not isinstance(actuator, PaceDCMotor):
            raise ValueError(
                "every PACE fitted joint must be controlled by PaceDCMotor; "
                f"found {type(actuator).__name__} for joint IDs "
                f"{sorted(requested.intersection(targets))}"
            )
        if max_lag > actuator.cfg.max_delay:
            raise ValueError(
                "PACE delay candidate exceeds the actuator's max_delay "
                f"({max_lag} > {actuator.cfg.max_delay})"
            )
        covered.update(requested.intersection(targets))
        actuators.append(actuator)
    missing = sorted(requested.difference(covered))
    if missing:
        raise ValueError(f"no PaceDCMotor controls PACE joint IDs: {missing}")
    return actuators


def apply_pace_parameters(
    env: Any,
    robot: Any,
    joint_ids: torch.Tensor | Sequence[int],
    *,
    armature: torch.Tensor,
    damping: torch.Tensor,
    friction: torch.Tensor,
    bias: torch.Tensor,
    delay: torch.Tensor,
    initial_encoder_position: torch.Tensor | None = None,
) -> None:
    """Apply one validated PACE candidate per mjlab world.

    The function is the single physical-model boundary for collection and
    optimization.  It validates all inputs, including PACE actuator coverage
    and torque-buffer capacity, before mutating any MuJoCo model field.
    """
    device = env.device
    joint_ids = torch.as_tensor(joint_ids, device=device, dtype=torch.long).reshape(-1)
    if joint_ids.numel() == 0:
        raise ValueError("joint_ids must not be empty")
    if torch.any(joint_ids < 0) or torch.any(joint_ids >= len(robot.joint_names)):
        raise ValueError("joint_ids contains an out-of-range robot joint ID")
    if torch.unique(joint_ids).numel() != joint_ids.numel():
        raise ValueError("joint_ids must not contain duplicates")

    shape = (env.num_envs, joint_ids.numel())
    armature = _matrix("armature", armature, shape=shape, device=device, nonnegative=True)
    damping = _matrix("damping", damping, shape=shape, device=device, nonnegative=True)
    friction = _matrix("friction", friction, shape=shape, device=device, nonnegative=True)
    bias = _matrix("bias", bias, shape=shape, device=device)
    lags = _delay_steps(delay, num_envs=env.num_envs, device=device)
    initial_position: torch.Tensor | None = None
    if initial_encoder_position is not None:
        initial_position = _matrix(
            "initial_encoder_position",
            initial_encoder_position,
            shape=shape,
            device=device,
        )
    actuators = _validate_pace_actuators(robot, joint_ids, lags)

    fields = ("dof_armature", "dof_damping", "dof_frictionloss")
    missing = tuple(name for name in fields if name not in env.sim.expanded_fields)
    if missing:
        env.sim.expand_model_fields(missing)
    dof_ids = robot.indexing.joint_v_adr[joint_ids]
    env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)

    def write(field: torch.Tensor, values: torch.Tensor) -> None:
        if field.ndim == 1:
            field[dof_ids] = values[0]
        else:
            field[env_ids[:, None], dof_ids] = values

    write(env.sim.model.dof_armature, armature)
    write(env.sim.model.dof_damping, damping)
    write(env.sim.model.dof_frictionloss, friction)
    env.sim.recompute_constants(RecomputeLevel.set_const_0)
    update_pace_encoder_bias(robot, joint_ids, bias)
    if initial_position is not None:
        robot.write_joint_state_to_sim(
            initial_position + bias,
            torch.zeros_like(initial_position),
            joint_ids=joint_ids,
        )
    for actuator in actuators:
        actuator.set_lags(lags)
        # Candidate updates must not consume torque values produced under a
        # previous physical model, matching the upstream actuator reset.
        actuator.reset()
    env.sim.forward()
