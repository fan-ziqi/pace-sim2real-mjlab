"""PACE actuator behavior implemented using mjlab's native DC motor model."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Sequence

import torch
from mjlab.actuator import DcMotorActuator
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.utils.buffers import DelayBuffer
from mjlab.utils.spec import create_motor_actuator

if TYPE_CHECKING:
    from .pace_actuator_cfg import PaceDCMotorCfg


class PaceDCMotor(DcMotorActuator):
    """mjlab DC motor with PACE encoder-bias and fixed-delay helpers.

    mjlab's :class:`JointPositionAction` already applies encoder calibration
    error before an actuator sees a position target.  PACE differs from a
    standard mjlab actuator in one essential way: it delays the *computed and
    clipped torque*, not the input command.  A custom ``compute`` makes this
    actuator intentionally non-fused, preserving the original stateful PACE
    semantics for every world.
    """

    cfg: "PaceDCMotorCfg"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._torque_delay_buffer: DelayBuffer | None = None
        # These are the *current* PACE parameters, not merely construction
        # defaults.  CMA-ES writes a distinct candidate per world and mjlab
        # resets worlds between generations, so actuator reset must preserve
        # the candidate that was most recently applied.
        self._pace_encoder_bias: torch.Tensor | None = None
        self._pace_lags: torch.Tensor | None = None

    @staticmethod
    def _resolve_parameter(
        value: Mapping[str, float] | Sequence[float] | float | None,
        target_names: Sequence[str],
        *,
        default: float,
        device: str,
        num_envs: int,
    ) -> torch.Tensor:
        """Resolve Isaac-style scalar/list/regex parameter values per target."""
        count = len(target_names)
        if value is None:
            values = [default] * count
        elif isinstance(value, (float, int)):
            values = [float(value)] * count
        elif isinstance(value, Mapping):
            values = [default] * count
            for expression, mapped_value in value.items():
                matches = [
                    index
                    for index, name in enumerate(target_names)
                    if re.fullmatch(expression, name) is not None
                ]
                if not matches:
                    raise ValueError(
                        f"PACE actuator expression {expression!r} matches no targets: "
                        f"{list(target_names)}"
                    )
                for index in matches:
                    values[index] = float(mapped_value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != count:
                raise ValueError(
                    f"PACE actuator parameter has {len(value)} values; expected {count} "
                    f"for {list(target_names)}"
                )
            values = [float(item) for item in value]
        else:
            raise TypeError(f"Unsupported PACE actuator parameter type: {type(value)!r}")
        return (
            torch.tensor(values, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .repeat(num_envs, 1)
        )

    def edit_spec(self, spec, target_names: list[str]) -> None:
        """Create motors with the original per-joint passive PACE values."""
        armature = self._resolve_parameter(
            self.cfg.pace_armature, target_names, default=0.0, device="cpu", num_envs=1
        )[0]
        friction = self._resolve_parameter(
            self.cfg.pace_frictionloss, target_names, default=0.0, device="cpu", num_envs=1
        )[0]
        viscous = self._resolve_parameter(
            self.cfg.pace_viscous_damping, target_names, default=0.0, device="cpu", num_envs=1
        )[0]
        for index, target_name in enumerate(target_names):
            actuator = create_motor_actuator(
                spec,
                target_name,
                effort_limit=float(
                    self._resolve_parameter(
                        self.cfg.pace_effort_limit,
                        target_names,
                        default=float("inf"),
                        device="cpu",
                        num_envs=1,
                    )[0, index]
                ),
                armature=float(armature[index]),
                frictionloss=float(friction[index]),
                viscous_damping=float(viscous[index]),
                transmission_type=self.cfg.transmission_type,
            )
            self._mjs_actuators.append(actuator)

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        num_envs = data.nworld
        names = self.target_names
        self.stiffness = self._resolve_parameter(
            self.cfg.pace_stiffness, names, default=0.0, device=device, num_envs=num_envs
        )
        self.damping = self._resolve_parameter(
            self.cfg.pace_damping, names, default=0.0, device=device, num_envs=num_envs
        )
        self.force_limit = self._resolve_parameter(
            self.cfg.pace_effort_limit,
            names,
            default=float("inf"),
            device=device,
            num_envs=num_envs,
        )
        self.saturation_effort = self._resolve_parameter(
            self.cfg.pace_saturation_effort, names, default=0.0, device=device, num_envs=num_envs
        )
        self.velocity_limit_motor = self._resolve_parameter(
            self.cfg.pace_velocity_limit, names, default=1.0, device=device, num_envs=num_envs
        )
        self.default_stiffness = self.stiffness.clone()
        self.default_damping = self.damping.clone()
        self.default_force_limit = self.force_limit.clone()
        # Original PACE defines q_encoder = q - bias, whereas mjlab uses
        # q_encoder = q + encoder_bias.
        bias = self._resolve_parameter(
            self.cfg.encoder_bias, names, default=0.0, device=device, num_envs=num_envs
        )
        # EntityData is allocated after all actuators initialize.  Apply this
        # on the first reset, before the action manager turns encoder-frame
        # commands into simulation-frame targets.
        self._pace_encoder_bias = bias
        if self.cfg.max_delay > 0:
            self._torque_delay_buffer = DelayBuffer(
                min_lag=0,
                max_lag=self.cfg.max_delay,
                batch_size=num_envs,
                device=device,
                # CMA-ES uses set_lags() for candidate values.  The original
                # task uses max_delay until an explicit override arrives.
                hold_prob=1.0,
            )
            self._pace_lags = torch.full(
                (num_envs,), self.cfg.max_delay, device=device, dtype=torch.long
            )
            self._torque_delay_buffer.set_lags(self._pace_lags)

    def apply_delay(self, cmd: ActuatorCmd) -> ActuatorCmd:
        """Disable mjlab command delay; PACE delays torques in ``compute``."""
        return cmd

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        """Compute current-state DC-PD torque, then apply PACE torque delay."""
        assert self.stiffness is not None
        assert self.damping is not None
        assert self.force_limit is not None
        assert self.saturation_effort is not None
        assert self.velocity_limit_motor is not None
        torque = DcMotorActuator.control_law(
            {
                "stiffness": self.stiffness,
                "damping": self.damping,
                "force_limit": self.force_limit,
                "saturation_effort": self.saturation_effort,
                "velocity_limit_motor": self.velocity_limit_motor,
            },
            cmd,
        )
        if self._torque_delay_buffer is None:
            return torque
        self._torque_delay_buffer.append(torque)
        return self._torque_delay_buffer.compute()

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if self._pace_encoder_bias is not None:
            ids = slice(None) if env_ids is None else env_ids
            if isinstance(ids, slice):
                self.entity.data.encoder_bias[ids, self.target_ids] = -self._pace_encoder_bias[ids]
            else:
                env_tensor = torch.as_tensor(
                    ids, dtype=torch.long, device=self.entity.data.encoder_bias.device
                ).reshape(-1)
                self.entity.data.encoder_bias[
                    env_tensor[:, None], self.target_ids
                ] = -self._pace_encoder_bias[env_tensor]
        if self._torque_delay_buffer is not None:
            self._torque_delay_buffer.reset(env_ids)
            assert self._pace_lags is not None
            ids = slice(None) if env_ids is None else env_ids
            self._torque_delay_buffer.set_lags(self._pace_lags[ids], env_ids)

    def update_encoder_bias(self, encoder_bias: torch.Tensor) -> None:
        """Set PACE-format encoder bias for every actuator target."""
        bias = encoder_bias.to(device=self.entity.data.encoder_bias.device)
        if bias.ndim == 1:
            bias = bias.unsqueeze(0)
        if bias.shape[-1] != len(self.target_ids):
            raise ValueError(
                f"encoder_bias has {bias.shape[-1]} columns; expected {len(self.target_ids)}"
            )
        assert self._pace_encoder_bias is not None
        if bias.shape[0] == 1:
            bias = bias.expand_as(self._pace_encoder_bias)
        elif bias.shape != self._pace_encoder_bias.shape:
            raise ValueError(
                "encoder_bias must have one row or one row per environment; got "
                f"{tuple(bias.shape)}, expected {tuple(self._pace_encoder_bias.shape)}"
            )
        self._pace_encoder_bias.copy_(bias)
        self.entity.data.encoder_bias[:, self.target_ids] = -self._pace_encoder_bias

    def update_time_lags(
        self, delay: int | torch.Tensor, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        """Set fixed PACE torque latency in simulation timesteps."""
        raw_lags = torch.as_tensor(delay, device=self.entity.data.encoder_bias.device)
        if not torch.isfinite(raw_lags).all() or torch.any(raw_lags < 0):
            raise ValueError("PACE time lags must be finite and non-negative")
        lags = raw_lags.to(dtype=torch.long)
        if lags.ndim > 1:
            lags = lags.squeeze(-1)
        self.set_lags(lags, env_ids=env_ids)

    def set_lags(
        self, lags: torch.Tensor, env_ids: Sequence[int] | torch.Tensor | slice | None = None
    ) -> None:
        """Set PACE torque-buffer lags (same public mjlab method name)."""
        device = self.entity.data.encoder_bias.device
        raw_values = torch.as_tensor(lags, device=device)
        if not torch.isfinite(raw_values).all() or torch.any(raw_values < 0):
            raise ValueError("PACE time lags must be finite and non-negative")
        values = raw_values.to(dtype=torch.long).reshape(-1)
        if self._torque_delay_buffer is None:
            if torch.any(values != 0):
                raise RuntimeError(
                    "cannot apply a non-zero PACE delay: this actuator was configured with max_delay=0"
                )
            return
        assert self._pace_lags is not None
        ids = slice(None) if env_ids is None else env_ids
        expected_count = self._pace_lags[ids].numel()
        if values.numel() == 1 and expected_count != 1:
            values = values.expand(expected_count)
        if values.numel() != expected_count:
            raise ValueError(
                f"expected one lag or {expected_count} lags for the selected environments; "
                f"got {values.numel()}"
            )
        if torch.any(values > self.cfg.max_delay):
            raise ValueError(
                f"PACE lag exceeds actuator max_delay ({int(values.max().item())} > {self.cfg.max_delay})"
            )
        self._pace_lags[ids] = values
        self._torque_delay_buffer.set_lags(values, env_ids)


def update_pace_encoder_bias(entity: Any, joint_ids: torch.Tensor, bias: torch.Tensor) -> None:
    """Apply PACE-format bias and retain it through subsequent world resets.

    The requested joints may belong to an actuator that also owns non-PACE
    joints.  Keep those existing values while sending the complete actuator
    vector through ``update_encoder_bias`` so the actuator's persistent state
    and ``EntityData`` cannot diverge.
    """
    joint_ids = joint_ids.to(device=entity.data.encoder_bias.device, dtype=torch.long)
    bias = bias.to(device=entity.data.encoder_bias.device, dtype=torch.float32)
    if bias.ndim == 1:
        bias = bias.unsqueeze(0)
    if bias.shape[0] == 1:
        bias = bias.expand(entity.data.encoder_bias.shape[0], -1)
    if bias.shape != (entity.data.encoder_bias.shape[0], len(joint_ids)):
        raise ValueError(
            f"bias must have shape (num_envs, requested_joints); got {tuple(bias.shape)}"
        )

    requested = {int(joint_id): index for index, joint_id in enumerate(joint_ids.tolist())}
    applied: set[int] = set()
    for actuator in entity.actuators:
        local = [
            index
            for index, target_id in enumerate(actuator.target_ids)
            if int(target_id) in requested
        ]
        if not local:
            continue
        if isinstance(actuator, PaceDCMotor):
            full_bias = -entity.data.encoder_bias[:, actuator.target_ids].clone()
            for local_index in local:
                target_id = int(actuator.target_ids[local_index])
                full_bias[:, local_index] = bias[:, requested[target_id]]
                applied.add(target_id)
            actuator.update_encoder_bias(full_bias)

    # Preserve the former direct EntityData behavior for non-PACE actuators.
    # PACE's default asset always takes the persistent branch above.
    missing = [
        index for index, joint_id in enumerate(joint_ids.tolist()) if joint_id not in applied
    ]
    if missing:
        entity.data.encoder_bias[:, joint_ids[missing]] = -bias[:, missing]
