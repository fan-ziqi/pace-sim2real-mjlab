"""mjlab actuator configuration with the public PACE naming convention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mjlab.actuator import DcMotorActuatorCfg

from .pace_actuator import PaceDCMotor


def _same_value(left: object, right: object) -> bool:
    """Compare configuration aliases without silently choosing a conflict."""
    return left == right


def _select_alias(name: str, *values: object | None) -> object | None:
    """Select one equivalent alias value or reject an ambiguous config."""
    supplied = [value for value in values if value is not None]
    if not supplied:
        return None
    if any(not _same_value(supplied[0], value) for value in supplied[1:]):
        raise ValueError(f"conflicting values supplied for PACE {name} aliases")
    return supplied[0]


@dataclass(init=False)
class PaceDCMotorCfg(DcMotorActuatorCfg):
    """PACE-compatible DC motor configuration for mjlab.

    It accepts the Isaac Lab field ``joint_names_expr`` as an alias for
    mjlab's ``target_names_expr``.  Encoder bias lives on mjlab's entity data
    rather than inside an actuator, so PACE's positive-bias convention is
    converted by :class:`PaceDCMotor` and the CMA-ES adapter.
    """

    # Isaac Lab accepted scalar, per-joint sequence, and regex-value mappings
    # for all of these values.  mjlab's native dataclasses intentionally use
    # scalar fields, so the PACE layer stores the original representation and
    # resolves it once target names are known.
    encoder_bias: Mapping[str, float] | Sequence[float] | float | None = 0.0
    max_delay: int = 0
    pace_saturation_effort: Mapping[str, float] | Sequence[float] | float = field(
        default=0.0, init=False
    )
    pace_effort_limit: Mapping[str, float] | Sequence[float] | float = field(
        default=float("inf"), init=False
    )
    pace_velocity_limit: Mapping[str, float] | Sequence[float] | float = field(
        default=1.0, init=False
    )
    pace_stiffness: Mapping[str, float] | Sequence[float] | float = field(default=0.0, init=False)
    pace_damping: Mapping[str, float] | Sequence[float] | float = field(default=0.0, init=False)
    pace_armature: Mapping[str, float] | Sequence[float] | float | None = field(
        default=None, init=False
    )
    pace_frictionloss: Mapping[str, float] | Sequence[float] | float | None = field(
        default=None, init=False
    )
    pace_viscous_damping: Mapping[str, float] | Sequence[float] | float | None = field(
        default=None, init=False
    )

    def __init__(
        self,
        *,
        joint_names_expr: Sequence[str] | str | None = None,
        target_names_expr: Sequence[str] | str | None = None,
        saturation_effort: Mapping[str, float] | Sequence[float] | float,
        effort_limit: Mapping[str, float] | Sequence[float] | float,
        velocity_limit: Mapping[str, float] | Sequence[float] | float,
        stiffness: Mapping[str, float] | Sequence[float] | float,
        damping: Mapping[str, float] | Sequence[float] | float,
        encoder_bias: Mapping[str, float] | Sequence[float] | float | None = 0.0,
        max_delay: int = 0,
        armature: Mapping[str, float] | Sequence[float] | float | None = None,
        friction: Mapping[str, float] | Sequence[float] | float | None = None,
        dynamic_friction: Mapping[str, float] | Sequence[float] | float | None = None,
        viscous_friction: Mapping[str, float] | Sequence[float] | float | None = None,
        frictionloss: Mapping[str, float] | Sequence[float] | float | None = None,
        viscous_damping: Mapping[str, float] | Sequence[float] | float | None = None,
        delay_min_lag: int | None = None,
        delay_max_lag: int | None = None,
        **kwargs: Any,
    ) -> None:
        if joint_names_expr is not None and target_names_expr is not None:
            normalized_joint_names = (
                (joint_names_expr,)
                if isinstance(joint_names_expr, str)
                else tuple(joint_names_expr)
            )
            normalized_target_names = (
                (target_names_expr,)
                if isinstance(target_names_expr, str)
                else tuple(target_names_expr)
            )
            if normalized_joint_names != normalized_target_names:
                raise ValueError(
                    "joint_names_expr and target_names_expr are aliases and must agree when both supplied"
                )
        if target_names_expr is None:
            target_names_expr = joint_names_expr
        if target_names_expr is None:
            raise TypeError("provide joint_names_expr or target_names_expr")
        if isinstance(target_names_expr, str):
            target_names_expr = (target_names_expr,)

        # PACE delays *computed torque*, while mjlab's base actuator delay
        # delays commands.  Keep the inherited command delay disabled and let
        # PaceDCMotor own a separate torque buffer.  The inherited values are
        # accepted as aliases so nested tyro reconstruction stays compatible.
        # Serialized mjlab configs always carry the inherited command-delay
        # value ``0``.  It is not a conflicting PACE alias because PACE owns a
        # separate torque buffer; only a non-zero legacy alias is meaningful.
        if delay_max_lag not in (None, 0) and max_delay not in (0, delay_max_lag):
            raise ValueError(
                "max_delay and delay_max_lag are aliases and must agree when both supplied"
            )
        if delay_max_lag not in (None, 0):
            max_delay = delay_max_lag
        if isinstance(max_delay, bool) or not isinstance(max_delay, int) or max_delay < 0:
            raise ValueError("max_delay must be a non-negative integer")
        if delay_min_lag not in (None, 0):
            raise ValueError("PACE torque delay has a fixed minimum lag of zero")

        # ``asdict`` is used by mjlab to persist training parameters.  Accept
        # its PACE-specific fields on a later reconstruction while keeping the
        # original rich scalar/list/regex values authoritative for this layer.
        saved_saturation_effort = kwargs.pop("pace_saturation_effort", saturation_effort)
        saved_effort_limit = kwargs.pop("pace_effort_limit", effort_limit)
        saved_velocity_limit = kwargs.pop("pace_velocity_limit", velocity_limit)
        saved_stiffness = kwargs.pop("pace_stiffness", stiffness)
        saved_damping = kwargs.pop("pace_damping", damping)
        saved_armature = kwargs.pop("pace_armature", armature)
        saved_frictionloss = kwargs.pop("pace_frictionloss", None)
        saved_viscous_damping = kwargs.pop("pace_viscous_damping", None)

        def scalar(
            value: Mapping[str, float] | Sequence[float] | float | None, default: float
        ) -> float:
            if value is None:
                return default
            if isinstance(value, (float, int)):
                return float(value)
            if isinstance(value, Mapping):
                return float(next(iter(value.values()), default))
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return float(value[0]) if value else default
            raise TypeError(f"Unsupported PACE joint parameter type: {type(value)!r}")

        initial_friction = _select_alias("friction", dynamic_friction, friction, frictionloss)
        initial_viscous_damping = _select_alias(
            "viscous friction", viscous_friction, viscous_damping
        )
        # Nested tyro reconstruction used by mjlab exposes inherited actuator
        # fields as well.  Consume them explicitly so serialization round trips
        # do not pass duplicate keyword arguments to the dataclass base class.
        kwargs.pop("delay_hold_prob", None)
        super().__init__(
            target_names_expr=tuple(target_names_expr),
            saturation_effort=scalar(saturation_effort, 0.0),
            effort_limit=scalar(effort_limit, float("inf")),
            velocity_limit=scalar(velocity_limit, 1.0),
            stiffness=scalar(stiffness, 0.0),
            damping=scalar(damping, 0.0),
            armature=scalar(armature, 0.0) if armature is not None else None,
            frictionloss=scalar(initial_friction, 0.0) if initial_friction is not None else None,
            viscous_damping=(
                scalar(initial_viscous_damping, 0.0)
                if initial_viscous_damping is not None
                else None
            ),
            delay_min_lag=0,
            delay_max_lag=0,
            delay_hold_prob=0.0,
            **kwargs,
        )
        self.encoder_bias = encoder_bias
        self.max_delay = max_delay
        self.class_type = PaceDCMotor
        self.pace_saturation_effort = saved_saturation_effort
        self.pace_effort_limit = saved_effort_limit
        self.pace_velocity_limit = saved_velocity_limit
        self.pace_stiffness = saved_stiffness
        self.pace_damping = saved_damping
        self.pace_armature = saved_armature
        self.pace_frictionloss = (
            initial_friction if saved_frictionloss is None else saved_frictionloss
        )
        self.pace_viscous_damping = (
            initial_viscous_damping if saved_viscous_damping is None else saved_viscous_damping
        )

    @property
    def joint_names_expr(self) -> tuple[str, ...]:
        """Isaac Lab alias retained for source-compatible PACE configs."""
        return self.target_names_expr

    @joint_names_expr.setter
    def joint_names_expr(self, value: Sequence[str] | str) -> None:
        self.target_names_expr = (value,) if isinstance(value, str) else tuple(value)

    def build(self, entity, target_ids: list[int], target_names: list[str]) -> PaceDCMotor:
        """Build the PACE extension around mjlab's native DC motor model."""
        return PaceDCMotor(self, entity, target_ids, target_names)
