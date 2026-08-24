# Optimization API

## `CMAESOptimizer`

The optimizer samples a candidate per mjlab world, applies candidate physical
parameters, accumulates tracking error with `tell`, and advances a generation
with `evolve`.  See [Advanced optimization](../tutorials/advanced.md) for both
the historical and explicit simulator-update signatures.

The constructor accepts finite ordered bounds in PACE's standard layout and a
positive iteration count. `tell()` accepts only finite floating-point tensors
with shape `[population_size, len(joint_order)]`; rejected input leaves the
optimizer unchanged. `update_simulator()` validates all model tensors and
actuator delay capacity before it changes an mjlab model, so a failed update
does not leave partial candidate parameters behind.

Both the collection helper and CMA-ES call the same validated model adapter.
This guarantees the same armature, damping, friction, encoder-bias, torque
delay, reset, and `forward()` behavior for direct simulation and fitting.
