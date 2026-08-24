# Advanced optimization

`CMAESOptimizer` keeps the upstream normalized `[-1, 1]` search space and
parameter layout.  The objective expects raw simulated joint positions and
encoder-frame measured positions; it applies the candidate encoder bias exactly
once internally.

The historical call remains valid after explicitly binding a manually-created
environment (the packaged scripts already do this):

```python
from pace_sim2real.utils import bind_environment

bind_environment(env)
optimizer.update_simulator(robot, joint_ids, initial_encoder_position)
```

For new code, the explicit form is clearer and needs no binding registry:

```python
optimizer.update_simulator(env, robot, joint_ids, initial_encoder_position)
```

Subclass `CMAESOptimizer.tell()` for a weighted or velocity-aware loss, and add
a known-parameter recovery test before trusting a new objective.

`tell()` requires one finite floating-point `[population_size, joint_count]`
sample for both simulation and measurement. Invalid samples are rejected before
they alter CMA-ES state. Bounds must be finite and ordered, and the final delay
bound must be non-negative.
