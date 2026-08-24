# Basic Usage

PACE keeps the original three-stage workflow.

1. Collect or supply `data/<robot>/chirp_data.pt` containing `time`, `dof_pos`,
   and `des_dof_pos` tensors of shape `[samples, joints]`.
2. Run `scripts/pace/fit.py`; its `--num_envs` is the CMA-ES population size
   and must be at least 4.
3. Use `scripts/pace/plot_trajectory.py` to inspect the latest PACE log.

Score plotting requires `sim2real.cmaes.save_optimization_process = True` at
fit time; it is disabled by default to avoid writing the full optimization
history.

```bash
uv run python scripts/pace/data_collection.py --duration 20 --device cuda:0
uv run python scripts/pace/fit.py --num_envs 256 --device cuda:0
uv run python scripts/pace/plot_trajectory.py --plot_trajectory
```

`--headless` is accepted by collection and fitting scripts for command-line
compatibility with Isaac Lab.  mjlab is headless by default for these workflows.

## Real-data contract

`time` is measured in **seconds**, starts at the first recorded encoder sample,
and must be strictly increasing at exactly `0.0025 s` (400 Hz) for the standard
ANYmal task.  `dof_pos` is encoder-frame joint position in the canonical order
printed by the task; `des_dof_pos` is the synchronized command sent to the
controller in that same encoder frame.  PACE validates these constraints before
fitting and refuses a mismatched rate rather than silently identifying the wrong
damping or delay.  Resample and synchronize real data before export.

Only tensor-only PyTorch artifacts are accepted (`weights_only=True` loading).
The CLI rejects arbitrary pickle globals, malformed fields, non-finite values,
and non-matching shapes before it starts a fit.

The collector writes this same 400 Hz grid.  A minimal validation run is:

```bash
uv run python scripts/pace/data_collection.py --duration 0.1 --output /tmp/chirp.pt
uv run python scripts/pace/fit.py --num_envs 4 --max_iterations 1 --data /tmp/chirp.pt
```

## Population size and memory

One world is allocated per CMA-ES candidate.  The optimizer prints the minimum
trajectory-history allocation at startup; simulation state needs additional GPU
memory.  Start at 64–256 candidates, increase only after confirming headroom,
and reserve 2048–4096 for large GPUs and final experiments.

## Parameter semantics

The fitted 49-element ANYmal vector preserves the upstream order:

```text
12 armatures | 12 viscous dampings | 12 Coulomb frictions |
12 encoder biases | 1 command delay
```

The torque delay is truncated toward zero to an integer number of 2.5 ms
simulation steps and held constant for each CMA-ES candidate over its rollout.
