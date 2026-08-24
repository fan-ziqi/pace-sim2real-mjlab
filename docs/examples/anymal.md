# ANYmal-D PACE workflow

The `Isaac-Pace-Anymal-D-v0` task fixes the simplified ANYmal-D model at
`z=1 m`, uses 400 Hz physics/control, and identifies 12 armatures, 12 passive
dampings, 12 Coulomb frictions, 12 encoder biases, and one torque delay.

```bash
uv run python scripts/pace/data_collection.py --task Isaac-Pace-Anymal-D-v0 --duration 20
uv run python scripts/pace/fit.py --task Isaac-Pace-Anymal-D-v0 --num_envs 256
uv run python scripts/pace/plot_trajectory.py --plot_trajectory
```

Use `--plot_score` only after setting
`sim2real.cmaes.save_optimization_process = True`; the default omits
`progress.pt` to save disk space.

Use a short collection and four candidates first; the full population should be
sized to available GPU memory as described in [Usage](../usage.md).
