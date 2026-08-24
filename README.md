# ⚙️ PACE — Sim-to-Real Transfer for Legged Robots (mjlab)

> ⚠️ **Experimental project: this project has not been fully tested.** If you
> encounter a problem, please [open an issue](https://github.com/fan-ziqi/pace-sim2real-mjlab/issues/new)
> or [submit a pull request](https://github.com/fan-ziqi/pace-sim2real-mjlab/pulls).

This repository ports [PACE Sim2Real](https://github.com/leggedrobotics/pace-sim2real)
from Isaac Lab to **mjlab / MuJoCo-Warp**.  It preserves PACE's package name,
ANYmal-D task ID, script locations, data format, CMA-ES log format, and public
Python imports so that existing users can switch simulators without learning a
new identification workflow.

PACE (Precise Adaptation through Continuous Evolution) identifies actuator and
joint dynamics from excitation data.  The standard ANYmal-D task fits, per
joint, armature, passive viscous damping, Coulomb friction, encoder bias, and a
fixed command delay.

## What is compatible

| Original PACE interface | mjlab port |
| --- | --- |
| `pace_sim2real` imports | Kept, including `PaceCfg`, `PaceSim2realEnvCfg`, `CMAESOptimizer`, `PaceDCMotorCfg` |
| `Isaac-Pace-Anymal-D-v0` | Kept as the mjlab registry task ID |
| `scripts/pace/data_collection.py` | Kept; writes the same `chirp_data.pt` keys |
| `scripts/pace/fit.py` | Kept; writes the same timestamped PACE log layout |
| `scripts/pace/plot_trajectory.py` | Kept; reads original-style PACE logs |
| `scripts/rsl_rl/{train,play}.py` | Kept as mjlab runner compatibility wrappers |

The task uses the official simplified ANYmal-D URDF from ANYbotics.  Visual
Collada meshes are intentionally stripped at load time: PACE fitting uses the
official collision and inertial model, while remaining self-contained and
independent of ROS mesh resolution.  See the bundled
[`source/pace_sim2real/pace_sim2real/assets/anymal_d/LICENSE`](source/pace_sim2real/pace_sim2real/assets/anymal_d/LICENSE)
for its BSD-3-Clause terms.

## Installation

This checkout is already an independent `uv` project.  It supports 64-bit Linux
and Python 3.10–3.13; CUDA is recommended for fitting and CPU is for small
debugging runs.  Install [uv](https://docs.astral.sh/uv/getting-started/installation/),
ensure an NVIDIA driver compatible with PyTorch when using CUDA, and reserve
roughly 8–10 GB of disk for the environment and kernel cache.  On a clean machine:

```bash
git clone https://github.com/fan-ziqi/pace-sim2real-mjlab.git
cd pace-sim2real-mjlab
uv sync --group dev
```

`uv` creates `.venv`; invoke commands through `uv run` (or activate it with
`source .venv/bin/activate`).  The first CUDA run compiles MuJoCo-Warp kernels;
later launches reuse the cache.  mjlab supports CPU execution too, although PACE
fitting is intended to run thousands of candidates on CUDA.

## Running the ANYmal-D example

Collect a simulated chirp trajectory:

```bash
uv run python scripts/pace/data_collection.py \
  --task Isaac-Pace-Anymal-D-v0 --num_envs 1 --device cuda:0
```

The data is stored in `data/anymal_d_sim/chirp_data.pt`.  It contains the same
three tensors as upstream PACE: `time`, `dof_pos`, and `des_dof_pos`.

Fit PACE parameters using one mjlab world per CMA-ES candidate:

```bash
uv run python scripts/pace/fit.py \
  --task Isaac-Pace-Anymal-D-v0 --num_envs 256 --device cuda:0
```

For a short sanity check, use a smaller valid population and iteration count:

```bash
uv run python scripts/pace/data_collection.py --duration 0.1 --output /tmp/chirp.pt
uv run python scripts/pace/fit.py --num_envs 16 --max_iterations 1 --data /tmp/chirp.pt
```

Then inspect the newest result:

```bash
uv run python scripts/pace/plot_trajectory.py --plot_trajectory
```

Score plots additionally require `sim2real.cmaes.save_optimization_process = True`
before fitting; otherwise PACE intentionally does not write `progress.pt`.

The default `--num_envs 4096` remains available for compatibility, but is a
large-GPU setting.  Start at 64–256 candidates: fitting prints its minimum
trajectory-history allocation, while simulation state requires more memory.
Input timestamps are in seconds and must advance at the simulation rate
(0.0025 s / 400 Hz); mismatched real data is rejected before fitting rather
than silently changing the delay and damping estimate.

## Training and playback compatibility

The retained RSL-RL wrappers accept the historical PACE flags and default to
local TensorBoard logging, so they do not contact W&B unless you explicitly
request `--logger wandb`:

```bash
uv run python scripts/rsl_rl/train.py \
  --task Isaac-Pace-Anymal-D-v0 --num_envs 64 --max_iterations 1 --logger tensorboard
uv run python scripts/rsl_rl/play.py \
  --task Isaac-Pace-Anymal-D-v0 --agent zero
```

For a trained policy, `play` automatically finds the newest local checkpoint
or accepts the historical `--checkpoint PATH` spelling.  The included PACE task
has a zero-weight reward and serves as a compatibility/simulation template;
add a task reward before expecting useful policy learning.

## Public Python API

```python
from pace_sim2real import PaceCfg, PaceSim2realEnvCfg, CMAESOptimizer
from pace_sim2real.utils import PaceDCMotorCfg, PaceDCMotor
import pace_sim2real.tasks  # registers Isaac-Pace-Anymal-D-v0 in mjlab
```

Use the normal mjlab registry and environment construction APIs after importing
`pace_sim2real.tasks`; details and migration notes are in [`docs/`](docs/index.md).
For a manually constructed `ManagerBasedRlEnv`, call
`bind_environment(env)` before using the historical three-argument optimizer
method. The explicit `update_simulator(env, robot, ...)` form needs no binding.

## Safety and reproducibility

PACE accepts only tensor-only `.pt` artifacts and loads them with PyTorch's
safe `weights_only=True` mode. Treat excitation data as an explicit input:
validate its time base, keep the raw source data under version control where
appropriate, and set `PACE_ROOT` when running an installed wheel so logs and
data do not land in an accidental working directory. Every fit creates a unique
microsecond-resolution run directory, preventing concurrent runs from sharing
checkpoints or TensorBoard events.

## Verification

```bash
uv run ruff check source/pace_sim2real scripts tests
uv run ruff format --check source/pace_sim2real scripts tests
uv run pytest
uv run python scripts/list_envs.py
```

The physical integration test runs once on CPU and additionally runs on CUDA
when available. It creates the real ANYmal-D mjlab environment, sets distinct
per-world PACE parameters, steps it, and verifies encoder-bias, delay,
candidate-isolation, and legacy optimizer-binding semantics. Run it explicitly
on a machine with the mjlab runtime:

```bash
uv run pytest -q tests/test_mjlab_integration.py
```

The hosted GitHub workflow uses Python 3.10 and a CPU PyTorch wheel for its
static and public-API tests because GitHub-hosted runners have no NVIDIA
driver. A GPU-capable development machine remains required for the physical
integration test and a meaningful fitting run.

## Documentation

The rendered guide is published at
[fan-ziqi.github.io/pace-sim2real-mjlab](https://fan-ziqi.github.io/pace-sim2real-mjlab/).
Build it locally with `uv run --group docs mkdocs build --strict`.

## License

This mjlab port is © 2026 Ziqi Fan and licensed under Apache-2.0; see
[LICENSE](LICENSE). The original PACE Sim2Real © 2025 ETH Zurich, Robotic
Systems Lab, Filip Bjelonic remains Apache-2.0, with its attribution retained
in [LICENSES/UPSTREAM-PACE-NOTICE.txt](LICENSES/UPSTREAM-PACE-NOTICE.txt).
The embedded simplified ANYmal-D description is separately BSD-3-Clause and
credited above. See the [full attribution guide](docs/legal.md).
