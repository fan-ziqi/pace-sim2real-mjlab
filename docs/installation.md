# Installation

PACE on mjlab is supported on 64-bit Linux with Python 3.10–3.13.  CUDA is
strongly recommended for fitting; CPU is suitable only for smoke tests.  Plan
for an NVIDIA driver compatible with the installed PyTorch/CUDA runtime, at
least 16 GB system RAM, and roughly **8–10 GB** free disk space for the virtual
environment and kernel cache.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first,
then create the project-local environment:

```bash
git clone https://github.com/fan-ziqi/pace-sim2real-mjlab.git
cd pace-sim2real-mjlab
uv sync --group dev
```

The environment is stored at `.venv`.  Use `uv run <command>` or activate it:

```bash
source .venv/bin/activate
```

The project pins `mjlab==1.5.2` and uses its MuJoCo-Warp CUDA runtime.  Check
the environment with `uv run python scripts/list_envs.py`; it must list
`Isaac-Pace-Anymal-D-v0`.

For CPU-only debugging, append `--device cpu` to PACE scripts.  Production
fitting should use CUDA because the CMA-ES population is evaluated in parallel.
The first CUDA launch compiles MuJoCo-Warp kernels and may take several minutes.

The lock file is generated against the public PyPI registry.  If your network
requires an internal mirror, configure uv at your organization level and then
run `uv lock` so that its source is explicit in your own lock file.

Published documentation is available at
[fan-ziqi.github.io/pace-sim2real-mjlab](https://fan-ziqi.github.io/pace-sim2real-mjlab/).
