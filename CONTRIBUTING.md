# Contributing to PACE Sim2Real for mjlab

Contributions are welcome: bug reports, documentation improvements, custom
robot examples, and core PACE improvements. For a substantial API or
robot-integration change, open an issue first so the design can be discussed.

## Development setup

```bash
git clone https://github.com/<your-account>/pace-sim2real-mjlab.git
cd pace-sim2real-mjlab
uv sync --group dev --group docs
```

Before opening a pull request, run:

```bash
uv run ruff format --check src scripts tests
uv run ruff check src scripts tests
uv run pytest
uv run --group docs mkdocs build --strict
```

The physical integration test is included in `pytest` and needs a functioning
mjlab runtime. It exercises CPU execution and CUDA execution when CUDA is
available.

## Pull requests

Create a feature branch, keep changes focused, and open a pull request against
`main`. Include a short summary and the verification commands you ran. Custom
robot examples should include data collection, parameter identification,
visualization, documentation, and no large generated data files.

## Bug reports

Please include the project commit hash, mjlab version, OS, Python version,
CUDA/GPU details when relevant, the exact command, and complete logs or error
messages. Feature requests should describe the target robot and workflow.

## License

By contributing, you agree that your contribution is licensed under Apache-2.0,
the same license used by this port and the retained upstream PACE code.
