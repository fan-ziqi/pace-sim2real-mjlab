"""Compatibility wrapper from PACE's historical CLI to mjlab train."""

from __future__ import annotations

import sys

from ._common import extract_task_module, import_task_module
from ._legacy_cli import translate_train_args


def main() -> None:
    from mjlab.scripts.train import main as mjlab_main

    task_module, args = extract_task_module(sys.argv[1:])
    import pace_sim2real.tasks  # noqa: F401

    import_task_module(task_module)

    sys.argv = [sys.argv[0], *translate_train_args(args)]
    mjlab_main()


if __name__ == "__main__":
    main()
