"""Project path helpers kept compatible with the Isaac Lab PACE package."""

from __future__ import annotations

import os
from pathlib import Path

PACE_ROOT_ENV = "PACE_ROOT"


def project_root() -> Path:
    """Return the top-level PACE project directory.

    ``PACE_ROOT`` has precedence, then a ``.project-root`` or ``.git`` marker
    is sought from this module upwards.  This preserves the original PACE
    override mechanism while also working from an editable uv installation.
    """
    if PACE_ROOT_ENV in os.environ:
        return Path(os.environ[PACE_ROOT_ENV]).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".project-root").exists() or (parent / ".git").exists():
            return parent
    # Wheels intentionally contain no repository marker.  For an installed
    # console script, the current directory is the least surprising writable
    # project/data location; callers can always make it explicit with
    # ``PACE_ROOT``.
    return Path.cwd().resolve()
