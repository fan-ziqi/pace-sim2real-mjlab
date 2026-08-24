"""Safe, schema-oriented loading helpers for PACE tensor artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch


def load_pace_artifact(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
    """Load a PACE tensor artifact without executing pickle payloads.

    PACE writes only tensors and standard Python containers.  ``weights_only``
    accepts that format while rejecting arbitrary executable pickle globals.
    """
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (OSError, pickle.UnpicklingError, RuntimeError, ValueError, TypeError) as error:
        raise ValueError(
            f"Cannot safely load PACE artifact {path}. Expected a tensor-only PACE .pt file."
        ) from error


def require_tensor(value: Any, *, name: str) -> torch.Tensor:
    """Return a tensor payload or report the corrupted artifact field clearly."""
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"PACE artifact field {name!r} must be a torch.Tensor")
    return value
