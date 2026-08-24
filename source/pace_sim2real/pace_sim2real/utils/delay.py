"""Discrete-delay conversions shared by PACE simulation paths."""

from __future__ import annotations

import torch


def pace_delay_steps(delay: torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
    """Convert CMA-ES's continuous delay coordinate using upstream PACE semantics.

    The original Isaac Lab implementation cast the non-negative candidate to
    ``int``.  That truncates rather than rounds (e.g. 5.8 means five physics
    steps), so retain the exact objective surface while mjlab's delay buffer
    requires an integer lag.
    """
    return delay.to(device=device, dtype=torch.long)
