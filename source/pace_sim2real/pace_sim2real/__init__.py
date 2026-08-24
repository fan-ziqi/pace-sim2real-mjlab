"""PACE: Precise Adaptation through Continuous Evolution, powered by mjlab.

The public imports deliberately match the Isaac Lab edition so existing PACE
projects can move their system-identification workflow with minimal changes.
"""

from .optim import CMAESOptimizer
from .tasks.manager_based.pace.pace_sim2real_env_cfg import (
    PaceCfg,
    PaceSim2realEnvCfg,
    PaceSim2realSceneCfg,
)

__all__ = [
    "CMAESOptimizer",
    "PaceCfg",
    "PaceSim2realEnvCfg",
    "PaceSim2realSceneCfg",
]
