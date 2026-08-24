"""Reward helpers retained from the original PACE task."""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg


def joint_pos_target_l2(env, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the squared wrapped joint-position deviation from ``target``."""
    asset = env.scene[asset_cfg.name]
    joint_pos = torch.atan2(
        torch.sin(asset.data.joint_pos[:, asset_cfg.joint_ids]),
        torch.cos(asset.data.joint_pos[:, asset_cfg.joint_ids]),
    )
    return torch.sum(torch.square(joint_pos - target), dim=1)
