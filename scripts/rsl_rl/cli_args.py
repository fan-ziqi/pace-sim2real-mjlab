"""Legacy import shim for scripts that used PACE's Isaac Lab CLI helpers.

mjlab uses typed, nested CLI flags.  The maintained compatibility wrappers
translate the common ``--task``, ``--num_envs``, and ``--max_iterations`` flags;
invoke ``scripts/rsl_rl/train.py --task ... --help`` for the complete mjlab CLI.
"""

from __future__ import annotations


def add_rsl_rl_args(parser):
    """Leave the parser untouched; mjlab owns runner option parsing."""
    return parser


def update_rsl_rl_cfg(agent_cfg, _args_cli):
    """Return the mjlab runner configuration unchanged."""
    return agent_cfg
