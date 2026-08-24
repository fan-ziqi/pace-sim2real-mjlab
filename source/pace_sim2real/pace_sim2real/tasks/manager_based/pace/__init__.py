"""PACE environment registration for mjlab."""

from mjlab.tasks.registry import register_mjlab_task

from .agents.rsl_rl_ppo_cfg import ppo_runner_cfg
from .anymal_pace_env_cfg import TASK_ID, anymal_d_pace_env_cfg

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=anymal_d_pace_env_cfg(),
    play_env_cfg=anymal_d_pace_env_cfg(play=True),
    rl_cfg=ppo_runner_cfg(),
)

# The original extension also shipped this template identifier.  It remains a
# usable alias so old scripts and project templates do not fail at discovery.
register_mjlab_task(
    task_id="Template-Pace-Sim2real-v0",
    env_cfg=anymal_d_pace_env_cfg(),
    play_env_cfg=anymal_d_pace_env_cfg(play=True),
    rl_cfg=ppo_runner_cfg(),
)

__all__ = ["TASK_ID"]
