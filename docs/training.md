# Training and Playback

PACE system identification is the primary workflow; the RSL-RL entry points
remain for existing project templates.  The bundled task has a zero-weight
reward, so it is an environment/API compatibility check rather than a useful
locomotion-training benchmark until you add a task reward.

Training is local by default and uses TensorBoard (no W&B login or upload):

```bash
uv run python scripts/rsl_rl/train.py \
  --task Isaac-Pace-Anymal-D-v0 --num_envs 64 --max_iterations 1 \
  --logger tensorboard
```

Historical flags including `--num_envs`, `--max_iterations`, `--seed`,
`--logger`, `--checkpoint`, `--device`, `--headless`, and boolean `--video`
are translated by the wrappers.  The native mjlab form is shown by `--help`.
Cloud logging is opt-in with `--logger wandb`; add `wandb/` to local ignores if
you enable it.

`play` automatically chooses the newest local `model_*.pt` under
`logs/rsl_rl` when no checkpoint is supplied.  You can be explicit instead:

```bash
uv run python scripts/rsl_rl/play.py \
  --task Isaac-Pace-Anymal-D-v0 --checkpoint logs/rsl_rl/pace_sim2real/<run>/model_1.pt
```

For a no-checkpoint environment smoke test use `--agent zero` or `--agent random`.
