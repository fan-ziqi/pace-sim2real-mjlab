"""Translation layer for PACE's Isaac Lab RSL-RL command-line contract."""

from __future__ import annotations

from pathlib import Path


def _split_task(args: list[str]) -> tuple[str | None, list[str]]:
    task: str | None = None
    rest: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--task":
            if index + 1 >= len(args):
                raise SystemExit("--task requires a task identifier")
            task = args[index + 1]
            index += 2
        elif arg.startswith("--task="):
            task = arg.split("=", 1)[1]
            index += 1
        else:
            rest.append(arg)
            index += 1
    return task, rest


def _value(args: list[str], index: int, flag: str) -> tuple[str, int]:
    if index + 1 >= len(args):
        raise SystemExit(f"{flag} requires a value")
    return args[index + 1], index + 2


def _has_option(args: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(option + "=") for arg in args)


def _option_value(args: list[str], option: str, default: str) -> str:
    for index, arg in enumerate(args):
        if arg == option and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(option + "="):
            return arg.split("=", 1)[1]
    return default


def _translate(args: list[str], *, play: bool) -> tuple[str | None, list[str]]:
    """Translate documented PACE flags into mjlab's typed flag spellings."""
    task, legacy = _split_task(args)
    translated: list[str] = []
    index = 0
    value_map = {
        "--num_envs": "--num-envs" if play else "--env.scene.num-envs",
        "--num-envs": "--num-envs" if play else "--env.scene.num-envs",
        "--video_length": "--video-length",
        "--video-length": "--video-length",
        "--checkpoint": "--checkpoint-file" if play else "--agent.load-checkpoint",
        "--logger": "--agent.logger",
        "--seed": "--env.seed" if play else "--agent.seed",
        "--max_iterations": "--agent.max-iterations",
        "--max-iterations": "--agent.max-iterations",
        "--experiment_name": "--agent.experiment-name",
        "--run_name": "--agent.run-name",
        "--load_run": "--agent.load-run",
        "--log_project_name": "--agent.wandb-project",
    }
    if not play:
        value_map["--video_interval"] = "--video-interval"
        value_map["--video-interval"] = "--video-interval"

    while index < len(legacy):
        arg = legacy[index]
        key, equals, inline_value = arg.partition("=")
        if play and key in {
            "--seed",
            "--logger",
            "--experiment_name",
            "--run_name",
            "--load_run",
            "--log_project_name",
            "--max_iterations",
            "--max-iterations",
        }:
            # mjlab play receives a fully materialized policy; these training
            # configuration flags have no effect but remain accepted.
            if not equals:
                _, index = _value(legacy, index, key)
            else:
                if not inline_value:
                    raise SystemExit(f"{key} requires a value")
                index += 1
            continue
        if key in value_map:
            if equals:
                translated.extend((value_map[key], inline_value))
                index += 1
            else:
                value, index = _value(legacy, index, key)
                translated.extend((value_map[key], value))
            continue
        if key == "--video":
            if equals:
                translated.extend(("--video", inline_value.title()))
                index += 1
            elif index + 1 < len(legacy) and legacy[index + 1].lower() in {"true", "false"}:
                translated.extend(("--video", legacy[index + 1].title()))
                index += 2
            else:
                translated.extend(("--video", "True"))
                index += 1
            continue
        if key == "--resume" and not play:
            translated.extend(("--agent.resume", inline_value if equals else "True"))
            index += 1
            continue
        if key == "--device":
            value = inline_value if equals else _value(legacy, index, key)[0]
            index += 1 if equals else 2
            if play:
                translated.extend(("--device", value))
            elif value == "cpu":
                # Tyro parses the union as Python-style values: ``None`` for
                # CPU and ``[0]`` for a one-element GPU list.
                translated.extend(("--gpu-ids", "None"))
            elif value.startswith("cuda:") and value.removeprefix("cuda:").isdigit():
                translated.extend(("--gpu-ids", f"[{value.removeprefix('cuda:')}]"))
            else:
                raise SystemExit("legacy --device must be 'cpu' or 'cuda:<index>' for mjlab")
            continue
        if key == "--agent":
            value = inline_value if equals else _value(legacy, index, key)[0]
            index += 1 if equals else 2
            if value == "rsl_rl_cfg_entry_point":
                translated.extend(("--agent", "trained")) if play else None
            elif play and value in {"zero", "random", "trained"}:
                translated.extend(("--agent", value))
            elif not play:
                raise SystemExit(
                    "mjlab exposes one registered PPO config per task; legacy --agent must be "
                    "rsl_rl_cfg_entry_point. Use nested --agent.* overrides instead."
                )
            else:
                raise SystemExit("play --agent must be one of zero, random, or trained")
            continue
        if key == "--distributed" and not play:
            translated.extend(("--gpu-ids", "all"))
            index += 1
            continue
        if key in {"--headless", "--enable_cameras", "--disable_fabric", "--real-time"}:
            # mjlab does not launch Isaac Sim/Fabric; the flags are harmless no-ops.
            index += 1
            continue
        if key == "--export_io_descriptors" and not play:
            index += 1
            continue
        if key == "--use_pretrained_checkpoint" and play:
            raise SystemExit(
                "No published mjlab PACE checkpoint exists. Pass --checkpoint <local-model.pt> "
                "or --wandb-run-path explicitly."
            )
        translated.append(arg)
        index += 1
    return task, translated


def translate_train_args(args: list[str]) -> list[str]:
    task, translated = _translate(args, play=False)
    if task is None:
        return args
    return [task, *translated]


def translate_play_args(args: list[str]) -> list[str]:
    task, translated = _translate(args, play=True)
    if task is None:
        return args
    if "-h" in translated or "--help" in translated:
        return [task, *translated]
    if not _has_option(translated, "--agent"):
        agent = "trained"
    else:
        agent = _option_value(translated, "--agent", "trained")
    if (
        agent == "trained"
        and not _has_option(translated, "--checkpoint-file")
        and not _has_option(translated, "--wandb-run-path")
    ):
        log_root = Path(_option_value(translated, "--log-root", "logs/rsl_rl"))
        candidates = list(log_root.glob("**/model_*.pt"))
        if not candidates:
            raise SystemExit(
                "No local PACE checkpoint found. Train first, pass --checkpoint <model.pt>, "
                "or use --agent zero/--agent random."
            )
        latest = max(candidates, key=lambda path: path.stat().st_mtime)
        translated.extend(("--checkpoint-file", str(latest)))
        print(f"[INFO] Using latest local checkpoint: {latest}")
    return [task, *translated]
