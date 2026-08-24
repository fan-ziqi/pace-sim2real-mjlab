"""Visualise PACE trajectory fitting logs (format-compatible with the original)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from pace_sim2real.utils import load_pace_artifact, project_root, require_tensor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot PACE optimization outputs.")
    parser.add_argument("--folder_name", type=str, default=None)
    parser.add_argument("--mean_name", type=str, default=None)
    parser.add_argument("--robot_name", type=str, default="anymal_d_sim")
    parser.add_argument("--plot_trajectory", action="store_true")
    parser.add_argument("--plot_score", action="store_true")
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Save figures instead of opening windows."
    )
    return parser


_MEAN_PATTERN = re.compile(r"^mean_(\d+)\.pt$")


def _latest_mean(folder: Path) -> tuple[Path, int]:
    candidates = []
    for item in folder.glob("mean_*.pt"):
        match = _MEAN_PATTERN.match(item.name)
        if match:
            candidates.append((int(match.group(1)), item))
    if not candidates:
        raise FileNotFoundError(f"No mean_*.pt files found under {folder}")
    iteration, path = max(candidates)
    return path, iteration


def run(args: argparse.Namespace) -> dict[str, torch.Tensor | list[str]]:
    root = project_root() / "logs" / "pace" / args.robot_name
    if not root.exists():
        raise FileNotFoundError(f"No logs for robot {args.robot_name} under {root}")
    if args.folder_name:
        run_dir = root / args.folder_name
    else:
        run_dirs = [item for item in root.iterdir() if item.is_dir()]
        if not run_dirs:
            raise FileNotFoundError(f"No PACE run directories found under {root}")
        run_dir = max(run_dirs, key=lambda item: item.stat().st_mtime)
    if args.mean_name:
        mean_path = run_dir / args.mean_name
        match = _MEAN_PATTERN.match(args.mean_name)
        if match is None or not mean_path.exists():
            raise FileNotFoundError(f"Invalid mean file: {mean_path}")
        iteration = int(match.group(1))
    else:
        mean_path, iteration = _latest_mean(run_dir)

    mean = require_tensor(load_pace_artifact(mean_path), name="mean")
    config = load_pace_artifact(run_dir / "config.pt")
    if not isinstance(config, dict):
        raise ValueError("PACE config.pt must contain a dictionary")
    required_config = {"joint_order", "dof_pos", "des_dof_pos", "time"}
    missing_config = required_config.difference(config)
    if missing_config:
        raise ValueError(f"PACE config.pt is missing keys: {sorted(missing_config)}")
    joint_order = config["joint_order"]
    if not isinstance(joint_order, list) or not all(isinstance(name, str) for name in joint_order):
        raise ValueError("PACE config.pt joint_order must be a list of strings")
    print(f"Latest params file: {mean_path}")
    print("Best parameter set:", mean)
    print("Armature params:", mean[: len(joint_order)])
    print("Viscous friction params:", mean[len(joint_order) : 2 * len(joint_order)])
    print("Static/dynamic friction params:", mean[2 * len(joint_order) : 3 * len(joint_order)])
    print("Encoder bias params:", mean[3 * len(joint_order) : 4 * len(joint_order)])
    print("Delay param:", mean[-1].item())

    if args.plot_trajectory or args.plot_score:
        import matplotlib.pyplot as plt

        save_dir = Path(args.save_dir) if args.save_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        if args.plot_score:
            progress_path = run_dir / "progress.pt"
            if progress_path.exists():
                progress = load_pace_artifact(progress_path)
                if not isinstance(progress, dict) or not isinstance(
                    progress.get("scores_buffer"), torch.Tensor
                ):
                    raise ValueError("PACE progress.pt must contain a tensor scores_buffer")
                values = torch.min(progress["scores_buffer"][: iteration + 1], dim=1).values
                plt.figure()
                plt.semilogy(values.numpy())
                plt.title("CMA-ES Score over Iterations")
                plt.xlabel("Iteration")
                plt.ylabel("Score")
                plt.grid()
                if save_dir:
                    plt.savefig(save_dir / "score.png", dpi=160)
            else:
                print("No progress.pt found; skipping score plot.")
        if args.plot_trajectory:
            simulated = require_tensor(
                load_pace_artifact(run_dir / "best_trajectory.pt"), name="best_trajectory"
            )
            real = require_tensor(config["dof_pos"], name="dof_pos")
            desired = require_tensor(config["des_dof_pos"], name="des_dof_pos")
            time = require_tensor(config["time"], name="time")
            encoder_bias = mean[3 * len(joint_order) : 4 * len(joint_order)]
            for index, name in enumerate(joint_order):
                plt.figure(figsize=(8, 4.5))
                plt.plot(time, simulated[:, index] - encoder_bias[index], label="Sim", linewidth=2)
                plt.plot(time, real[:, index], "--", label="Real", linewidth=2)
                plt.plot(time, desired[:, index], "--", color="grey", alpha=0.5, label="Target")
                plt.title(f"Joint {name}")
                plt.xlabel("Time [s]")
                plt.ylabel("Joint position [rad]")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                if save_dir:
                    plt.savefig(save_dir / f"trajectory_{name}.png", dpi=160)
        if save_dir is None:
            plt.show()
        else:
            plt.close("all")
    return {"mean": mean, "joint_order": joint_order}


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
