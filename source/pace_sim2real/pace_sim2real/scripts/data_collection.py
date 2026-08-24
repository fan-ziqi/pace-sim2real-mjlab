"""Collect PACE chirp excitation data in mjlab."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ._common import (
    data_path,
    make_env,
    pace_joint_ids,
    pace_position_action,
    prepare_pace_model,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect PACE chirp data with mjlab.")
    parser.add_argument("--num_envs", type=int, default=1, help="Parallel environments.")
    parser.add_argument("--task", type=str, default="Isaac-Pace-Anymal-D-v0")
    parser.add_argument(
        "--task-module",
        "--task_module",
        dest="task_module",
        default=None,
        help="Import this module before resolving --task (for a custom registered task).",
    )
    parser.add_argument(
        "--min_frequency", type=float, default=0.1, help="Chirp start frequency [Hz]."
    )
    parser.add_argument(
        "--max_frequency", type=float, default=10.0, help="Chirp end frequency [Hz]."
    )
    parser.add_argument("--duration", type=float, default=20.0, help="Trajectory duration [s].")
    parser.add_argument(
        "--device", type=str, default=None, help="mjlab device, e.g. cuda:0 or cpu."
    )
    parser.add_argument("--output", type=str, default=None, help="Optional output .pt path.")
    parser.add_argument(
        "--plot", action="store_true", help="Display captured trajectories after collection."
    )
    parser.add_argument(
        "--headless", action="store_true", help="Accepted for Isaac Lab CLI compatibility."
    )
    return parser


def chirp_trajectory(
    *,
    duration: float,
    dt: float,
    joint_count: int,
    min_frequency: float,
    max_frequency: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the original ANYmal-D PACE chirp excitation trajectory."""
    if duration <= 0.0 or min_frequency <= 0.0 or max_frequency <= 0.0:
        raise ValueError("duration and chirp frequencies must be positive")
    # PACE has one simulator step per saved sample.  ``linspace`` would make
    # its spacing duration/(N-1), which differs from the fixed simulator dt.
    num_steps = max(1, int(round(duration / dt)))
    time = torch.arange(num_steps, dtype=torch.float32, device=device) * dt
    phase = (
        2
        * torch.pi
        * (
            min_frequency * time
            + ((max_frequency - min_frequency) / (2 * duration)) * torch.square(time)
        )
    )
    trajectory = torch.sin(phase).unsqueeze(-1).repeat(1, joint_count)
    if joint_count == 12:
        directions = torch.tensor(
            [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            device=device,
        )
        bias = torch.tensor([0.0, 0.4, 0.8] * 4, device=device)
        scale = torch.tensor([0.25, 0.5, -2.0] * 4, device=device)
        trajectory = (trajectory + bias) * directions * scale
    return time, trajectory


def run(args: argparse.Namespace) -> torch.Tensor:
    device = resolve_device(args.device)
    env = make_env(args.task, args.num_envs, device, task_module=args.task_module)
    try:
        robot = env.scene["robot"]
        joint_order = env.cfg.sim2real.joint_order
        joint_ids = pace_joint_ids(robot, joint_order, device)
        time, trajectory = chirp_trajectory(
            duration=args.duration,
            dt=env.physics_dt,
            joint_count=len(joint_ids),
            min_frequency=args.min_frequency,
            max_frequency=args.max_frequency,
            device=device,
        )
        env.reset()

        known_bias = torch.full((env.num_envs, len(joint_ids)), 0.05, device=device)
        prepare_pace_model(
            env,
            robot,
            joint_ids,
            armature=torch.full_like(known_bias, 0.1),
            damping=torch.full_like(known_bias, 4.5),
            friction=torch.full_like(known_bias, 0.05),
            bias=known_bias,
            delay=torch.full((env.num_envs, 1), 5, device=device),
            initial_encoder_position=trajectory[0].unsqueeze(0).repeat(env.num_envs, 1),
        )

        positions = torch.zeros((len(time), len(joint_ids)), device=device)
        targets = torch.zeros_like(positions)
        for step in range(len(time)):
            positions[step] = robot.data.joint_pos_biased[0, joint_ids]
            pace_target = trajectory[step].unsqueeze(0).repeat(env.num_envs, 1)
            action = pace_position_action(env, robot, joint_ids, pace_target)
            env.step(action)
            targets[step] = pace_target[0]
            if (step + 1) % max(1, int(1.0 / env.physics_dt)) == 0:
                print(f"[INFO] Step {(step + 1) * env.physics_dt:.1f} seconds")

        path = data_path(env.cfg.sim2real.data_dir) if args.output is None else Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"time": time.cpu(), "dof_pos": positions.cpu(), "des_dof_pos": targets.cpu()},
            path,
        )
        print(f"[INFO] Saved excitation data to {path}")

        if args.plot:
            import matplotlib.pyplot as plt

            for index, name in enumerate(joint_order):
                plt.figure()
                plt.plot(time.cpu(), positions[:, index].cpu(), label=f"{name} pos")
                plt.plot(time.cpu(), targets[:, index].cpu(), "--", label=f"{name} target")
                plt.legend()
                plt.grid()
            plt.show()
        return positions
    finally:
        env.close()


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
