"""Finite random/zero PACE rollouts for quick environment checks."""

from __future__ import annotations

import argparse

import torch

from ._common import make_env, resolve_device


def run(agent: str, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=f"Run a {agent} PACE agent in mjlab.")
    parser.add_argument("--task", default="Isaac-Pace-Anymal-D-v0")
    parser.add_argument("--task-module", "--task_module", dest="task_module", default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--headless", action="store_true", help="Accepted for Isaac Lab CLI compatibility."
    )
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        help="Accepted as an Isaac Lab no-op; mjlab has no Fabric backend.",
    )
    args = parser.parse_args(argv)
    env = make_env(
        args.task,
        args.num_envs,
        resolve_device(args.device),
        play=True,
        task_module=args.task_module,
    )
    try:
        observations, _ = env.reset()
        print("[INFO] Observation keys:", list(observations))
        print("[INFO] Action space:", env.action_space)
        for _ in range(args.steps):
            if agent == "random":
                action = 2 * torch.rand(env.action_space.shape, device=env.device) - 1
            else:
                action = torch.zeros(env.action_space.shape, device=env.device)
            env.step(action)
    finally:
        env.close()


def random_main() -> None:
    run("random")


def zero_main() -> None:
    run("zero")
