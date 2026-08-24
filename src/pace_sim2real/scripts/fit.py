"""Run the PACE CMA-ES identification loop in mjlab."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pace_sim2real import CMAESOptimizer
from pace_sim2real.utils import load_pace_artifact, project_root

from ._common import (
    estimate_cmaes_trajectory_memory,
    make_env,
    pace_joint_ids,
    pace_position_action,
    resolve_device,
    validate_pace_trajectory_data,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit PACE parameters with mjlab and CMA-ES.")
    parser.add_argument("--num_envs", type=int, default=4096, help="CMA-ES population size.")
    parser.add_argument("--task", type=str, default="Isaac-Pace-Anymal-D-v0")
    parser.add_argument(
        "--task-module",
        "--task_module",
        dest="task_module",
        default=None,
        help="Import this module before resolving --task (for a custom registered task).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data", type=str, default=None, help="Override chirp_data.pt path.")
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument(
        "--headless", action="store_true", help="Accepted for Isaac Lab CLI compatibility."
    )
    return parser


def run(args: argparse.Namespace) -> torch.Tensor:
    if args.num_envs < 4:
        raise ValueError("--num_envs must be at least 4 for CMA-ES")
    if args.max_iterations is not None and args.max_iterations <= 0:
        raise ValueError("--max_iterations must be positive when supplied")
    device = resolve_device(args.device)
    env = make_env(args.task, args.num_envs, device, task_module=args.task_module)
    optimizer: CMAESOptimizer | None = None
    try:
        robot = env.scene["robot"]
        sim2real = env.cfg.sim2real
        joint_ids = pace_joint_ids(robot, sim2real.joint_order, device)
        source = Path(args.data) if args.data else project_root() / "data" / sim2real.data_dir
        if not source.exists():
            raise FileNotFoundError(
                f"No excitation data at {source}. Run scripts/pace/data_collection.py first "
                "or pass --data."
            )
        data = load_pace_artifact(source, map_location=device)
        time, measured, target = validate_pace_trajectory_data(
            data, physics_dt=env.physics_dt, joint_count=len(joint_ids)
        )
        measured = measured.to(device=device)
        target = target.to(device=device)
        history_bytes = estimate_cmaes_trajectory_memory(
            population_size=env.num_envs, samples=len(measured), joint_count=len(joint_ids)
        )
        print(
            "[INFO] CMA-ES position-history allocation: "
            f"{history_bytes / 1024**3:.2f} GiB minimum; simulation state requires additional memory."
        )
        max_iterations = (
            sim2real.cmaes.max_iteration if args.max_iterations is None else args.max_iterations
        )
        optimizer = CMAESOptimizer(
            bounds=sim2real.bounds_params,
            population_size=env.num_envs,
            log_dir=project_root() / "logs" / "pace" / sim2real.robot_name,
            joint_order=sim2real.joint_order,
            max_iteration=max_iterations,
            data=data,
            device=device,
            epsilon=sim2real.cmaes.epsilon,
            sigma=sim2real.cmaes.sigma,
            save_interval=sim2real.cmaes.save_interval,
            save_optimization_process=sim2real.cmaes.save_optimization_process,
        )
        initial_encoder_position = measured[0].unsqueeze(0).repeat(env.num_envs, 1)
        env.reset()
        optimizer.update_simulator(env, robot, joint_ids, initial_encoder_position)

        while not optimizer.finished():
            for step in range(measured.shape[0]):
                # ``tell`` retains PACE's upstream residual q_sim - q_real - bias.
                # Therefore it must receive raw simulated joint position here;
                # ``joint_pos_biased`` has already applied q_encoder = q - bias.
                simulated_joint_pos = robot.data.joint_pos[:, joint_ids]
                optimizer.tell(
                    simulated_joint_pos,
                    measured[step].unsqueeze(0).repeat(env.num_envs, 1),
                )
                action = pace_position_action(
                    env, robot, joint_ids, target[step].unsqueeze(0).repeat(env.num_envs, 1)
                )
                env.step(action)
                if (step + 1) % max(1, int(1.0 / env.physics_dt)) == 0:
                    print(
                        f"[INFO] Step {(step + 1) * env.physics_dt:.1f} / "
                        f"{time[-1].item():.1f} seconds "
                        f"({(step + 1) / len(measured) * 100:.1f} %)"
                    )
            optimizer.evolve()
            if optimizer.finished():
                break
            env.reset()
            optimizer.update_simulator(env, robot, joint_ids, initial_encoder_position)
        best = optimizer.get_best_sim_params()
        print("[INFO] Best PACE parameters:", best.tolist())
        return best
    finally:
        if optimizer is not None:
            optimizer.close()
        env.close()


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
