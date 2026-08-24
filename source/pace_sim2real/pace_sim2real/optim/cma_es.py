"""CMA-ES optimizer used by the PACE system-identification loop."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cmaes
import torch
from torch.utils.tensorboard import SummaryWriter

from pace_sim2real.utils.model import apply_pace_parameters


def _create_run_dir(log_dir: str | os.PathLike[str]) -> Path:
    """Create a unique CMA-ES output directory without cross-run mixing."""
    base_dir = Path(log_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    for sequence in range(1000):
        timestamp = datetime.now().strftime("%y_%m_%d_%H-%M-%S-%f")
        suffix = "" if sequence == 0 else f"-{sequence:03}"
        run_dir = base_dir / f"{timestamp}{suffix}"
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir
    raise RuntimeError(f"could not create a unique CMA-ES run directory under {base_dir}")


class CMAESOptimizer:
    """Fit PACE joint parameters using one mjlab world per CMA-ES candidate.

    The normalized CMA-ES search space is kept identical to the Isaac Lab
    implementation: every parameter is optimized in ``[-1, 1]`` and mapped to
    the physical limits in ``bounds``.  The physical parameter layout is
    ``[armature, viscous damping, Coulomb friction, encoder bias, delay]``.
    """

    def __init__(
        self,
        bounds: torch.Tensor,
        population_size: int,
        log_dir: str | os.PathLike[str],
        joint_order: list[str],
        max_iteration: int,
        data: dict[str, torch.Tensor],
        device: str,
        epsilon: float | None = None,
        sigma: float = 0.5,
        save_interval: int = 10,
        save_optimization_process: bool = False,
    ) -> None:
        if not isinstance(bounds, torch.Tensor) or bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (num_parameters, 2)")
        if not torch.isfinite(bounds).all() or torch.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("bounds must be finite and have lower <= upper for every parameter")
        if population_size < 4:
            raise ValueError("population_size must be at least 4 for stable CMA-ES updates")
        if (
            isinstance(max_iteration, bool)
            or not isinstance(max_iteration, int)
            or max_iteration <= 0
        ):
            raise ValueError("max_iteration must be a positive integer")
        if epsilon is not None and (not torch.isfinite(torch.tensor(epsilon)) or epsilon < 0):
            raise ValueError("epsilon must be non-negative and finite when supplied")
        if not torch.isfinite(torch.tensor(sigma)) or sigma <= 0:
            raise ValueError("sigma must be positive and finite")
        if save_interval < 0:
            raise ValueError("save_interval must be non-negative")
        expected_params = 4 * len(joint_order) + 1
        if bounds.shape[0] != expected_params:
            raise ValueError(
                f"expected {expected_params} bounds for {len(joint_order)} joints, "
                f"received {bounds.shape[0]}"
            )
        if not joint_order or len(set(joint_order)) != len(joint_order):
            raise ValueError("joint_order must contain unique joint names")
        required_data = {"time", "dof_pos", "des_dof_pos"}
        if not isinstance(data, dict) or required_data.difference(data):
            raise ValueError(f"data must be a dict containing {sorted(required_data)}")
        if not all(isinstance(data[name], torch.Tensor) for name in required_data):
            raise ValueError("PACE data fields time, dof_pos, and des_dof_pos must be tensors")
        if (
            data["dof_pos"].ndim != 2
            or data["des_dof_pos"].shape != data["dof_pos"].shape
            or data["dof_pos"].shape[1] != len(joint_order)
            or data["time"].numel() != data["dof_pos"].shape[0]
        ):
            raise ValueError("PACE data must contain matching [samples, joint_order] tensors")

        self.joint_order = list(joint_order)
        self.max_iteration = max_iteration
        self.epsilon = epsilon
        self.save_interval = save_interval
        self.device = device
        self.save_optimization_process = save_optimization_process
        self._timer_start = datetime.now()

        run_dir = _create_run_dir(log_dir)
        self.writer = SummaryWriter(log_dir=str(run_dir))
        torch.save(
            {
                "bounds": bounds.detach().cpu(),
                "joint_order": self.joint_order,
                "dof_pos": data["dof_pos"].detach().cpu(),
                "des_dof_pos": data["des_dof_pos"].detach().cpu(),
                "time": data["time"].detach().cpu(),
            },
            run_dir / "config.pt",
        )

        self.bounds = bounds.to(device=device, dtype=torch.float32)
        normalized_bounds = torch.ones_like(self.bounds)
        normalized_bounds[:, 0] = -1.0
        self.optimizer = cmaes.CMA(
            mean=torch.zeros(self.bounds.shape[0]).cpu().numpy(),
            sigma=sigma,
            bounds=normalized_bounds.cpu().numpy(),
            seed=0,
            population_size=population_size,
        )

        self.scores_counter = 0
        self.iteration_counter = 0
        self.scores = torch.zeros(population_size, device=device)
        self.scores_buffer = torch.zeros((max_iteration, population_size), device=device)
        self.sim_dof_pos_buffer = torch.zeros(
            (population_size, data["dof_pos"].shape[0], len(joint_order)), device=device
        )
        self.params = torch.zeros((population_size, self.bounds.shape[0]), device=device)
        self.sim_params = torch.zeros_like(self.params)
        if save_optimization_process:
            self.sim_params_buffer = torch.zeros(
                (max_iteration, population_size, self.bounds.shape[0]), device=device
            )

        num_joints = len(joint_order)
        self.armature_idx = slice(0, num_joints)
        self.damping_idx = slice(num_joints, 2 * num_joints)
        self.friction_idx = slice(2 * num_joints, 3 * num_joints)
        self.bias_idx = slice(3 * num_joints, 4 * num_joints)
        self.delay_idx = 4 * num_joints

        self._reset_population()
        print("CMA-ES optimizer initialized.")
        print("Current iteration:", self.iteration_counter)

    def ask(self) -> Any:
        """Return one normalized candidate from the underlying CMA-ES sampler."""
        return self.optimizer.ask()

    def tell(self, sim_dof_pos: torch.Tensor, real_dof_pos: torch.Tensor) -> None:
        """Accumulate a trajectory sample for every population member."""
        if self.scores_counter >= self.sim_dof_pos_buffer.shape[1]:
            raise IndexError("received more trajectory samples than provided measurement data")
        expected_shape = (self.optimizer.population_size, len(self.joint_order))
        if (
            tuple(sim_dof_pos.shape) != expected_shape
            or tuple(real_dof_pos.shape) != expected_shape
        ):
            raise ValueError(
                "simulated and measured positions must each have shape "
                f"{expected_shape}; got {tuple(sim_dof_pos.shape)} and {tuple(real_dof_pos.shape)}"
            )
        if sim_dof_pos.device != self.scores.device or real_dof_pos.device != self.scores.device:
            raise ValueError(f"trajectory samples must be on optimizer device {self.scores.device}")
        if not (sim_dof_pos.is_floating_point() and real_dof_pos.is_floating_point()):
            raise ValueError("trajectory samples must use a floating-point dtype")
        if not (torch.isfinite(sim_dof_pos).all() and torch.isfinite(real_dof_pos).all()):
            raise ValueError("trajectory samples must contain only finite values")
        residual = sim_dof_pos - real_dof_pos - self.sim_params[:, self.bias_idx]
        sample_scores = torch.sum(torch.square(residual), dim=1)
        # Complete every operation that can fail before committing either
        # score state or the sample counter, so callers can recover cleanly.
        self.sim_dof_pos_buffer[:, self.scores_counter, :].copy_(sim_dof_pos)
        self.scores.add_(sample_scores)
        self.scores_counter += 1

    def evolve(self) -> None:
        """Submit the completed population and sample the next generation."""
        if self.scores_counter == 0:
            raise RuntimeError("cannot evolve before at least one call to tell()")
        if self.iteration_counter >= self.max_iteration:
            raise RuntimeError("optimizer has already completed its configured iterations")

        self.scores /= self.scores_counter
        self.scores_buffer[self.iteration_counter] = self.scores
        if self.save_optimization_process:
            self.sim_params_buffer[self.iteration_counter] = self.sim_params

        solutions = [
            (self.params[i].detach().cpu().numpy(), self.scores[i].item())
            for i in range(self.optimizer.population_size)
        ]
        self.optimizer.tell(solutions)
        if self.save_interval > 0 and self.iteration_counter % self.save_interval == 0:
            mean = torch.tensor(self.optimizer._mean, device=self.device)
            self.save_checkpoint(self._params_to_sim_params(mean), self.iteration_counter)
        self._print_iteration()

        self._reset_population()
        self.scores.zero_()
        self.scores_counter = 0
        self.iteration_counter += 1
        print("CMA-ES optimizer iteration:", self.iteration_counter)

    def finished(self) -> bool:
        """Return whether the configured stopping criterion has been met."""
        if self.iteration_counter == 0:
            return False
        finished = self.iteration_counter >= self.max_iteration
        current_scores = self.scores_buffer[self.iteration_counter - 1]
        min_score = torch.min(current_scores)
        spread = (torch.max(current_scores) - min_score) / torch.clamp_min(
            min_score.abs(), torch.finfo(current_scores.dtype).eps
        )
        finished = finished or (self.epsilon is not None and spread < self.epsilon)
        if finished:
            print("CMA-ES optimization finished.")
            mean = torch.tensor(self.optimizer._mean, device=self.device)
            self.save_checkpoint(
                self._params_to_sim_params(mean), self.iteration_counter - 1, finished=True
            )
        return bool(finished)

    def update_simulator(
        self,
        env: Any,
        articulation: Any | None = None,
        joint_ids: torch.Tensor | None = None,
        initial_position: torch.Tensor | None = None,
    ) -> None:
        """Apply every candidate's PACE parameters to its mjlab world.

        ``env`` is an :class:`mjlab.envs.ManagerBasedRlEnv`; ``articulation``
        defaults to ``env.scene[\"robot\"]``.  The method expands MuJoCo model
        fields once so armature, damping and friction can vary independently for
        all CMA-ES candidates.
        """
        # Accept the upstream API as well as the explicit mjlab API:
        #   update_simulator(articulation, joint_ids, initial_position)
        #   update_simulator(env, articulation, joint_ids, initial_position)
        if not hasattr(env, "scene"):
            original_articulation = env
            original_joint_ids = articulation
            original_initial_position = joint_ids
            from pace_sim2real.utils import bound_environment

            env = bound_environment(original_articulation)
            if env is None:
                raise TypeError(
                    "the original update_simulator(articulation, ...) signature requires "
                    "a PACE-owned environment. Import pace_sim2real.tasks before constructing "
                    "ManagerBasedRlEnv, or pass env explicitly."
                )
            articulation = original_articulation
            joint_ids = original_joint_ids
            initial_position = original_initial_position
        if articulation is None:
            articulation = env.scene["robot"]
        if joint_ids is None:
            joint_ids = torch.tensor(
                [articulation.joint_names.index(name) for name in self.joint_order],
                device=self.device,
                dtype=torch.long,
            )
        joint_ids = joint_ids.to(device=self.device, dtype=torch.long)
        if len(joint_ids) != len(self.joint_order):
            raise ValueError("joint_ids must contain exactly one entry per PACE joint")
        if env.num_envs != self.optimizer.population_size:
            raise ValueError(
                "the mjlab world count must equal the CMA-ES population size "
                f"({env.num_envs} != {self.optimizer.population_size})"
            )

        apply_pace_parameters(
            env,
            articulation,
            joint_ids,
            armature=self.sim_params[:, self.armature_idx],
            damping=self.sim_params[:, self.damping_idx],
            friction=self.sim_params[:, self.friction_idx],
            bias=self.sim_params[:, self.bias_idx],
            delay=self.sim_params[:, self.delay_idx],
            initial_encoder_position=initial_position,
        )

    def _reset_population(self) -> None:
        for i in range(self.optimizer.population_size):
            self.params[i] = torch.tensor(self.optimizer.ask(), device=self.device)
        self.sim_params = self._params_to_sim_params(self.params)

    def _params_to_sim_params(self, params: torch.Tensor) -> torch.Tensor:
        normalized = (params + 1.0) / 2.0
        return self.bounds[:, 0] + normalized * (self.bounds[:, 1] - self.bounds[:, 0])

    def get_best_sim_params(self) -> torch.Tensor:
        return self._params_to_sim_params(torch.tensor(self.optimizer._mean, device=self.device))

    def _print_iteration(self) -> None:
        min_score, min_index = torch.min(self.scores, dim=0)
        max_score = torch.max(self.scores)
        print("Max score:", max_score.item())
        print("Min score:", min_score.item(), "at index:", min_index.item())
        print("Armature:", self.sim_params[min_index, self.armature_idx].tolist())
        print("Viscous Friction:", self.sim_params[min_index, self.damping_idx].tolist())
        print("Static/Dynamic Friction:", self.sim_params[min_index, self.friction_idx].tolist())
        print("Bias:", self.sim_params[min_index, self.bias_idx].tolist())
        print("Delay:", self.sim_params[min_index, self.delay_idx].item())
        print(f"Elapsed time: {(datetime.now() - self._timer_start).total_seconds():.1f} seconds")
        self._timer_start = datetime.now()
        self._log()

    def _log(self) -> None:
        min_score, min_score_index = torch.min(self.scores, dim=0)
        max_score = torch.max(self.scores)
        for i, name in enumerate(self.joint_order):
            self.writer.add_histogram(
                f"4_Bias/distribution_{name}",
                self.sim_params[:, self.bias_idx][:, i],
                self.iteration_counter,
            )
            self.writer.add_histogram(
                f"3_Static_Dynamic_Friction/distribution_{name}",
                self.sim_params[:, self.friction_idx][:, i],
                self.iteration_counter,
            )
            self.writer.add_histogram(
                f"2_Viscous_Friction/distribution_{name}",
                self.sim_params[:, self.damping_idx][:, i],
                self.iteration_counter,
            )
            self.writer.add_histogram(
                f"1_Armature/distribution_{name}",
                self.sim_params[:, self.armature_idx][:, i],
                self.iteration_counter,
            )
            self.writer.add_scalar(
                f"4_Bias/best_{name}",
                self.sim_params[min_score_index, self.bias_idx][i].item(),
                self.iteration_counter,
            )
            self.writer.add_scalar(
                f"3_Static_Dynamic_Friction/best_{name}",
                self.sim_params[min_score_index, self.friction_idx][i].item(),
                self.iteration_counter,
            )
            self.writer.add_scalar(
                f"2_Viscous_Friction/best_{name}",
                self.sim_params[min_score_index, self.damping_idx][i].item(),
                self.iteration_counter,
            )
            self.writer.add_scalar(
                f"1_Armature/best_{name}",
                self.sim_params[min_score_index, self.armature_idx][i].item(),
                self.iteration_counter,
            )
        self.writer.add_histogram(
            "0_Delay/distribution", self.sim_params[:, self.delay_idx], self.iteration_counter
        )
        self.writer.add_scalar(
            "0_Delay/best",
            self.sim_params[min_score_index, self.delay_idx].item(),
            self.iteration_counter,
        )
        self.writer.add_scalar("0_Episode/score", min_score.item(), self.iteration_counter)
        self.writer.add_scalar("0_Episode/max_score", max_score.item(), self.iteration_counter)
        self.writer.add_scalar(
            "0_Episode/diff_score",
            (
                (max_score - min_score)
                / torch.clamp_min(min_score.abs(), torch.finfo(self.scores.dtype).eps)
            ).item(),
            self.iteration_counter,
        )

    def save_checkpoint(self, mean: torch.Tensor, iteration: int, finished: bool = False) -> None:
        min_index = torch.argmin(self.scores_buffer[iteration])
        best_trajectory = self.sim_dof_pos_buffer[min_index].detach().cpu()
        run_dir = Path(self.writer.log_dir)
        torch.save(best_trajectory, run_dir / "best_trajectory.pt")
        torch.save(mean.detach().cpu(), run_dir / f"mean_{iteration:03}.pt")
        if finished and self.save_optimization_process:
            torch.save(
                {
                    "params_buffer": self.sim_params_buffer.detach().cpu(),
                    "scores_buffer": self.scores_buffer.detach().cpu(),
                },
                run_dir / "progress.pt",
            )

    def close(self) -> None:
        self.writer.close()
