from __future__ import annotations

import pytest
import torch
from mjlab.actuator.actuator import ActuatorCmd

from pace_sim2real import CMAESOptimizer
from pace_sim2real.scripts._common import make_env, pace_joint_ids, prepare_pace_model


@pytest.mark.parametrize("device", ["cpu", *(["cuda:0"] if torch.cuda.is_available() else [])])
def test_anymal_pace_environment_applies_distinct_world_parameters(device: str, tmp_path) -> None:
    env = make_env("Isaac-Pace-Anymal-D-v0", num_envs=4, device=device)
    try:
        robot = env.scene["robot"]
        joint_ids = pace_joint_ids(robot, env.cfg.sim2real.joint_order, env.device)
        observations, _ = env.reset()
        assert observations["actor"].shape == (4, 36)
        assert env.action_space.shape == (4, 12)
        actuator = robot.actuators[0]
        assert actuator._torque_delay_buffer.current_lags.tolist() == [10, 10, 10, 10]

        columns = len(joint_ids)
        bias = torch.linspace(-0.05, 0.05, 4, device=env.device).unsqueeze(1).repeat(1, columns)
        armature = torch.linspace(0.01, 0.04, 4, device=env.device).unsqueeze(1).repeat(1, columns)
        damping = torch.full((4, columns), 1.5, device=env.device)
        friction = torch.full((4, columns), 0.1, device=env.device)
        delay = torch.tensor([[0], [2], [5], [10]], device=env.device)
        initial = torch.zeros((4, columns), device=env.device)
        prepare_pace_model(
            env,
            robot,
            joint_ids,
            armature=armature,
            damping=damping,
            friction=friction,
            bias=bias,
            delay=delay,
            initial_encoder_position=initial,
        )

        dof_ids = robot.indexing.joint_v_adr[joint_ids]
        assert torch.allclose(env.sim.model.dof_armature[:, dof_ids], armature)
        assert torch.allclose(robot.data.encoder_bias[:, joint_ids], -bias)
        assert torch.allclose(robot.data.joint_pos_biased[:, joint_ids], initial, atol=1e-5)
        assert actuator._torque_delay_buffer.current_lags.tolist() == [0, 2, 5, 10]

        # CMA-ES and user-supplied parameters must survive both global and
        # partial mjlab resets.  The upstream actuator only cleared history;
        # it never restored the construction-time bias/lag defaults.
        env.reset()
        assert torch.allclose(robot.data.encoder_bias[:, joint_ids], -bias)
        assert actuator._torque_delay_buffer.current_lags.tolist() == [0, 2, 5, 10]
        env.reset(env_ids=torch.tensor([1, 3], device=env.device))
        assert torch.allclose(robot.data.encoder_bias[:, joint_ids], -bias)
        assert actuator._torque_delay_buffer.current_lags.tolist() == [0, 2, 5, 10]

        # PACE delays the torque computed at the old state, not the target then
        # recomputed against the new state.  This differs materially in motion.
        actuator.update_time_lags(torch.ones(4, device=env.device, dtype=torch.long))
        first = ActuatorCmd(
            position_target=torch.ones((4, 12), device=env.device),
            velocity_target=torch.zeros((4, 12), device=env.device),
            effort_target=torch.zeros((4, 12), device=env.device),
            pos=torch.zeros((4, 12), device=env.device),
            vel=torch.zeros((4, 12), device=env.device),
        )
        actuator.compute(first)
        second = ActuatorCmd(
            position_target=torch.zeros((4, 12), device=env.device),
            velocity_target=torch.zeros((4, 12), device=env.device),
            effort_target=torch.zeros((4, 12), device=env.device),
            pos=torch.full((4, 12), 0.5, device=env.device),
            vel=torch.zeros((4, 12), device=env.device),
        )
        delayed = actuator.compute(second)
        assert torch.allclose(delayed, torch.full_like(delayed, 85.0))
        # Invalid candidates must not partially overwrite the live model.
        armature_before = env.sim.model.dof_armature.clone()
        with pytest.raises(ValueError, match="max_delay"):
            prepare_pace_model(
                env,
                robot,
                joint_ids,
                armature=armature * 2,
                damping=damping * 2,
                friction=friction * 2,
                bias=bias * 2,
                delay=torch.full((4, 1), 11, device=env.device),
            )
        assert torch.equal(env.sim.model.dof_armature, armature_before)
        # Reapplying a candidate must clear the prior candidate's delayed
        # torques.  The first sample after preparation is consequently the
        # current torque rather than the old 85 Nm command.
        prepare_pace_model(
            env,
            robot,
            joint_ids,
            armature=armature,
            damping=damping,
            friction=friction,
            bias=bias,
            delay=torch.ones((4, 1), device=env.device),
        )
        isolated = actuator.compute(second)
        assert torch.allclose(isolated, torch.full_like(isolated, -42.5))
        env.step(torch.zeros((4, 12), device=env.device))
        assert torch.isfinite(robot.data.joint_pos[:, joint_ids]).all()
        assert torch.isfinite(robot.data.joint_vel[:, joint_ids]).all()

        # The original three-argument API still works for environments made by
        # PACE helpers, without monkey-patching any mjlab class.
        optimizer = CMAESOptimizer(
            bounds=env.cfg.sim2real.bounds_params,
            population_size=4,
            log_dir=tmp_path,
            joint_order=env.cfg.sim2real.joint_order,
            max_iteration=1,
            data={
                "time": torch.tensor([0.0], device=env.device),
                "dof_pos": torch.zeros((1, 12), device=env.device),
                "des_dof_pos": torch.zeros((1, 12), device=env.device),
            },
            device=env.device,
        )
        try:
            optimizer.update_simulator(robot, joint_ids, torch.zeros((4, 12), device=env.device))
        finally:
            optimizer.close()
    finally:
        env.close()
