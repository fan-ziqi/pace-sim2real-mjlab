# Custom PACE environment

The public Python names are intentionally retained.  `PaceSim2realEnvCfg`,
`PaceSim2realSceneCfg`, `PaceCfg`, and `PaceDCMotorCfg` are mjlab-native
counterparts; `AnymalDPaceEnvCfg` is a real inheritable class rather than a
factory alias.

```python
from pace_sim2real.tasks.manager_based.pace.anymal_pace_env_cfg import AnymalDPaceEnvCfg

class MyPaceCfg(AnymalDPaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.0025
        self.actions.joint_pos.scale = 1.0
```

`PaceDCMotorCfg` accepts the old scalar, per-joint list, and regex dictionary
forms for gains, passive parameters, and encoder bias.  `max_delay` means a
fixed delay of **computed torque** in simulation steps.

Register a custom config with `mjlab.tasks.registry.register_mjlab_task`, then
import its package before constructing `ManagerBasedRlEnv`.

## 1. Define fitted joints and bounds

PACE fits armature, viscous damping, Coulomb friction, encoder bias, and a
torque delay for an ordered subset of joints. `joint_order` is used by
collection files, CMA-ES parameter vectors, and deployment, so keep it stable.

```python
from dataclasses import dataclass, field
import torch
from pace_sim2real import PaceCfg

JOINT_ORDER = ["hip_left", "knee_left", "hip_right", "knee_right"]

@dataclass(kw_only=True)
class MyRobotPaceCfg(PaceCfg):
    robot_name: str = "my_robot"
    data_dir: str = "my_robot/chirp_data.pt"
    joint_order: list[str] = field(default_factory=lambda: list(JOINT_ORDER))
    bounds_params: torch.Tensor = field(default_factory=lambda: torch.tensor([
        *[[1e-5, 1.0]] * 4,  # armature
        *[[0.0, 7.0]] * 4,   # viscous damping
        *[[0.0, 0.5]] * 4,   # Coulomb friction
        *[[-0.1, 0.1]] * 4,  # encoder bias
        [0.0, 10.0],         # torque delay in physics steps
    ], dtype=torch.float32))
```

`bounds_params` must have `4 * len(joint_order) + 1` rows.

## 2. Create the mjlab asset and PACE actuator

Use an `EntityCfg` backed by your MuJoCo `MjSpec` or URDF loader.
`PaceDCMotorCfg` accepts the original scalar, list, and regex dictionary
formats. `joint_names_expr` remains an alias for mjlab's `target_names_expr`.

```python
import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from pace_sim2real.utils import PaceDCMotorCfg

PACE_ACTUATOR = PaceDCMotorCfg(
    joint_names_expr=JOINT_ORDER,
    saturation_effort={".*": 140.0}, effort_limit={".*": 89.0},
    velocity_limit={".*": 8.5}, stiffness={".*": 85.0}, damping={".*": 0.6},
    armature={".*": 0.0}, friction={".*": 0.0}, viscous_friction={".*": 0.0},
    encoder_bias={".*": 0.0}, max_delay=10,
)

def get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file("assets/my_robot.xml"),
        articulation=EntityArticulationInfoCfg(actuators=(PACE_ACTUATOR,)),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
        ),
    )
```

`max_delay` delays computed, clipped **torque**, rather than a position
command. The continuous CMA-ES coordinate is truncated to a non-negative
physics-step count, matching the Isaac Lab implementation.

Every `joint_order` member must be controlled by `PaceDCMotor`; a native mjlab
actuator cannot substitute for it because it delays commands instead of the
computed torque. Set `max_delay` at least to the truncated upper bound of the
last PACE parameter range. Collection and fitting reject an undersized or wrong
actuator rather than silently changing the objective.

## 3. Use the base PACE environment configuration

`PaceSim2realEnvCfg` is directly usable by custom robots. When `scene.robot`
is set, it supplies the original all-actuator position action,
joint-position/joint-velocity/last-action observations, zero-weight joint-limit
reward, and no-op timeout.

```python
from dataclasses import dataclass, field
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg

@dataclass(kw_only=True)
class MyRobotPaceSceneCfg(PaceSim2realSceneCfg):
    robot: EntityCfg = field(default_factory=get_robot_cfg)

@dataclass(kw_only=True)
class MyRobotPaceEnvCfg(PaceSim2realEnvCfg):
    scene: MyRobotPaceSceneCfg = field(default_factory=MyRobotPaceSceneCfg)
    sim2real: MyRobotPaceCfg = field(default_factory=MyRobotPaceCfg)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.dt = 0.0025
```

The action vector covers every controlled actuator joint, while `joint_order`
only contains the fitted subset. PACE's collection and fitting scripts scatter
fitted targets into that full vector and hold other action columns at zero. If
you replace `actions.joint_pos`, it must control every `joint_order` name.

To make policy observations contain fitted joints only (the ANYmal task's
36-value layout), build observation terms using
`SceneEntityCfg("robot", joint_names=tuple(self.sim2real.joint_order))`.

## 4. Register and run the custom task

Put registration in a module that is imported before the task is created:

```python
from mjlab.tasks.registry import register_mjlab_task
from pace_sim2real.tasks.manager_based.pace.agents.rsl_rl_ppo_cfg import ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Pace-My-Robot-v0",
    env_cfg=MyRobotPaceEnvCfg(),
    play_env_cfg=MyRobotPaceEnvCfg(episode_length_s=1.0e9),
    rl_cfg=ppo_runner_cfg(),
)
```

```bash
uv run python scripts/pace/data_collection.py --task Mjlab-Pace-My-Robot-v0 --task-module my_robot.tasks --device cuda:0
uv run python scripts/pace/fit.py --task Mjlab-Pace-My-Robot-v0 --task-module my_robot.tasks --device cuda:0
```

The saved `.pt` data has `time`, `dof_pos`, and `des_dof_pos` in
`joint_order`. Every sample is one physics step apart; fitting rejects another
time base so damping and delay retain their physical meaning.

`--task-module` imports the module that calls `register_mjlab_task`; install
your custom package or put its parent directory on `PYTHONPATH` first. The same
option is accepted by `pace-train`, `pace-play`, `scripts/zero_agent.py`, and
`scripts/random_agent.py`.

## 5. Deploy fitted values

`pace-fit` writes `logs/pace/<robot_name>/<run>/mean_*.pt`. Split the vector
into blocks of `n = len(joint_order)` and pass the values into a deployment
actuator before constructing the environment:

```python
import torch

mean = torch.load("logs/pace/my_robot/<run>/mean_000.pt", weights_only=True)
n = len(JOINT_ORDER)
DEPLOY_ACTUATOR = PaceDCMotorCfg(
    joint_names_expr=JOINT_ORDER,
    saturation_effort=140.0, effort_limit=89.0, velocity_limit=8.5,
    stiffness=85.0, damping=0.6,
    armature=mean[:n].tolist(),
    viscous_friction=mean[n:2*n].tolist(),
    friction=mean[2*n:3*n].tolist(),
    encoder_bias=mean[3*n:4*n].tolist(),
    max_delay=int(mean[4*n].item()),
)
```

For an already-created entity, use its PACE actuator's
`update_encoder_bias(...)` and `update_time_lags(...)`; values persist across
full and partial mjlab resets. Recreate the environment after changing
armature, damping, or friction because they are MuJoCo model parameters. See
[Actuator model](../concepts/actuators.md) and [Fitting API](../api/optim.md).
