# Importing PACE

The intended public imports match upstream PACE:

```python
from pace_sim2real import (
    CMAESOptimizer,
    PaceCfg,
    PaceSim2realEnvCfg,
    PaceSim2realSceneCfg,
)
from pace_sim2real.utils import PaceDCMotor, PaceDCMotorCfg, bind_environment
import pace_sim2real.tasks
```

`pace_sim2real.tasks` registers `Isaac-Pace-Anymal-D-v0` with `mjlab`'s task
registry.  Create it using `mjlab.tasks.registry.load_env_cfg` and
`mjlab.envs.ManagerBasedRlEnv`; no Isaac Sim application launcher is needed.
PACE never monkey-patches mjlab classes. `make_env()` (used by the supplied
scripts) registers the environment automatically. If you construct
`ManagerBasedRlEnv` yourself and want to retain the historical three-argument
`CMAESOptimizer.update_simulator(robot, ...)` call, register it explicitly:

```python
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
bind_environment(env)
optimizer.update_simulator(env.scene["robot"], joint_ids, initial_encoder_position)
```

New applications should prefer the explicit four-argument form with `env`.
