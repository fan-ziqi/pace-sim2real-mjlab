# Simulation migration notes

PACE's four simulator parameters map directly to native per-world MuJoCo-Warp
model fields:

| PACE parameter | mjlab / MuJoCo field |
| --- | --- |
| Armature | `dof_armature` |
| Viscous friction | `dof_damping` |
| Static/dynamic friction | `dof_frictionloss` |
| Encoder bias | `EntityData.encoder_bias` |
| Command delay | `PaceDCMotor` torque-delay buffer |

PACE defines `q_encoder = q - bias`; mjlab defines
`q_encoder = q + encoder_bias`.  The port applies the sign conversion at its
boundary, so PACE configuration files and fitted values retain their original
meaning.

The upstream fixed-base excitation setting is retained (`dt=0.0025 s`,
`decimation=1`, base at `z=1.0 m`).

Unlike mjlab's generic actuator latency, PACE computes and torque-speed-clips
the PD command from the current state first, then delays that torque.  This is
the upstream physical model and is important when a joint moves during a delay.
