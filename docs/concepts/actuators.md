# PACE actuator model

PACE uses encoder-frame position feedback (`q_encoder = q - bias`), DC motor
torque-speed saturation, passive joint armature/damping/friction, and a fixed
per-world torque delay.  mjlab stores encoder calibration with the opposite
sign internally; the PACE compatibility layer performs that conversion at the
boundary so configuration and fitted values preserve their original meaning.
