# Real-world workflow

Export synchronized 400 Hz tensors named `time`, `dof_pos`, and `des_dof_pos`.
All joints must use the task's canonical order and encoder frame.  Run a small
CMA-ES fit first, validate the result on held-out commands, and only then widen
the population or apply parameters to a hardware validation simulator.  PACE
does not make hardware deployment safe by itself.
