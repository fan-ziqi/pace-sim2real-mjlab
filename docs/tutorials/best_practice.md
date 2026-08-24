# Best practices

- Use physically credible bounds and validate on an excitation trajectory not
  used for fitting.
- Keep real-data timestamps and commands synchronized at 400 Hz; PACE rejects
  a different timebase rather than guessing.
- Start with 64–256 candidates and inspect the reported trajectory allocation.
- Delay is integer-valued, so expect a discontinuous objective around lag
  boundaries.
- Never deploy an identified parameter set to hardware without independent
  simulation and conservative torque/temperature checks.
