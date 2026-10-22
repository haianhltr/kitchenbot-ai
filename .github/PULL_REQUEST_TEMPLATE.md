Closes #

## REPORT
**Done:** what exists now that did not before, in two or three lines.
**Verified by:** the commands run and their actual output (numbers, not adjectives).
**Measured:** any metric from docs/evaluation.md this touches, as measured, with seed and env count.
**Reused vs developed:** what came from Isaac Lab / Isaac Sim / third parties, what is ours.
**Not done / follow-ups:** anything in the issue left out, and why.
**Risks:** what could invalidate this.
**GPU time:** wall-clock of any training or long sim run, with num_envs.

## Checklist
- [ ] `pytest` passes locally; sim tests run with `--run-sim` if the change touches simulation
- [ ] every run referenced above has a committed config under `configs/`
- [ ] `physics=isaacsim_physx` on every command and config
- [ ] no file over 5 MB, no checkpoints, logs, datasets or secrets
- [ ] docs updated where behaviour or numbers changed
