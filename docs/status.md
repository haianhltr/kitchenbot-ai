# Status

Rewritten by kb-director after every merge. Newest state only; history is in git.

**As of:** 2026-10-23, repository created, no code, Isaac Sim not yet installed.

## What exists

- Approved architecture, roadmap, setup guide, evaluation protocol, decision log, team protocol.
- Package skeleton with docstrings naming which issue creates each file.
- CI: lint, unit tests, 5 MB file guard.
- GitHub labels, milestones M0 to M6, issues #1 to #26 (planning IDs P01 to P26), the pinned GPU
  queue #27, and two no-simulation issues #28 and #29 added so work continues before the install.

## What is running

Nothing. No training run has started. Ready now, none needing Isaac Sim:

| Issue | Owner | What |
| --- | --- | --- |
| #2 | kb-sim | Verify CI, pre-commit, sim-marker placeholder |
| #28 | kb-sim | Provisional joint map and Dex3 limits from the Unitree URDF |
| #29 | kb-sim | Asset fetch script, licence manifest, inspect ArtVIP and MolmoSpaces candidates |
| #23 | kb-eval | Evaluator runner, schema, fake scenario on CI |

#1 (the install) is on hold without `status:ready` until the operator reboots.

## What needs the operator

1. Install Visual Studio 2022 Build Tools with the C++ workload and `uv` (no reboot needed).
2. Enable Windows long paths, then reboot (`docs/setup.md` section 2).
3. After the reboot, tell the director; it re-adds `status:ready` to #1.
4. Open `kb-sim` and `kb-eval` sessions with the bootstrap prompts in `AGENTS.md` section 8.

## Next critical step

#1: install Isaac Lab 3.0.0 + Isaac Sim 6.2 on Windows and pass smoke steps 1 to 4. Then #3 and
#4 (the Windows gate). Until the reboot, #2, #28, #29 and #23 run in parallel without a simulator.
