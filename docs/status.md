# Status

Rewritten by kb-director after every merge. Newest state only; history is in git.

**As of:** 2026-10-22, repository created, no code.

## What exists

- Approved architecture, roadmap, setup guide, evaluation protocol, decision log, team protocol.
- Package skeleton with docstrings naming which issue creates each file.
- CI: lint, unit tests, 5 MB file guard.
- GitHub labels, milestones M0 to M6, issues #1 to #26 (planning IDs P01 to P26, same numbers) and
  the pinned GPU queue #27. Ready now: #1 and #2.

## What is running

Nothing. No training run has started.

## What needs the operator

1. Enable Windows long paths and reboot (`docs/setup.md` section 2).
2. Install Visual Studio 2022 Build Tools with the C++ workload.
3. Install `uv`.
4. Open the `kb-sim` session with the bootstrap prompt in `AGENTS.md` section 8 so it can start P01.

## Next critical step

#1: install Isaac Lab 3.0.0 + Isaac Sim 6.2 on Windows and pass smoke steps 1 to 4. Then #3 and
#4 (the Windows gate). Everything else waits on them. #2 (scaffolding) can run in parallel.
