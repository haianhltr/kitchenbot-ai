# Status

Rewritten by kb-director after every merge. Newest state only; history is in git.

**As of:** 2026-10-23, #2 and #28 merged, Isaac Sim not yet installed.

## What exists

- Approved architecture, roadmap, setup guide, evaluation protocol, decision log, team protocol.
- Package skeleton with docstrings naming which issue creates each file.
- CI: lint, unit tests, 5 MB file guard (guard proven to fail on a 6 MB file, PR #31 experiment).
  Pre-commit hook with ruff; `sim` marker plumbing proven with a placeholder test (#2).
- Provisional joint map `source/kitchenbot/robots/joint_maps.py` and URDF fixture: 43 movable
  joints, 12 legs / 3 waist / 7 per arm / 7 per Dex3 hand, Dex3 limits sign-mirrored between
  hands (#28). The Isaac Sim USD check is still #3's job.
- GitHub labels, milestones M0 to M6, issues #1 to #26 (planning IDs P01 to P26), the pinned GPU
  queue #27, and no-simulation issues #28, #29, #30 added so work continues before the install.

## What is running

No training run has started. In progress now, each by a worker session in its own worktree,
none needing Isaac Sim:

| Issue | Owner | What |
| --- | --- | --- |
| #29 | kb-sim | Asset fetch script, licence manifest, inspect ArtVIP and MolmoSpaces candidates |
| #30 | kb-sim | Procedural rimmed plate USD assets, three diameters |
| #23 | kb-eval | Evaluator runner, schema, fake scenario on CI |

#1 (the install) is on hold without `status:ready` until the operator reboots. #24 becomes ready
when #23 merges; #3 when #1 closes.

## What needs the operator

1. Install Visual Studio 2022 Build Tools with the C++ workload and `uv` (no reboot needed).
2. Enable Windows long paths, then reboot (`docs/setup.md` section 2).
3. After the reboot, tell the director; it re-adds `status:ready` to #1.

## Next critical step

#1: install Isaac Lab 3.0.0 + Isaac Sim 6.2 on Windows and pass smoke steps 1 to 4. Then #3
(now a diff against the #28 map) and #4 (the Windows gate). Until the reboot, #29, #30 and #23
run in parallel without a simulator.
