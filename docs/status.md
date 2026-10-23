# Status

Rewritten by kb-director after every merge. Newest state only; history is in git.

**As of:** 2026-10-23, #2, #23, #28, #29 and #30 merged, Isaac Sim not yet installed.

## What exists

- Approved architecture, roadmap, setup guide, evaluation protocol, decision log, team protocol.
- Package skeleton with docstrings naming which issue creates each file.
- CI: lint, unit tests, 5 MB file guard (guard proven to fail on a 6 MB file, PR #31 experiment).
  Pre-commit hook with ruff; `sim` marker plumbing proven with a placeholder test (#2).
- Provisional joint map `source/kitchenbot/robots/joint_maps.py` and URDF fixture: 43 movable
  joints, 12 legs / 3 waist / 7 per arm / 7 per Dex3 hand, Dex3 limits sign-mirrored between
  hands (#28). The Isaac Sim USD check is still #3's job.
- Three rimmed plate assets (22, 25, 27 cm) as watertight text USD with rigid-body, mass,
  convex-decomposition collision and a friction-1.0 physics material, generated and validated by
  `scripts/make_plate_usd.py`; `usd-core` now installs on CI (#30). PhysX behaviour is #9's job.
- `scripts/fetch_assets.py` with a 93-file manifest (ArtVIP dishwashers and cabinets, two
  MolmoSpaces THOR objects, 195 MB, already downloaded into the main checkout's git-ignored
  `third_party/` directory) and `scripts/usd_inspect.py`. Finding: ArtVIP `dishwasher_2` has a
  separately articulated door and racks (0.2 m travel), so #11 wraps it (D011) (#29).
- Evaluator: `scripts/evaluate.py` with the protocol's flags, `kitchenbot.evaluation` (scenario
  registry S0 to S4 with placeholder loaders naming their issues, strict JSON schema, Wilson
  intervals, failure-tag precedence, seed splits with `seen` as the default and a held-out notice
  on `unseen`), and a synthetic `fake` scenario that runs on CI (#23). 160 unit tests pass.
- GitHub labels, milestones M0 to M6, issues #1 to #26 (planning IDs P01 to P26), the pinned GPU
  queue #27, and no-simulation issues #28, #29, #30 added so work continues before the install.

## What is running

No training run has started. In progress now, each by a worker session in its own worktree,
none needing Isaac Sim:

| Issue | Owner | What |
| --- | --- | --- |
| #24 | kb-eval | Plot script and `docs/experiments.md` table generation from result JSONs |
| #35 | kb-sim | Windows host check script (long paths, build tools, uv, driver, disk) |

#1 (the install) is on hold without `status:ready` until the operator reboots; #3 follows it.

## What needs the operator

1. Install Visual Studio 2022 Build Tools with the C++ workload and `uv` (no reboot needed).
2. Enable Windows long paths, then reboot (`docs/setup.md` section 2).
3. After the reboot, tell the director; it re-adds `status:ready` to #1.
4. Read the MolmoSpaces dataset card's licence wording ("research and educational use" on top of
   CC BY 4.0) and say whether its two static filler assets may appear in the portfolio video; they
   are optional (D011).

## Next critical step

#1: install Isaac Lab 3.0.0 + Isaac Sim 6.2 on Windows and pass smoke steps 1 to 4. Then #3
(now a diff against the #28 map) and #4 (the Windows gate). Until the reboot, #24 and #35 finish
without a simulator; after them nothing is left that does not need Isaac Sim.
