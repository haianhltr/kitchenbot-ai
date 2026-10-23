# Status

Rewritten by kb-director after every merge. Newest state only; history is in git.

**As of:** 2026-10-23, #2, #23, #24, #28, #29, #30, #35, #40, #42, #43, #46, #47 and #48 merged,
Isaac Sim not yet installed. Actuals: those thirteen issues (38 estimated hours) were finished in
about five hours of wall-clock by parallel worker sessions, reviewed and merged by the director.

## What exists

- Approved architecture, roadmap, setup guide, evaluation protocol, decision log, team protocol.
- Package skeleton with docstrings naming which issue creates each file.
- CI: lint, unit tests, 5 MB file guard (guard proven to fail on a 6 MB file, PR #31 experiment).
  Pre-commit hook with ruff; `sim` marker plumbing proven with a placeholder test (#2).
- Provisional joint map `source/kitchenbot/robots/joint_maps.py` and URDF fixture (joint origins
  included since #48): 43 movable joints, 12 legs / 3 waist / 7 per arm / 7 per Dex3 hand, Dex3
  limits sign-mirrored between hands (#28). The Isaac Sim USD check is still #3's job.
- Three rimmed plate assets (22, 25, 27 cm) as watertight text USD with rigid-body, mass,
  convex-decomposition collision and a friction-1.0 physics material, generated and validated by
  `scripts/make_plate_usd.py`; `usd-core` now installs on CI (#30). PhysX behaviour is #9's job.
- `scripts/fetch_assets.py` with a 93-file manifest (ArtVIP dishwashers and cabinets, two
  MolmoSpaces THOR objects, 195 MB, already downloaded into the main checkout's git-ignored
  `third_party/` directory) and `scripts/usd_inspect.py`. Finding: ArtVIP `dishwasher_2` has a
  separately articulated door and racks (0.2 m travel), so #11 wraps it (D011) (#29).
- Authored kitchen geometry: `scene/kitchen_dims.py` holds every dimension with its source (counter
  0.90 m high, 0.60 m deep; rack from the ArtVIP measurements) and `RackSlotLayout` with the
  planner's slot poses; `scripts/make_rack_tines_usd.py` and `scripts/make_counter_usd.py` write
  `rack_tines.usda`, `counter.usda` and `floor.usda` as static primitive colliders, each checked
  against the module to 1 mm (#47). Known tension: the tines leave 12 mm per side and stop an
  8.5 degree lean, tighter than the planner's provisional tolerances (noted on #21).
- Evaluator: `scripts/evaluate.py` with the protocol's flags, `kitchenbot.evaluation` (scenario
  registry S0 to S4 with placeholder loaders naming their issues, strict JSON schema, Wilson
  intervals, failure-tag precedence, seed splits with `seen` as the default and a held-out notice
  on `unseen`), and a synthetic `fake` scenario that runs on CI (#23). `scripts/plot_results.py`
  (success rates with intervals, failure histograms, torso error over time from optional traces)
  and `scripts/update_experiments.py` (idempotent table rows, synthetic documents refused by
  default) (#24); continuous metrics get mean, median and a bootstrap interval (#40).
- Hand controller interface `controllers/hand/base.py` (`HandController`: open, pre_grasp, close,
  is_holding, plus `update(dt)` and the commanded `posture`) with `MockHand` for tests; Dex3, arm
  and waist joint limits copied from the URDF in `robots/joint_limits.py`; the three Dex3 postures
  as closing fractions in one table, the two thumb closing directions in one constant marked
  unverified; `Dex3SynergyHand` logic with sensing injected as callables (#42). #17 adds the
  kinematic baseline on the same interface, #18 the Isaac Lab wiring and the tuning.
- Plate-to-rack planner core `planning/plate_to_rack.py`: the section 9 FSM (idle to done, plus
  failed), observation in, palm pose target and hand verb out, every number sourced or marked
  provisional, failure tags from the evaluator's taxonomy. Tested against a fake arm, hand, plate
  and rack, and against the #42 `MockHand` through the hand verbs (#43). #21 wires it to the
  simulation, #22 decides on vectorisation.
- Arm target pipeline `controllers/target_filter.py` (world target → pelvis frame → reachability
  gate → slew at 0.3 m/s and 1 rad/s, returning the planner's `arm_infeasible`), shared pose and
  quaternion helpers in `kitchenbot/geometry.py`, numpy forward kinematics of the right arm from
  the URDF origins (`robots/kinematics.py`) and a 2 cm voxel reachability envelope sampled from
  2,000,000 configurations (`controllers/data/right_arm_envelope.npz`, 4 KB, seed 48, waist at
  zero) (#48). #14 builds the IK on it, #16 validates the envelope against the real IK.
- Stand environment drafts for #5: `locomotion/stand_env_cfg.py`, `commands.py`,
  `upper_body_action.py`, `rewards.py`, `curriculums.py`, `agents/rsl_rl_ppo_cfg.py`,
  `robots/g1_dex3.py`, `tasks/registry.py` and `configs/train/stand_g1_dex3.yaml`, every file
  headed UNTESTED DRAFT and only parsed on CI; the tested parts are the upper-body sampling and
  curriculum math (`locomotion/upper_body_targets.py`) and the observation layout
  (`locomotion/obs_layout.py`, 138 per frame, 690 over five frames). `docs/isaaclab_3_api_notes.md`
  pins every Isaac Lab symbol used to a v3.0.0 file and line (commit e3f7d9f, read-only clone at
  `C:\User_Storage\code\Projects\_ref\IsaacLab`) and lists eight places where the source differs
  from the architecture text, among them: Isaac Lab 3.0 quaternions are (x, y, z, w) while this
  repository's geometry is (w, x, y, z), PhysX is a CLI preset resolved through a `PresetCfg`
  field, un-actuated joint targets go through `set_position_index` (#46). #5 runs and fixes the
  drafts against the 15-item checklist in PR #53.
- `scripts/host_check.ps1`: read-only Windows readiness check with a `-Fix` mode for long paths,
  git `core.longpaths` and `uv` (#35). Run on this host on 2026-10-23: GPU, driver 616.92, disk
  and RAM pass; long paths, git option, VS 2022 C++ tools and `uv` fail; a reboot is pending.
- GitHub labels, milestones M0 to M6, issues #1 to #26 (planning IDs P01 to P26), the pinned GPU
  queue #27, and no-simulation issues #28, #29, #30, #35, #40, #42, #43, #46 to #49 and #52 added
  so work continued before the install. CI is green on `main`.

## What is running

No training run has started. Two no-simulation issues are in progress: #49 (technical report
skeleton and demo video script, kb-eval; the report file had to be written by the director because
the harness blocks worker sessions from writing report files) and #52 (seeded randomization sampler
and the ten eval YAMLs, kb-sim). Every other open issue needs Isaac Sim.

#1 (the install) is on hold without `status:ready` until the operator reboots; #3 follows it.

## What needs the operator

1. In an elevated PowerShell: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\host_check.ps1 -Fix`
   (enables long paths, sets the git option, installs `uv`).
2. Install Visual Studio 2022 Build Tools with the "Desktop development with C++" workload
   (https://aka.ms/vs/17/release/vs_BuildTools.exe), then reboot.
3. After the reboot, run the check again without `-Fix`; when it reports no FAIL, tell the
   director, who re-adds `status:ready` to #1.
4. Read the MolmoSpaces dataset card's licence wording ("research and educational use" on top of
   CC BY 4.0) and say whether its two static filler assets may appear in the portfolio video; they
   are optional (D011).

## Next critical step

#1: install Isaac Lab 3.0.0 + Isaac Sim 6.2 on Windows and pass smoke steps 1 to 4. Then #3
(now a diff against the #28 map, the #46 asset draft and the #48 kinematic chain) and #4 (the
Windows gate). After #49 and #52 merge, nothing that does not need Isaac Sim is left; the next move
is the operator's.
