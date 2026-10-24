# Status

Rewritten by kb-director after every merge. Newest state only; history is in git.

**As of:** 2026-10-23, #2, #23, #24, #28, #29, #30, #35, #40, #42, #43, #46, #47, #48, #49, #52,
#56 to #60, #68, #70 and #71 merged, D012 and D013 decided, #66, #67 and #69 open and in
progress, Isaac Sim not yet installed. Actuals: those twenty issues
(55 estimated hours) were finished in about nine hours of wall-clock by parallel worker sessions,
reviewed and merged by the director. 1,226 unit tests run in under a minute without a simulator.
CI is green on `main`.

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
- Dishwasher placement and drafts (#59): `kitchen_dims.DISHWASHER_POSE_KITCHEN` puts the compact
  ArtVIP unit on a 0.55 m plinth to the robot's right (provisional; rack base at
  (0.35, -0.26, 0.65) m with the rack out 0.40 m, slot entries at 0.74 m), with the measured
  dishwasher constants, `RACK_TINES_POSE_IN_LINK`, `rack_fits_plate` and the tested frame math in
  `scene/dishwasher_frames.py`; `scripts/make_dishwasher_overrides.py` writes
  `dishwasher_2_overrides.usda`, a layer that references the git-ignored ArtVIP file, puts the root
  link at the origin, replaces the `sdf` colliders with convex ones and removes ArtVIP's door
  script, checked with `usd-core` to 1 mm; `scene/dishwasher_cfg.py`, `scene/kitchen_scene_cfg.py`
  and `scripts/smoke/load_kitchen.py` are UNTESTED DRAFTS parsed on CI; `docs/isaaclab_3_api_notes.md`
  section 9 holds the 20-item checklist for #10 and #11. The joint prims are
  `RevoluteJoint_dishwasher_2_middle` and `PrismaticJoint_dishwasher_2_{up,down}`, not the `E_*`
  link names the issues quote.
- Seeded randomization `scene/randomization.py`: every row of `docs/evaluation.md` section 2 as
  `RandomizationRanges`, one deterministic `EpisodeSpec` per (scenario, split, seed), the ten
  `configs/eval/s<N>_<split>.yaml` files generated and checked by `scripts/make_eval_configs.py`,
  and `scripts/evaluate.py --config` recording the YAML's SHA256 in the result (`schema_version`
  2) (#52). #13 wraps the sampler in Isaac Lab event terms.
- Evaluator: `scripts/evaluate.py` with the protocol's flags and `--extra KEY=VALUE`,
  `kitchenbot.evaluation` (scenario registry S0 to S4 with placeholder loaders naming their issues,
  strict JSON schema, Wilson intervals, failure-tag precedence, seed splits with `seen` as the
  default and a held-out notice on `unseen`), a synthetic `fake` scenario (#23) and a synthetic
  `dry` scenario (#60, below). `scripts/plot_results.py` (success rates with intervals, failure
  histograms, torso error over time from optional traces) and `scripts/update_experiments.py`
  (idempotent table rows, synthetic documents refused by default) (#24); continuous metrics get
  mean, median and a bootstrap interval (#40).
- Judges `evaluation/judges.py` (#57): the reference implementation of the section 3 definitions
  from state alone: `plate_in_slot` (the `RackSlotLayout` slot volume, with signed offsets),
  `vertical_error_deg`, `StabilityWindow`, `GraspJudge` (lifted 15 cm, held 2 s), `PlacementJudge`
  (in the slot, within 15 deg of vertical, still for 2 s after release; rack contact above 20 N
  before release as a metric) and `episode_outcome` over the failure-tag precedence. Thresholds that
  are also planner numbers are read from `PlannerConfig`; the two stability tolerances are
  provisional. Decisions taken where the protocol was silent are on #22 (falls and timeouts fail S2
  and S3 too; a collision becomes a tag only when the placement also failed).
- Hand controllers: the interface `controllers/hand/base.py` (`HandController`: open, pre_grasp,
  close, is_holding, plus `update(dt)` and the commanded `posture`) with `MockHand`; Dex3, arm and
  waist joint limits from the URDF in `robots/joint_limits.py`; the three Dex3 postures as closing
  fractions in one table, the two thumb closing directions in one constant marked unverified;
  `Dex3SynergyHand` logic with sensing injected as callables (#42); and `KinematicFollowHand`
  (#58), the labelled attach baseline: `close()` attaches once when the palm is within 3 cm of the
  rim circle, `update` writes the plate from the palm with the captured offset, `open()` restores
  dynamics with the palm velocity, sensing and writing injected. #17 wires the baseline to the
  plate rigid body, #18 the contact sensors and the Dex3 tuning.
- Plate-to-rack planner core `planning/plate_to_rack.py`: the section 9 FSM (idle to done, plus
  failed), observation in, palm pose target and hand verb out, every number sourced or marked
  provisional, failure tags from the evaluator's taxonomy (#43). The lift waypoint is aimed one
  position tolerance above the 15 cm so the exit height meets the judges (#60 found it leaving at
  14.9 cm). The fakes it is tested against live in `planning/fakes.py` (#60). #21 wires it to the
  simulation and must fix which rim point the grasp frame is (the #58 tests and the dry run hold the
  bottom rim, the reach analysis the top rim; the top rim is the physical one).
- Arm target pipeline `controllers/target_filter.py` (world target → pelvis frame → reachability
  gate → slew at 0.3 m/s and 1 rad/s, returning the planner's `arm_infeasible`), shared pose and
  quaternion helpers in `kitchenbot/geometry.py`, numpy forward kinematics of the right arm from
  the URDF origins (`robots/kinematics.py`) and two 2 cm voxel reachability envelopes:
  `right_arm_envelope.npz` with the waist at zero (#48) and `right_arm_envelope_waist.npz` with
  the waist sampled (pitch 0 to 29.8 deg forward, yaw ±20 deg; #56). #14 builds the IK on the
  pipeline, #16 validates both envelopes against the real IK and decides which gates when.
- Reach-gap analysis `docs/reach_gap_analysis.md` with `scripts/reach_gap_analysis.py` (#56): the
  planner's waypoints for every committed seed of the ten (scenario, split) pairs against both
  envelopes, per stand-off and plate position, with the figure `docs/figures/reach_gap.png`. The
  result is D012 (section below).
- Dry run `evaluation/dry_run.py` (#60): scenario `dry` runs the whole chain on CI without a
  simulator (eval YAML → sampler → planner against the fakes behind the arm target pipeline →
  judges → result JSON → figures), marked synthetic so it can never enter `docs/experiments.md`.
  With the reachability gate on (waist-zero envelope, palm-link point, 0.30 m stand-off since D012)
  every episode still ends `ik_infeasible`, the reach gap as a number until #14 and #16; with `--extra reach_gate=false` the chain reaches `done` on 50 of 50 seeds by
  construction. It is the template of a `ScenarioBackend` for #21 and #22 (note on #22).
- Stand environment drafts for #5: `locomotion/stand_env_cfg.py`, `commands.py`,
  `upper_body_action.py`, `rewards.py`, `curriculums.py`, `agents/rsl_rl_ppo_cfg.py`,
  `robots/g1_dex3.py`, `tasks/registry.py` and `configs/train/stand_g1_dex3.yaml`, every file
  headed UNTESTED DRAFT and only parsed on CI; the tested parts are the upper-body sampling and
  curriculum math (`locomotion/upper_body_targets.py`) and the observation layout
  (`locomotion/obs_layout.py`, 139 per frame, 695 over five frames). `docs/isaaclab_3_api_notes.md`
  pins every Isaac Lab symbol used to a v3.0.0 file and line (commit e3f7d9f, read-only clone at
  `C:\User_Storage\code\Projects\_ref\IsaacLab`) and lists eight places where the source differs
  from the architecture text, among them: Isaac Lab 3.0 quaternions are (x, y, z, w) while this
  repository's geometry is (w, x, y, z), PhysX is a CLI preset resolved through a `PresetCfg`
  field, un-actuated joint targets go through `set_position_index` (#46). #5 runs and fixes the
  drafts against the 15-item checklist in PR #53; if the waist lean below is adopted, the
  upper-body command must also sample a waist pitch target before #6 trains (notes on #5, #6).
- Waist lean command in the stand environment drafts (#68, D012): `WaistLeanCommand` with its
  numpy rules in `locomotion/waist_lean_targets.py` (targets in [0, 0.52] rad, exactly zero for
  half of the draws, a new target every 1 to 3 s, the observed command ramped at 1 rad/s), the
  observation term at index 9 of each frame, and `waist_tracking_l1` in place of the zero-waist
  reward. #5 runs it; the deployed lean command (#21) must not jump faster than 1 rad/s.
- Dishwasher cabinet (#70, D013): `scripts/make_cabinet_usd.py` writes
  `scene/assets/kitchen/dishwasher_cabinet.usda`, the plinth as a low cabinet (carcass with a toe
  kick on the plinth footprint, 18 mm door front in the counter's body colour on the face the
  dishwasher front looks at, steel-grey bar handle; six static colliders, one friction material),
  with a strict `--check`; the clearances come from `footprints_kitchen()`: 16.5 mm under the
  open dishwasher door, 370 mm to the right leg. `kitchen_scene_cfg.cabinet_cfg()` spawns it;
  the `CABINET_*` constants and `PLINTH_GAP_M` live in `kitchen_dims`, all provisional.
- Report and video skeletons: `docs/report.md` with every empty slot an HTML comment naming the
  issue that fills it, and `docs/demo_video_script.md` (16 shots, 163 s, selection rules fixed
  before any result exists, overlay specification for `record_demo.py`); `tests/test_docs_skeletons.py`
  keeps result rows digit-free (#49). Protocol gaps it exposed were added as tasks to #7, #20, #21,
  #22 and #26 (unseen-split runs, committed figures, the Agile checkpoint licence).
- The reach-gap story in `docs/report.md` and `README.md` (#71): a Method subsection with the
  paper numbers and their caveats, D012 and D013 in the decision list, the README section "What
  has been checked without a simulator", and the demo script's raised dishwasher and lean.
- `scripts/host_check.ps1`: read-only Windows readiness check with a `-Fix` mode for long paths,
  git `core.longpaths` and `uv` (#35). Run on this host on 2026-10-23: GPU, driver 616.92, disk
  and RAM pass; long paths, git option, VS 2022 C++ tools and `uv` fail; a reboot is pending.
- GitHub labels, milestones M0 to M6, issues #1 to #26 (planning IDs P01 to P26), the pinned GPU
  queue #27, and no-simulation issues #28, #29, #30, #35, #40, #42, #43, #46 to #49, #52 and #56
  to #60 added so work continued before the install.

## Decided on 2026-10-23: stand-off, waist lean, IK point, raised dishwasher (D012, D013)

From the #56 analysis (`docs/reach_gap_analysis.md`: forward kinematics only, no collision, no
palm orientation, so an upper bound) the operator chose the lower-risk row:

| stand-off | waist at zero | commanded lean up to 29.8 deg |
| --- | --- | --- |
| 0.45 m (before) | S2 0 / 0 | S2 0 / 30 |
| **0.30 m (now)** | S2 2 / 30 | S2 73 / 96, **98 / 100 with the IK point at the finger bases** |
| 0.25 m (fallback) | S2 21 / 58 | S2 97 / 100 |

(seen / unseen percent of S2 episodes with every waypoint reachable, 2 cm margin.)

`ROBOT_STANDOFF_M` is 0.30 m (constant and test changed; everything else reads it). The lower-body
policy takes a commanded forward waist pitch of 0 to 0.52 rad from its first training run (#5,
#6); the planner commands the lean during pre-grasp, grasp and lift and zero again from the
counter-edge waypoint on (#21); #14 puts the IK `body_offset` at the finger bases, (0.0777,
-0.0016, 0) m in the palm link frame. 0.25 m is the fallback if #14 and #16 measure short, and
#16 says so with numbers before any constant moves again; the plate box stays. The compact
dishwasher stays raised on its 0.55 m plinth beside the counter (D013); #10 dresses the plinth as
a low cabinet with a door front in the counter's colour, and `DISHWASHER_POSE_KITCHEN` holds at
either stand-off (the lateral clearance to the right leg does not depend on it). Until #16, the
committed envelopes and the dry run's gate still use the palm-link point with the waist at zero,
so the gate-on dry run still fails every episode as `ik_infeasible` at 0.30 m, as the tests pin.
Recorded on #5, #6, #10, #11, #14, #16, #21.

## What is running

No training run has started. Of the six no-simulation issues opened on 2026-10-24 after D012 and
D013, #68, #70 and #71 are merged and three are in progress, one worker each: #66 (finger-base envelopes and a selectable gate envelope,
kb-controls), #67 (planner waist-lean command and the top-rim grasp frame, kb-manip), #69 (contract audit with
`tests/test_contracts.py`, kb-eval). Every other open issue needs Isaac Sim. The director is pre-downloading the
Isaac Lab, Isaac Sim and PyTorch wheels with `pip download` (no install) into
`C:\User_Storage\code\Projects\_ref\wheels\isaaclab-3.0.0-cu130` so #1 can install offline.

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
Windows gate). After #66, #67 and #69 nothing that does not need Isaac Sim is left; the next move is the operator's.
