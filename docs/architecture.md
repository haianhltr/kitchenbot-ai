# KitchenBot AI — technical architecture (MVP 1)

Status: draft for operator approval, 2026-10-22. Facts below were verified against Isaac Lab 3.0.0
source and documentation and Isaac Sim 6.2 documentation on that date; items marked **VERIFY** are
inferred and must be confirmed in the milestone that first touches them.

## 1. Scope

**MVP 1.** A Unitree G1 humanoid, simulated with PhysX in Isaac Sim 6.2 through Isaac Lab 3.0.0,
stands in a detailed kitchen on a reinforcement-learning balance policy that we train, reaches a
rigid plate on the counter with a classical inverse-kinematics arm controller, grasps it with a
Dex3-1 three-finger hand, and places it into a slot of an articulated dishwasher's lower rack. Object
poses are known from simulation state; there is no camera perception. Success is measured by the
protocol in `docs/evaluation.md`.

**Non-goals for the MVP.** Walking, opening the dishwasher door, vision, bimanual manipulation,
cabinet interaction, sim-to-real deployment. Each has a seam reserved in section 13.

## 2. System overview

```
                    ┌──────────────────────── Task planner (FSM, 50 Hz) ───────────────────────┐
                    │ idle → pre-grasp → grasp → lift → transport → align → insert → release   │
                    └───────┬───────────────────────────┬───────────────────────────┬──────────┘
                            │ wrist pose target          │ hand command (open/close) │ stand command (v=0)
                            ▼                           ▼                           ▼
             ┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────────────┐
             │ Arm controller       │    │ Hand controller      │    │ Lower-body policy (frozen)   │
             │ DifferentialIK (dls) │    │ Dex3 synergies       │    │ PPO actor, trained in M1     │
             │ 7 joints per arm     │    │ 7 joints per hand    │    │ 12 leg + 3 waist joints      │
             └──────────┬───────────┘    └──────────┬───────────┘    └──────────────┬───────────────┘
                        │ arm targets (14)          │ hand targets (14)              │ leg+waist targets (15)
                        └──────────────┬────────────┴────────────────────────────────┘
                                       ▼
                         Isaac Lab ActionManager: three action terms, disjoint joint sets,
                         concatenated into one 43-joint position-target command
                                       ▼
                         Isaac Sim 6.2 / PhysX  —  G1 (G1_29DOF_CFG, floating base)
                         kitchen scene: counter, sink, articulated dishwasher, cabinets, plate
                                       ▼
                         Observations → (policy obs, planner state, eval metrics)
```

The conflict between locomotion and manipulation is resolved by **joint ownership**: every one of the
43 joints has exactly one writer. Nothing arbitrates at runtime because nothing overlaps.

## 3. Toolchain

Pinned in `docs/setup.md`: Isaac Lab 3.0.0, Isaac Sim 6.2.0.0, Python 3.12, PyTorch 2.11 (CUDA 13.0),
RSL-RL 5.5.1, physics preset `isaacsim_physx` on every command. Windows 11 first, Ubuntu 24.04 as
the documented fallback.

## 4. Robot model

| Item | Value | Source |
| --- | --- | --- |
| Isaac Lab config | `isaaclab_assets.robots.unitree.G1_29DOF_CFG`, docstring "for locomanipulation tasks" | Isaac Lab 3.0.0 `unitree.py` |
| USD | Isaac Sim asset `Robots/Unitree/G1/g1.usd`, 45 joints, 46 links, 43 DoF | Isaac Sim 6.2 humanoid asset table |
| Layout | 29 body joints + 14 Dex3 joints. Matches Unitree `g1_29dof_with_hand_rev_1_0.urdf` joint names (**VERIFY** by dumping joint names at M0) | Isaac Lab Pink config for G1 uses those names |
| Base | `fix_root_link=False`, marked configurable. Fixed-pelvis development mode flips it to `True` | `unitree.py` |
| Actuators | legs and feet `DCMotorCfg`, waist and arms and hands `ImplicitActuatorCfg` | `unitree.py` |
| Known defect | hip roll effort and velocity limits are the pre-rev-1.0 values (88 N·m, 32 rad/s) instead of 139 N·m, 20 rad/s; fix PR open upstream | IsaacLab issue #8530, PR #8687 |

Joint names and limits below are taken from the URDF in #28 (`robots/joint_maps.py`, fixture `tests/fixtures/g1_dex3_joints_urdf.json`); the check against the Isaac Sim USD is pending in #3.

Joint groups and names (Unitree rev 1.0 naming):

| Group | Count | Names |
| --- | --- | --- |
| Legs | 12 | `{left,right}_hip_{pitch,roll,yaw}_joint`, `{left,right}_knee_joint`, `{left,right}_ankle_{pitch,roll}_joint` |
| Waist | 3 | `waist_{yaw,roll,pitch}_joint` |
| Arms | 14 | `{left,right}_shoulder_{pitch,roll,yaw}_joint`, `{left,right}_elbow_joint`, `{left,right}_wrist_{roll,pitch,yaw}_joint` |
| Hands (Dex3-1) | 14 | `{left,right}_hand_thumb_{0,1,2}_joint`, `{left,right}_hand_index_{0,1}_joint`, `{left,right}_hand_middle_{0,1}_joint` |

We override the hip roll limits in our own robot config (`kitchenbot/robots/g1_dex3.py`, a copy of
`G1_29DOF_CFG` with the rev 1.0 values) rather than waiting for upstream.

Hand fallback: Unitree publishes `g1_29dof_mode_15_with_dex1_1.urdf`, the 29-DoF body with its own
two-jaw Dex1 gripper. If the Dex3 grasp is not stable after the 15-hour budget in M4, we convert
that URDF with `scripts/tools/convert_urdf.py` (documented for Windows) and switch the hand
controller to a one-dimensional gripper. Both hands share the `HandController` interface in section 8.

## 5. Joint ownership and action composition

| Joints | Owner | Action term | Rate | Notes |
| --- | --- | --- | --- | --- |
| Legs (12) + waist (3) | Lower-body policy | `LowerBodyPolicyActionCfg` (ours) | 50 Hz policy, 200 Hz physics | Frozen TorchScript actor; outputs position targets around the default pose. The waist belongs to the balance policy, as in AMO and GR00T's decoupled controller, so it can counter-lean when the arm extends; a reward keeps it near zero unless commanded |
| Arms (14) | Arm controller | `UpperBodyFollowAction` (ours) fed by `DifferentialIKController` for the right arm and a constant tucked posture for the left | 50 Hz | Right arm is the task arm |
| Hands (14) | Hand controller | `UpperBodyFollowAction` fed by synergy targets | 50 Hz | Open, pre-grasp and closed postures interpolated by the planner |

Isaac Lab's manager-based environment concatenates action terms with disjoint `joint_names`
regexes, so this composition needs no custom arbitration. Two terms are ours:

- `LowerBodyPolicyActionCfg` wraps the frozen actor. Isaac Lab 3.0 ships a reference for exactly
  this pattern, `AgileBasedLowerBodyActionCfg` in the contrib task
  `IsaacContrib-PickPlace-Locomanipulation-G1-Abs`, which runs NVIDIA's pretrained
  `Policies/Agile/agile_locomotion.pt` on the hip, knee and ankle joints while Pink IK drives the
  upper body. We reuse the wrapper's structure and the Agile checkpoint as a **baseline**, and
  replace Pink IK with DifferentialIK because Pink is Linux-only in 3.0.
- `UpperBodyFollowAction` is an action term with agent action dimension zero: it writes joint
  position targets for a joint set from a source object instead of from the agent's action vector.
  In training (section 6) the source is a `CommandTerm` that samples random upper-body targets. In
  deployment the source is the arm and hand controllers. One mechanism, two sources, so the policy
  sees identical upper-body behaviour in both. Isaac Lab 3.0 has no built-in event that holds
  un-actuated joints at targets, and `joint_pos_target` is deprecated in favour of
  `actuators.target_command.position`, which this term uses.

Frames: the lower-body policy and the IK both work in the pelvis frame. World-frame targets from the
planner are transformed into the pelvis frame every step, so body sway is compensated by the arm
controller rather than fought by the legs.

## 6. Lower-body balance policy (M1, developed)

Why train our own when a pretrained Agile policy exists: the career focus is robot learning, the
operator chose to retrain reused components, and we need a policy whose training explicitly
randomizes upper-body motion so that arm movement during manipulation does not cause falls. The
Agile policy is kept as the baseline row in every table.

**Design references.** The recipe below is the one used by HOMIE (RSS 2025: legs-only actions,
random upper-body targets resampled every second with linear interpolation and a growing amplitude
curriculum), AMO (RSS 2025, 29-DoF G1: 15-joint lower policy over legs and waist, upper-body
actuator states in the observation) and NVIDIA's GR00T decoupled whole-body controller (15 actions,
action scale 0.25, 50 Hz, full 29-joint positions and velocities observed, 6-frame history). None of
their training code runs on Isaac Lab 3.0, so the environment is written here.

**Environment** `KitchenBot-Stand-G1-Dex3` (ours), derived from Isaac Lab's flat velocity task but
with these changes:

| Aspect | Setting |
| --- | --- |
| Asset | `kitchenbot/robots/g1_dex3.py` (G1_29DOF_CFG with fixed hip limits), floating base |
| Terrain | flat plane (kitchen floor), no height scan |
| Action | legs and waist, 15 position targets, scale 0.25 around the default pose |
| Upper body during training | `UpperBodyCommand`, a `CommandTerm` with `resampling_time_range` 1 to 3 s, samples arm and hand targets inside joint limits and interpolates linearly; a `CurriculumTerm` scales the sampling range from 0.2 to 1.0 of the joint range as the fall rate drops; 20 percent of episodes instead command a slow reach of the right arm into the front workspace. `UpperBodyFollowAction` applies the targets |
| Observations | base angular velocity (3), projected gravity (3), velocity command (3), leg and waist joint positions and velocities relative to default (30), last action (15), arm and hand joint positions and velocities (56), arm and hand target positions (28); 5-frame history |
| Commands | planar velocity in [-0.3, 0.3] m/s and yaw rate in [-0.3, 0.3] rad/s, with 60 percent of episodes at exactly zero (standing) |
| Rewards | tracking of linear and angular velocity, base height around 0.75 m, flat orientation, waist deviation from zero, penalties on joint torques, joint accelerations, action rate, feet slip, undesired contacts of knees and torso; standing-specific bonus for feet staying planted when the command is zero |
| Terminations | torso or pelvis contact with the ground, tilt over 60 degrees, timeout at 20 s |
| Domain randomization | friction 0.6 to 1.0, added base mass -1 to +3 kg, random pushes 10 to 30 N every 3 s, actuator gain scale 0.9 to 1.1 |
| Algorithm | RSL-RL PPO, actor and critic MLP [256, 128, 128], 1,500 iterations as a first budget, 24 steps per env |
| Compute | `physics=isaacsim_physx`, `num_envs` 1024 for the first run then 2048, `--viz none`, VRAM recorded |
| Export | TorchScript `.pt` plus a JSON with joint order and observation layout, so the wrapper and a future real-robot bridge load the same artifact |

Calibration only, from published runs on an RTX 4090 at 4096 environments: Unitree reports its G1
velocity policy converging in about 2,000 iterations in 30 to 45 minutes; NVIDIA's throughput table
implies about one hour for the 3,000-iteration rough G1 task. Our flat standing task at 2,048
environments on an RTX 5080 should land in the same order of magnitude, but no wall-clock is
promised until the first run is measured and recorded in `docs/experiments.md`.

## 7. Arm controller (M3, reused with configuration)

`DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")`, wired
through `DifferentialInverseKinematicsActionCfg(asset_name="robot", joint_names=[right arm 7],
body_name=<right palm link, VERIFY name in USD>, body_offset=<palm center>)`. Isaac Lab's Franka
`IsaacContrib-Lift-Cube-Franka-IK-Abs` task is the template. We add:

- A target filter: the planner's pose target is slewed at most 0.3 m/s and 1 rad/s so the IK step is
  small and the balance policy sees smooth upper-body motion.
- Reachability check: targets outside a precomputed workspace envelope are rejected and reported as
  `ik_infeasible` by the evaluator.
- Joint-limit margin of 5 degrees on the wrist to keep the Dex3 clear of the forearm.

Posture regularization is not available in DifferentialIK. If elbow drift becomes a problem, the
fallback is Isaac Lab's `OperationalSpaceController` with a null-space posture task, which supports
floating bases, at the cost of switching the arm to effort control. This is a recorded risk, not a
plan.

## 8. Grasping (M4, developed)

**Plate asset.** A rigid dinner plate with a 10 mm rim thickness and a 25 cm diameter, mass 0.35 kg,
friction 1.0, convex-decomposition collision. Two diameters in the seen split, one held out. A thin
real-world plate is a known PhysX failure case; the rim is the concession, stated in the report.

**HandController interface.** `open()`, `pre_grasp()`, `close()`, `is_holding()`. Implementations:

1. `KinematicFollowHand` (baseline, M4a): on `close()`, when the palm is within 3 cm of the rim, the
   plate is switched to kinematic and its pose is written each step from the palm pose with the
   captured offset; `open()` restores it to dynamic with the palm's velocity. Runs on GPU PhysX.
   Isaac Lab's official attach mechanism, `SurfaceGripper`, requires `--device cpu` in 3.0 and is
   therefore used only for cross-checking the baseline, not for training.
2. `Dex3SynergyHand` (M4b): three stored joint postures per hand (open, pre-grasp around a rim,
   closed); `close()` interpolates to the closed posture over 0.5 s while fingertip contact sensors
   (`ContactSensorCfg` filtered on the plate, the pattern used by Isaac Lab's Kuka-Allegro lift task)
   confirm two or more fingertips in contact; `is_holding()` is contact plus plate velocity tracking
   the palm. Finger stiffness and friction are tuned against the S2 protocol.
3. `Dex1GripperHand` (fallback): one prismatic or revolute joint, same interface.

**Status (#42, no simulation).** The interface exists in `controllers/hand/base.py`: the four
methods plus `update(dt)` and a read-only commanded `posture`, with `MockHand` for tests and the
planner. `Dex3SynergyHand` exists as logic only, sensing and target writing injected as callables;
its three postures are closing fractions of the URDF joint limits (`robots/joint_limits.py`) in
one table in `controllers/hand/dex3_postures.py`. The closing directions of `thumb_0` and
`thumb_1` are reasoned from the URDF joint origins and **unverified** until #18 runs them in
simulation; they sit in one constant there. #17 adds `KinematicFollowHand`; #18 adds the contact
sensors and the Isaac Lab wiring and tunes the table.

Decision rule from the plan: if `Dex3SynergyHand` is below the S2 target after 15 hours of effort,
switch to `Dex1GripperHand` and record the decision in `docs/decisions.md`.

## 9. Task planner (M5, developed)

A finite-state machine in `kitchenbot/planning/plate_to_rack.py`:

```
idle → pre_grasp(plate pose + 8 cm above rim) → grasp(close) → lift(+15 cm) →
transport(waypoints: above counter edge → above rack slot at +25 cm) → align(slot pose, plate vertical) →
insert(descend until slot contact or depth reached) → release(open, retreat 10 cm up) → done
```

Each state has a timeout and a failure tag matching `docs/evaluation.md` section 7. Waypoints are
computed from the rack slot pose in the scene config; there is no motion planner in the MVP. Collision
with the rack is avoided geometrically by a corridor above the slot, and measured, not assumed.

The core exists since #43, tested against a fake arm (0.3 m/s, 1 rad/s slew) and hand; #21 wires it
to the simulation. `step(obs, dt)` returns a palm pose target and a hand verb (`open`, `pre_grasp`,
`close`, the `HandController` method names); it never calls the arm or the hand. `plate_pose` in
the observation is the plate's grasp frame on the rim and `slot_pose` is that frame's pose at the
slot entry, so the plate diameter is folded in by the environment layer, not the planner. Each
step checks the current state's failure conditions first, then its exit; the first cause observed
moves the planner to `failed`, which holds the palm where it is and never changes again. Causes seen
in the same step resolve by the section 7 precedence. A 30 s budget (`timeout`) applies in every
state. Timeouts, tolerances (2 cm, 10 deg), the 2 N seat contact, the 5 cm insertion depth and the
15 cm counter-edge clearance are provisional, tuned in #21:

| State | Timeout | Exit condition | Failure tags |
| --- | --- | --- | --- |
| `idle` | 0.5 s | first step after `reset` | `rack_collision`, `timeout` |
| `pre_grasp` | 4.0 s | palm at the grasp frame + 8 cm | `ik_infeasible`, `rack_collision`, `timeout` |
| `grasp` | 2.5 s | `hand_holding` (hand in `pre_grasp` until the palm reaches the rim, then `close`) | `ik_infeasible`, `rack_collision`, `grasp_miss` on timeout without holding |
| `lift` | 2.0 s | palm at the grasp frame + 15 cm | `ik_infeasible`, `drop`, `rack_collision`, `timeout` |
| `transport` | 6.0 s | palm above the counter edge, then 25 cm above the slot | as `lift` |
| `align` | 5.0 s | palm at the plate-vertical orientation from the slot yaw, plate within 15 deg of vertical | as `lift` |
| `insert` | 3.0 s | rack contact of 2 N to 20 N, or palm 5 cm below the slot entry | as `lift`; `rack_collision` above 20 N |
| `release` | 4.0 s | hand open, palm 10 cm up, 2 s hold elapsed, `plate_in_slot` | `ik_infeasible`, `misplace` if not in the slot after the hold, `timeout` |

The timeouts sum to 27 s. The planner is single-environment pure Python (numpy, no simulator
import); #22 decides whether it is vectorised across `num_envs`.

## 10. Scene (M2)

Assembled as one Isaac Lab `InteractiveSceneCfg`:

| Element | Representation | Source |
| --- | --- | --- |
| Floor, counter, sink | USD primitives with collision and basic materials | authored in this repo |
| Dishwasher | `ArticulationCfg`: door revolute joint, lower rack prismatic joint, slots as rigid tines; door starts open and rack pulled out in the MVP | ArtVIP dishwasher twin if one loads in 6.2, otherwise authored USD (section 10a) |
| Cabinets | `ArticulationCfg` with door joints, not actuated in the MVP | ArtVIP or MolmoSpaces cabinets (section 10a) |
| Plate(s) | `RigidObjectCfg` | authored mesh, section 8; YCB `029_plate` as an optional extra unseen object if present in the 6.2 asset root |
| Robot | `ArticulationCfg` from section 4 | Isaac Sim asset |

Collision budgets: convex decomposition on rack tines (they are what the plate touches), convex hulls
for cabinet shells. The scene is one USD layer per element so M4 and M5 can load counter plus
dishwasher without cabinets for speed.

### 10a. Asset sources

Isaac Sim 6.2 ships no kitchen environment and no articulated cabinet or dishwasher, so articulated
furniture comes from outside. Order of preference, by licence and format:

| Source | What | Format | Licence | Notes |
| --- | --- | --- | --- | --- |
| ArtVIP | 476 articulated digital twins including dishwashers and cabinets | USD, built for Isaac Sim 5.0 | Apache-2.0 | First choice; M2 verifies that a dishwasher and a cabinet open in 6.2 and that the rack is a separate articulated link. Listed in NVIDIA's 6.2 third-party asset catalogue |
| MolmoSpaces | Articulated objects and indoor scenes with Isaac exports | USD | code Apache-2.0, data CC BY 4.0 with a `commercial_safe` filter | Second choice for cabinets and counters |
| ReplicaCAD | Articulated kitchen counter, fridge, doors | URDF plus GLB, converted with `convert_urdf.py` | CC BY 4.0 | Fallback for the counter block |
| Authored USD | Dishwasher body, door hinge, rack slide, tines | USD authored in this repo | ours | Guaranteed fallback for the dishwasher; the rack tines are authored in every case so slot geometry is under our control |
| PartNet-Mobility | 48 dishwashers, 345 cabinets | URDF plus OBJ | non-commercial research only, gated access | Not used: the licence would restrict the portfolio and access requires an institutional e-mail |

Inspected 2026-10-23 with `usd-core` (#29, not yet loaded in Sim 6.2): ArtVIP `dishwasher_2`
(alternate `dishwasher_4`) has a separately articulated door (revolute, 0 to 90 degrees) and two
prismatic racks with 0.2 m travel (#11 assumes 0.4 m), all with `sdf` collision, no tines or slot
geometry, so `rack_tines.usda` stays authored. MolmoSpaces has no dishwasher and its THOR cabinets
are static. Cabinet dressing: ArtVIP `kitchen/cupboard` and `counter`, MolmoSpaces `thor_Cabinet` as
static filler; files and licences are in `scene/assets/third_party/MANIFEST.md`.

Dish props: the plate is authored (section 8). YCB `029_plate` is 258 mm across, 24 mm tall and
279 g; whether a USD for it exists under `Props/YCB/Axis_Aligned` in the 6.2 asset root is checked
in M2 (**VERIFY**), and if so it becomes an extra unseen object, not the training plate.

Plates: `plate_22.usda`, `plate_25.usda` and `plate_27.usda` are authored procedurally by #30 (`scripts/make_plate_usd.py`); physics validation (drop test, `RigidObjectCfg` factory) is #9.

Import commands, when a URDF is involved:
`uv run --extra importers isaaclab -p scripts/tools/convert_urdf.py <in.urdf> <out_dir> --merge_joints --fix_base`
(documented for Windows; the `importers` extra has no platform restriction).

## 11. Observation and action spaces

Lower-body policy: section 6. Composite task environment used by the planner and the evaluator:

| Observation group | Terms |
| --- | --- |
| `proprio` | all 43 joint positions and velocities, base angular velocity, projected gravity, pelvis height |
| `task` | plate pose in pelvis frame, slot pose in pelvis frame, palm pose in pelvis frame, hand contact flags |
| `planner` | FSM state one-hot, time in state |

Action: the 43-dimensional concatenation in section 5. The learned-arm stretch goal (M3b) replaces
the DifferentialIK term with an RL policy over the 7 right-arm joints that observes `proprio` and
`task` and is rewarded for palm-to-target distance, orientation alignment, action rate and
collision-free insertion. Same interface, same evaluator.

## 12. Domain randomization, evaluation, logging, checkpoints

- Randomization ranges are frozen in `docs/evaluation.md` section 2 and implemented as Isaac Lab
  event terms in `configs/eval/*.yaml`.
- `scripts/evaluate.py` runs scenario S0 to S4 and writes JSON per `docs/evaluation.md` section 4.
- Training logs: RSL-RL TensorBoard under `logs/` (git-ignored); optional Weights and Biases if the
  operator sets a key, never required.
- Videos: Isaac Lab's `--video` recording for training clips; the demo video is recorded from the
  evaluator with on-screen labels for which controller is active.
- Checkpoints: `checkpoints/<task>/<date>_<commit>/model_<iter>.pt`, git-ignored; the ones behind a
  reported result are uploaded to a GitHub Release and listed with SHA256 in `docs/experiments.md`.

## 13. Sim-to-real seams

Three interfaces are defined now and implemented only for simulation:

- `LowerBodyPolicy.step(obs) -> leg_targets` at 50 Hz, with the JSON joint-order manifest from
  section 6. The real-robot bridge would feed Unitree SDK2 low-state and publish low-cmd.
- `ArmController.step(pose_target) -> arm_targets`.
- `HandController` (section 8); the Dex3 real driver speaks the SDK's seven motors per hand.

A `RobotIO` abstraction holds the joint-name to index map for both sim and the Unitree SDK order.
No hardware code is written in the MVP.

## 14. Reused versus developed

| Reused as is | Reused with configuration | Developed here |
| --- | --- | --- |
| Isaac Sim 6.2, PhysX, G1 USD with Dex3 | `G1_29DOF_CFG` (hip limits fixed), `DifferentialIKController`, `ContactSensorCfg`, RSL-RL PPO | Standing environment with upper-body randomization, its command and curriculum terms, its rewards |
| Isaac Lab manager-based env framework, ActionManager, event terms | Agile lower-body action wrapper pattern, Franka IK-Abs task as template | `LowerBodyPolicyAction`, `UpperBodyFollowAction`, the trained lower-body policy |
| NVIDIA Agile locomotion checkpoint (baseline only) | ArtVIP and MolmoSpaces furniture, URDF converter for ReplicaCAD and the Dex1 fallback | Hand controllers, plate asset, dishwasher rack geometry, kitchen scene assembly, task planner, evaluator, plots, report |
| Published recipes: HOMIE, AMO, GR00T decoupled WBC (design, not code) | | |

## 15. Repository layout

```
kitchenbot-ai/
├── AGENTS.md  README.md  pyproject.toml  .gitignore
├── .github/workflows/ci.yml  .github/ISSUE_TEMPLATE/  .github/PULL_REQUEST_TEMPLATE.md
├── docs/  architecture.md  roadmap.md  setup.md  evaluation.md  experiments.md  status.md  decisions.md
├── source/kitchenbot/
│   ├── robots/        g1_dex3.py (config), joint_maps.py
│   ├── scene/         kitchen_scene_cfg.py, assets/ (USD layers, plate mesh)
│   ├── locomotion/    stand_env_cfg.py, rewards.py, lower_body_action.py, agents/
│   ├── controllers/   arm_ik.py, hand/{base,kinematic_follow,dex3_synergy,dex1}.py
│   ├── planning/      plate_to_rack.py
│   ├── tasks/         composite env cfg and gym registration
│   └── evaluation/    runner.py, metrics.py, failure_tags.py
├── scripts/  train_stand.py  evaluate.py  plot_results.py  record_demo.py  smoke/
├── configs/  train/  eval/
└── tests/    unit (no sim)  sim (marked, local only)
```

`perception/` and `training/` directories are created when their first file exists, not before.

## 16. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Windows toolchain fails the M0 gate | documented NO-GO path to Ubuntu 24.04 in `docs/setup.md` |
| 16 GB VRAM | flat terrain, no height scan, 1024 then 2048 envs, `--viz none` |
| Lower-body policy falls when the arm moves | upper-body randomization in training plus upper-body state in observations; Agile baseline shows whether the problem is ours |
| Dex3 cannot hold a plate | rimmed plate, contact-confirmed closing, 15-hour budget then Dex1 fallback |
| Link and joint names differ from the URDF | M0 dumps names from the USD into `joint_maps.py`; every regex is tested against that dump |
| Default physics is Newton | every command and config sets `isaacsim_physx`; a unit test asserts it |
| Scope creep from the detailed kitchen | cabinets are scene dressing; door starts open; one USD layer per element |

## 17. Decisions already taken

Recorded as the first entries of `docs/decisions.md`: Windows-first with an Ubuntu gate; PhysX over
Newton; DifferentialIK over Pink IK (Linux-only); own lower-body policy over legs and waist with the
Agile checkpoint as baseline; kinematic-follow attach over SurfaceGripper (CPU-only); Dex3 with Dex1
fallback; detailed kitchen with a pre-opened dishwasher door; ArtVIP and MolmoSpaces over
PartNet-Mobility for licence reasons.
