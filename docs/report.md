# KitchenBot AI: technical report

> **Skeleton, no results yet.** This file fixes the shape of the final report before any number exists. Every empty slot is a hidden to-do comment in the page source. Its first words give the number of the GitHub issue whose work produces the content, and `tests/test_docs_skeletons.py` checks that each number is a real issue of this project. Measured values will be copied from `outputs/eval/*.json` through [experiments.md](experiments.md), never typed from memory. Delete this notice when the last slot is filled.

<!-- TODO(#26): checklist map for docs/evaluation.md section 8. Result files for every scenario, both splits and every applicable baseline land in Results, Baselines and ablations and Failure analysis, produced by #6 #7 #8 #14 #15 #16 #17 #18 #20 #21 #22. GAP: no issue runs the unseen split of S0, S2, S3 or S4, S3 with the balance policy, S4 with a fixed pelvis, or the training run without domain randomization. Figures: scripts/plot_results.py (#24) over those files; the torso error figure needs per-sample traces that no backend issue promises, so without them it shows the per-episode distribution. Demo video: docs/demo_video_script.md and #25. Measured values with dates, commits, seeds and GPU time: docs/experiments.md rows from #4, #6 and scripts/update_experiments.py. Reused versus developed table in the README: the same table as the one below (#26). Failure analysis: #26 on the data of #20 and #22. -->

This report describes a simulated Unitree G1 humanoid that stands on a balance policy trained here, reaches a plate on a kitchen counter with an inverse-kinematics arm controller, grasps it with a three-finger hand and places it into the rack of an articulated dishwasher. It is written for a reader who has not seen the repository and knows robotics: each section says what the system does, what was reused and what was developed, how it was measured, and where it fails.

## Abstract

*What goes here: one short paragraph with the task, the approach, the headline measured result with its confidence interval and the largest limitation, written last so that every number in it is a copy of a number in the results section.*

<!-- TODO(#26): write the abstract after the results exist; copy, do not recompute, the headline number and its interval from the S4 unseen rows -->

## Problem and scope

*What goes here: the task in plain words, what the first version deliberately leaves out, and how it fits the longer KitchenBot vision of a robot that clears and loads a kitchen; point to section 1 of [architecture.md](architecture.md) and to the README.*

<!-- TODO(#26): problem statement, non-goals (walking, door opening, vision, bimanual work, sim-to-real) and the one-sentence vision -->

## What was built

*What goes here: one paragraph per subsystem, each ending with a pointer to the matching section of [architecture.md](architecture.md), saying what runs, what is trained and what is scripted.*

### Lower-body balance policy

*What goes here: the reinforcement-learning policy over legs and waist that keeps the robot upright while the arms move, and the reused Agile policy that serves as the baseline; architecture sections 5 and 6.*

<!-- TODO(#6): the trained policy and its export, in the words of the final result -->
<!-- TODO(#7): the action-term wrapper and the Agile baseline wrapper behind one interface -->

### Arm controller

*What goes here: the differential-IK controller on the right arm with the left arm held, and the target filter in front of it; architecture sections 5 and 7.*

<!-- TODO(#16): the arm pipeline as finally built, with the reachability gate and the slew limits -->

### Hand controllers

*What goes here: the three hand controllers behind one interface, stating plainly that the kinematic attach is a baseline and not a grasp; architecture section 8.*

<!-- TODO(#18): the hand that the final results use, and the outcome of the time-boxed decision between the Dex3 and the Dex1 hand -->

### Task planner

*What goes here: the finite-state planner from approach to release, with its per-state timeouts and failure tags; architecture section 9.*

<!-- TODO(#22): the planner as finally wired, including how success is detected from simulation state -->

### Kitchen scene

*What goes here: the kitchen elements, the articulated dishwasher and the rimmed plate, with the origin of every asset; architecture section 10.*

<!-- TODO(#12): the full kitchen composition and its load time and frame rate -->

### Evaluator and plots

*What goes here: the evaluator that runs scenarios S0 to S4 and writes one JSON per run, the plot script and the table generator; [evaluation.md](evaluation.md).*

<!-- TODO(#24): can be written now from the merged evaluator and plot scripts; no run is needed -->

## Toolchain and host

*What goes here: the versions and hardware that produced the results, as observed on the host, next to the pinned versions in [setup.md](setup.md).*

| Item | Value used | Where it is recorded |
| --- | --- | --- |
| Operating system | <!-- TODO(#1): value from the host record --> | host record in experiments.md |
| GPU and driver | <!-- TODO(#1): value from the host record --> | host record in experiments.md |
| System memory | <!-- TODO(#1): value from the host record --> | host record in experiments.md |
| Python | <!-- TODO(#1): value from the host record --> | host record in experiments.md |
| Isaac Lab | <!-- TODO(#1): installed version --> | host record in experiments.md |
| Isaac Sim | <!-- TODO(#1): installed version --> | host record in experiments.md |
| PyTorch and CUDA build | <!-- TODO(#1): installed version --> | host record in experiments.md |
| RSL-RL | <!-- TODO(#1): installed version --> | host record in experiments.md |
| Physics backend | `isaacsim_physx` on every command, asserted by a unit test | decision D003 |
| First Isaac Sim launch | <!-- TODO(#1): observed launch time --> | host record in experiments.md |
| Host check before the install | <!-- TODO(#1): outcome of scripts/host_check.ps1 after the operator's fixes --> | setup.md section 2 |
| Windows go or no-go gate | <!-- TODO(#4): GO or NO-GO with the criteria applied --> | setup.md section 5 |

## Reused versus developed

*What goes here: every component placed in exactly one column, following the rule of AGENTS.md section 7 that "developed" means code in `source/kitchenbot` and everything from Isaac Lab, Isaac Sim, Unitree or a paper's code is "reused"; the structure is the table of architecture section 14.*

| Reused as is | Reused with configuration | Developed here |
| --- | --- | --- |
| <!-- TODO(#26): one row per component, filled from the code as it stands at the release; this is the same table as in the README --> | | |

## Method

*What goes here: for each subsystem, how it works and which parameters matter, with enough detail for a robotics reader to judge the design; every choice that was tuned says on which seeds.*

### Lower-body policy

*What goes here: training environment, observation layout, rewards, curriculum, randomization, algorithm and the export format; the Agile checkpoint is described as reused; architecture section 6.*

<!-- TODO(#46): the observation layout table, from the module that holds it as data -->
<!-- TODO(#5): the environment as it was actually run, including what had to change from the draft -->
<!-- TODO(#6): the training budget actually used and the export manifest -->
<!-- TODO(#8): the arm-sweep robustness test that decided between the own policy and the Agile baseline -->

### Arm controller

*What goes here: the controller configuration, the target slew filter, the reachability envelope and how it was sampled, and the pelvis-frame transform; architecture section 7.*

<!-- TODO(#14): the differential-IK term, the held left arm and the waist hold -->
<!-- TODO(#48): the forward-kinematics sampling of the reachability envelope and the slew filter -->
<!-- TODO(#16): the envelope checked against the real IK and the effect of the filter on balance -->

### Hand controllers

*What goes here: the common interface, the kinematic attach baseline, the Dex3 postures with contact confirmation, the tuning log and, only if it was triggered, the Dex1 fallback; architecture section 8.*

<!-- TODO(#17): the kinematic attach baseline and why it is labelled a baseline -->
<!-- TODO(#18): the Dex3 postures, the contact sensors and the tuning log from experiments.md -->
<!-- TODO(#19): the Dex1 fallback, or one sentence that it was not needed -->

### Task planner

*What goes here: the states, the waypoints, the per-state timeouts and tolerances, and which of them were tuned on which seeds; architecture section 9.*

<!-- TODO(#21): the constants after tuning on the seen split, and the rack force sensor -->
<!-- TODO(#22): success detection and collision tagging, and whether the planner runs once per environment or vectorised -->

### Scene

*What goes here: the kitchen dimensions, the dishwasher twin and every change made to it, the authored rack tines, the plate in its diameters and the randomization of the scene; architecture sections 10 and 10a.*

<!-- TODO(#47): the table of kitchen dimensions and rack slot poses, from the module that holds them -->
<!-- TODO(#11): how the dishwasher was wrapped, its collision change and any scale factor applied -->
<!-- TODO(#9): the plate asset and its drop test -->
<!-- TODO(#13): the randomization terms and the frozen seed splits -->

### Reach-gap analysis before the simulator

Everything in this subsection was computed on paper, not measured in simulation. Isaac Sim was not installed when the work was done. No number below was printed by `scripts/evaluate.py`, so none of it is a result and none of it is in [experiments.md](experiments.md). The question was whether the right hand can reach the plates the planner will ask for. The answer comes from the robot's URDF and the planner's own waypoints, before any controller existed. Sources: [reach_gap_analysis.md](reach_gap_analysis.md) (#56), [decisions.md](decisions.md) (D012, D013) and the issues named in parentheses.

**What was computed (#56).**

- **Two reach envelopes** of the right palm, meaning the origin of `right_hand_palm_link` in the pelvis frame, from forward kinematics of Unitree's `g1_29dof_with_hand_rev_1_0.urdf`. The seven right-arm joints are drawn uniformly inside their limits with the 5 degree wrist margin of architecture section 7. The reachable points are stored as 2 cm voxels. One envelope holds the waist at zero (2,000,000 samples, #48). The other samples the waist: forward pitch 0 to 29.79 deg (the URDF limit), yaw ±20 deg, roll 0 (8,000,000 samples, #56).
- **The planner's own waypoints** (`plan_waypoints` with the default `PlannerConfig`, #43) for the 700 committed episodes of the ten (scenario, split) pairs, S0 to S4, seen and unseen, drawn by the seeded sampler (#52). Each waypoint was put in the pelvis frame, with the pelvis 0.75 m above the floor and a chosen stand-off between the pelvis and the counter's front edge.
- **The test.** An episode counts as reachable only if every waypoint of its set lies inside the envelope with a 2 cm margin. The set for S2, grasp and lift, is `pre_grasp`, `grasp` and `lift`.

**The gap in metres.** At the 0.45 m stand-off of the original design, the grasp points lie 0.52 to 0.77 m ahead of the pelvis in the seen split and 0.45 to 0.68 m in the unseen split, all 0.17 m above it. At that height, 2 cm inside the boundary and 5 to 25 cm to the right of the pelvis, the palm reaches 0.36 to 0.38 m forward with the waist at zero and 0.50 to 0.53 m with the waist sampled. With the waist at zero, which is the envelope the arm target pipeline loads (#48), 0 percent of the S1, S2 and S4 plate episodes had every waypoint reachable at 0.45 m, seen or unseen. With the lean, 0 to 3 percent of the seen episodes and 30 to 42 percent of the unseen ones were reachable there; the unseen plate box is 5 cm closer to the robot.

**The stand-off table.** S2 episodes with every waypoint reachable, in percent, seen / unseen (100 and 50 seeds), 2 cm margin. Source: the S2 sensitivity table of [reach_gap_analysis.md](reach_gap_analysis.md).

| Stand-off | Waist at zero | Lean | Lean, IK point at the finger bases |
| --- | --- | --- | --- |
| 0.45 m (original design) | 0 / 0 | 0 / 30 | 29 / 60 |
| 0.30 m (D012) | 2 / 30 | 73 / 96 | 98 / 100 |
| 0.25 m (D012 fallback) | 21 / 58 | 97 / 100 | 100 / 100 |

*Waist at zero* is the #48 envelope. *Lean* is the #56 envelope, the union over waist postures from upright to 29.79 deg of forward pitch; one held posture of 29.8 deg scores 69 / 94 at 0.30 m. The last column re-samples the lean envelope for a point 7.8 cm along the palm link, the mean of the index and middle finger base joints, where the rim sits in a Dex3 grasp; #14 is to set the IK's controlled point there. No single change closes the gap. At 0.30 m the lean alone gives 73 / 96 and the finger-base point with the waist at zero gives 33 / 72. Moving the plate box 5 cm closer buys what 5 cm of stand-off buys and puts 24 percent of the unseen 27 cm plates past the counter's front edge, so the box stayed.

![Reach envelopes with the plate targets, and S2 against stand-off](figures/reach_gap.png)

Figure, from `scripts/reach_gap_analysis.py` (#56), computed on paper. Left: side slice of both envelopes at pelvis y = -0.15 m, with the S2 and S4 plate targets at the 0.45 m stand-off (hollow) and the 0.25 m stand-off (filled). Middle: slice at the grasp height with the grasp points. Right: S2 against stand-off for both envelopes, seen and unseen. It was drawn before D012, so the targets are drawn at 0.45 and 0.25 m, not at the chosen 0.30 m.

**What the analysis changed.**

- **Slot side (#59, D013).** The first rack pose derived from the architecture, a floor-standing dishwasher under the counter on the right, was out of reach at every stand-off from 0.25 to 0.45 m and in both envelopes: S3 0 / 0 percent. The slot entries need the rack base 0.5 to 0.7 m above the floor and within about 0.25 m to the right of the pelvis. #59 put the compact ArtVIP `dishwasher_2` on a 0.55 m plinth to the robot's right, with the slot entries 0.74 m above the floor and 0.35 m to the right, because the open door and the pulled-out rack must clear the right leg. At that pose the S3 slot targets are 100 / 100 percent reachable at 0.25 to 0.40 m in both envelopes, and with the waist at zero 92 / 100 at 0.45 m. D013 keeps the raised unit. A compact unit on a base cabinet is a real arrangement. #10 dresses the plinth as a low cabinet. A full-size unit loaded through its upper rack is not pursued. The pose is still a paper result; #11 checks it against the loaded asset.
- **Plate side (D012).** On paper, 0.25 m with the lean gives S2 97 / 100 with the torso about 5 cm from the counter's front edge in the worst case. 0.30 m with the lean and the IK point at the finger bases gives 98 / 100 with about 10 cm. The operator chose the second. `ROBOT_STANDOFF_M` is 0.30 m. The lower-body policy is trained from its first run with a commanded forward waist pitch of 0 to 0.468 rad, the soft joint limit (#5, #6; this extends D006; D012 was amended on 2026-10-24 from the 0.52 rad hard limit after the #69 audit, applied by #77, which costs S2 seen 98 to 97 and S4 seen 96 to 92 on paper). The planner commands the lean during `pre_grasp`, `grasp` and `lift` and upright again from the counter-edge waypoint on (#21). #14 sets the IK's `body_offset` to (0.0777, -0.0016, 0) m in the palm link frame. 0.25 m is the fallback, taken only if #14 and #16 measure short in simulation, and #16 reports the numbers before any constant moves again. So far one constant and its test have changed.
- **The 1 mm lift bug (#60).** The CI dry run chains the seeded sampler, the planner, the arm target pipeline, the judges and the result writer against fakes, with no simulator (`scripts/evaluate.py --scenario dry`). Its first run ended every episode in `drop`. The planner aimed the `lift` waypoint at exactly 15 cm above the grasp, the state exits within its 2 cm position tolerance of the waypoint, and the plate never rose above 14.9 cm, 1 mm under the 15 cm the grasp judge asks for ([evaluation.md](evaluation.md), section 3). The planner's tests (#43) and the judges' tests (#57) each passed; the mismatch sat between them. The waypoint is now aimed one tolerance higher, 17 cm, and `PlannerConfig.lift_height_m` stays at the protocol's 15 cm (`tests/test_plan_lift_waypoint.py`, `tests/test_dry_run.py`).

**The dry run today.** On `configs/eval/s4_seen.yaml` and `s4_unseen.yaml`, 50 seeds each, at commit `f2274bf` (2026-10-23): with the reachability gate on (waist-zero envelope, palm-link point, 0.30 m stand-off) 0 of 50 episodes succeed in each split and all 50 end `ik_infeasible`. With `--extra reach_gate=false` 50 of 50 reach `done` in each split. The second number is true by construction, because the fakes cannot fail; the first is the reach gap as a number until #14 and #16 move the point and the envelope. These files are marked synthetic and `scripts/update_experiments.py` refuses them.

**What this does not show.**

- **No collision.** The URDF has no collision meshes. The 5 cm and 10 cm torso-to-counter gaps are estimates from Unitree's published standing thickness of 0.20 m, in the worst case with 2 cm of pelvis noise toward the counter. #10 measures the gap in simulation.
- **No palm orientation and no IK.** A voxel is reachable with some orientation, not with the planner's grasp orientation, so every fraction is an upper bound with respect to orientation. #16 checks both envelopes against the real IK.
- **No balance.** A 27 deg lean (the 0.468 rad soft limit; the analysis sampled to 29.8 deg) moves the upper-body mass toward the toes, near the edge of the URDF range. If the policy shifts the hips back to compensate, the pelvis moves away from the counter and the reach shrinks by the same amount. The pelvis-frame envelope does not see that, and the policy has not been trained with a waist command yet (#5, #6).
- **A reading of the grasp point.** The tables take the grasp point to be the near rim of the plate at the rim top. At the plate centre the gap reopens: S2 at 0.30 m with the lean is 10 / 40 (same sensitivity table). #21 fixes the grasp frame.
- **A sampled boundary.** The envelopes are random samples; near their boundary they are slightly conservative, which the 2 cm margin absorbs.
- **A nominal pelvis.** The pelvis is at 0.75 m plus the specification's noise (±2 cm, ±5 deg). Sway, a crouch or a hip shift are not in the numbers.

Reproduce: `py -3.12 scripts/reach_gap_analysis.py --figure docs/figures/reach_gap.png` (about two minutes, numpy only; 128 s on 2026-10-23). The tables of the analysis were generated before #60 raised the lift waypoint by 2 cm. A re-run at commit `f2274bf` (2026-10-23) reproduced every cell quoted in this subsection; a few other cells move by 1 to 2 points (S2 unseen at 0.35 m with the lean, 82 to 80), and the committed tables are kept as generated.

<!-- TODO(#10): the torso-to-counter gap measured in simulation with the waist leaning, next to the 10 cm estimate above -->
<!-- TODO(#11): the dishwasher pose of #59 checked against the loaded asset, next to the 100 / 100 percent slot reach above -->
<!-- TODO(#16): the envelopes checked against the real IK with the IK point at the finger bases; add the measured reach next to the paper numbers of the stand-off table and say which of them moved -->

## Evaluation protocol

*What goes here: a few sentences on how a trial is run and judged (seen and held-out seeds, success decided from simulation state, intervals on every rate, one primary failure cause per failed trial) and a link to [evaluation.md](evaluation.md) for the definition; the tables of that file are not copied.*

<!-- TODO(#26): the protocol summary, written against the version of evaluation.md at the release -->
<!-- TODO(#13): confirmation that the split definitions in code match evaluation.md -->

## Results

*What goes here: measured values only, each copied from the Evaluation results table of [experiments.md](experiments.md) with its confidence interval, trial count and the name of the JSON file it came from; a bare number is not allowed, and a missed target is stated as missed.*

<!-- TODO(#24): embed the success-rate figure after copying outputs/figures/success_rates.png to a committed folder; the figure comes from scripts/plot_results.py run on the final JSON files -->

### Targets versus measured

*What goes here: one row per target of [evaluation.md](evaluation.md) (the targets section), with the measured value on the unseen split and whether the target was met.*

| Metric | Measured | Interval | Target met | JSON file |
| --- | --- | --- | --- | --- |
| S0 standing duration | <!-- TODO(#7): value from the S0 rows --> | | | |
| S0 torso orientation error | <!-- TODO(#7): value from the S0 rows --> | | | |
| S1 hand target error | <!-- TODO(#16): value from the S1 rows --> | | | |
| S2 grasp success, Dex3 | <!-- TODO(#20): value from the S2 rows --> | | | |
| S2 grasp success, kinematic attach baseline | <!-- TODO(#17): value from the S2 rows --> | | | |
| S3 placement success | <!-- TODO(#21): value from the S3 rows --> | | | |
| S3 collision rate | <!-- TODO(#21): value from the S3 rows --> | | | |
| S4 task success, balance policy active | <!-- TODO(#22): value from the S4 rows --> | | | |
| S4 fall rate | <!-- TODO(#22): value from the S4 rows --> | | | |
| S4 completion time (median, successes only) | <!-- TODO(#22): value from the S4 rows --> | | | |

### S0 Stand

*What goes here: standing duration, fall rate and torso orientation error under pushes for the own policy and the Agile baseline, seen and unseen.*

<!-- TODO(#7): embed the torso error figure from outputs/figures/torso_error.png, noting whether it shows traces over time or only the per-episode distribution -->

| Split | Lower-body policy | Standing duration | Fall rate | Torso orientation error | Trials | JSON file |
| --- | --- | --- | --- | --- | --- | --- |
| seen | own policy v1 | <!-- TODO(#7): the three metrics with intervals, trials and file name from the experiments rows --> | | | | |
| seen | Agile baseline | <!-- TODO(#7): the three metrics with intervals, trials and file name from the experiments rows --> | | | | |
| unseen | own policy v1 | <!-- TODO(#7): GAP, no issue schedules the unseen S0 run; fill from it if it is run --> | | | | |
| unseen | Agile baseline | <!-- TODO(#7): GAP, no issue schedules the unseen S0 run; fill from it if it is run --> | | | | |

### S1 Stand and reach

*What goes here: hand target error, the fraction of trials under the distance threshold and the fall rate, with a fixed pelvis and with the balance policy.*

| Split | Base | Lower-body policy | Hand target error | Hand target within threshold | Fall rate | Trials | JSON file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seen | fixed pelvis | none | <!-- TODO(#14): the metrics with intervals, trials and file name --> | | | | |
| seen | balance policy | own policy v1 | <!-- TODO(#15): the metrics with intervals, trials and file name --> | | | | |
| seen | balance policy | Agile baseline | <!-- TODO(#15): the metrics with intervals, trials and file name --> | | | | |
| unseen | balance policy | own policy v1 | <!-- TODO(#16): the metrics with intervals, trials and file name --> | | | | |
| unseen | balance policy | Agile baseline | <!-- TODO(#16): fill only if the baseline was run on the unseen split, otherwise say so --> | | | | |

### S2 Grasp and lift

*What goes here: grasp success and fall rate for each hand, with a fixed pelvis and with the balance policy, and the difference between them explained through the failure tags.*

| Split | Base | Lower-body policy | Hand | Grasp success rate | Fall rate | Trials | JSON file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seen | fixed pelvis | none | KinematicFollowHand | <!-- TODO(#17): the metrics with intervals, trials and file name --> | | | |
| seen | fixed pelvis | none | Dex3SynergyHand | <!-- TODO(#18): the metrics with intervals, trials and file name --> | | | |
| seen | balance policy | own policy v1 | Dex3SynergyHand | <!-- TODO(#20): the metrics with intervals, trials and file name --> | | | |
| seen | balance policy | Agile baseline | Dex3SynergyHand | <!-- TODO(#20): the metrics with intervals, trials and file name --> | | | |
| unseen | balance policy | own policy v1 | Dex3SynergyHand | <!-- TODO(#20): GAP, no issue schedules the unseen S2 run; fill from it if it is run --> | | | |

<!-- TODO(#19): if the Dex1 fallback was used, replace the Dex3 rows by Dex1GripperHand rows and say why -->

### S3 Place in rack

*What goes here: placement success, collision rate and fall rate for the plate-in-hand start, with the hands that were tested.*

| Split | Base | Lower-body policy | Hand | Placement success rate | Collision rate | Trials | JSON file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seen | fixed pelvis | none | KinematicFollowHand | <!-- TODO(#21): the metrics with intervals, trials and file name --> | | | |
| seen | fixed pelvis | none | Dex3SynergyHand | <!-- TODO(#21): the metrics with intervals, trials and file name --> | | | |
| seen | balance policy | own policy v1 | Dex3SynergyHand | <!-- TODO(#22): GAP, no issue schedules S3 with the balance policy; fill from it if it is run --> | | | |
| unseen | fixed pelvis | none | Dex3SynergyHand | <!-- TODO(#21): GAP, no issue schedules the unseen S3 run; fill from it if it is run --> | | | |

### S4 Full task

*What goes here: task success, grasp success, placement success, collision rate, fall rate and completion time for both lower-body controllers, seen and unseen; the headline numbers of the report are the unseen rows.*

| Split | Lower-body policy | Hand | Task success rate | Grasp success rate | Placement success rate | Collision rate | Fall rate | Completion time | Trials | JSON file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seen | own policy v1 | Dex3SynergyHand | <!-- TODO(#22): the metrics with intervals, trials and file name --> | | | | | | | |
| seen | Agile baseline | Dex3SynergyHand | <!-- TODO(#22): the metrics with intervals, trials and file name --> | | | | | | | |
| unseen | own policy v1 | Dex3SynergyHand | <!-- TODO(#22): GAP, no issue schedules the unseen S4 run; this is the headline row, fill from it once it is run --> | | | | | | | |
| unseen | Agile baseline | Dex3SynergyHand | <!-- TODO(#22): GAP, no issue schedules the unseen S4 run; fill from it if it is run --> | | | | | | | |

## Baselines and ablations

*What goes here: the comparisons of [evaluation.md](evaluation.md) (the baselines section) plus the own-policy versus Agile comparison that every table carries, each with the question it answers, the outcome with intervals and the files behind it; a comparison that was not run says "not run" and why.*

| Comparison | What it isolates | Scenarios | Outcome | JSON files |
| --- | --- | --- | --- | --- |
| Own policy v1 versus Agile baseline | Whether training a lower-body policy here beats the reused one | S0 to S4 | <!-- TODO(#8): outcome with intervals --> | |
| Fixed pelvis versus balance policy active | The cost of real whole-body balance | S1 to S3 | <!-- TODO(#20): outcome with intervals; GAP, S3 and S4 have no paired run in any issue --> | |
| Kinematic attach versus Dex3 friction grasp | Contact physics, separated from planning | S2, S3 | <!-- TODO(#18): outcome with intervals --> | |
| Learned arm policy versus IK arm | The robot-learning claim, only if the stretch goal is reached | S1, S4 | <!-- TODO(#26): GAP, no issue exists for the stretch goal; write "not attempted" unless the operator decides otherwise --> | |
| Seen versus unseen split | Generalization, not memorization | S0 to S4 | <!-- TODO(#22): GAP, the unseen runs are not scheduled; outcome only if they are run --> | |
| With versus without domain randomization in training | Robustness of the policy | S0, S4 | <!-- TODO(#6): GAP, no issue trains the policy without randomization and the result schema has no field to tell it apart --> | |

## Failure analysis

*What goes here: the causes of failed trials as tagged by the evaluator, the histogram per scenario, and for each scenario a discussion of the two most frequent causes with the seeds that show them; a missed target is explained here.*

Every failed trial carries exactly one primary cause. When several occur in one trial the one listed first below wins:

- `fall`: balance was lost.
- `ik_infeasible`: no IK solution within limits.
- `grasp_miss`: the fingers closed and the plate was not held.
- `drop`: the plate was lost after the lift.
- `rack_collision`: the contact force threshold was hit.
- `misplace`: the plate was released outside the slot volume.
- `timeout`: none of the above, the clock ran out.

<!-- TODO(#22): embed the failure histogram after copying outputs/figures/failure_tags.png to a committed folder; it comes from scripts/plot_results.py on the final JSON files -->

| Scenario | Most frequent cause | Second most frequent cause | Explanation | Seeds that show it |
| --- | --- | --- | --- | --- |
| S0 | <!-- TODO(#7): tag from the histogram --> | | | |
| S1 | <!-- TODO(#16): tag from the histogram --> | | | |
| S2 | <!-- TODO(#20): tag from the histogram --> | | | |
| S3 | <!-- TODO(#21): tag from the histogram --> | | | |
| S4 | <!-- TODO(#22): tag from the histogram --> | | | |

## Compute (GPU time)

*What goes here: the wall-clock, throughput and memory of every training run behind a reported result, copied from the training runs table of [experiments.md](experiments.md), and the wall-clock of the evaluation runs from their JSON files.*

| Run | Task | Parallel environments | Iterations | Steps per second | Peak VRAM | Wall-clock | Checkpoint release and SHA256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Windows gate, first size | <!-- TODO(#4): row from the training runs table --> | | | | | | |
| Windows gate, second size | <!-- TODO(#4): row from the training runs table --> | | | | | | |
| Training run of own policy v1 | <!-- TODO(#6): row from the training runs table --> | | | | | | |
| Evaluation runs | <!-- TODO(#22): sum of the wall-clock field over the result files used --> | | | | | | |

## Decisions taken

*What goes here: the decisions of [decisions.md](decisions.md) by title, each a link-sized reminder that a choice was made on purpose; decisions taken after the list below are added by the director.*

- D001: Windows 11 first, Ubuntu 24.04 as a gated fallback
- D002: Isaac Lab 3.0.0 + Isaac Sim 6.2, not the Isaac Lab 2.2 / Isaac Sim 5.x stack
- D003: PhysX through full Isaac Sim, not Kit-less Newton
- D004: DifferentialIK for the arms, not Pink IK
- D005: Train our own lower-body policy; keep NVIDIA's Agile checkpoint as the baseline
- D006: Waist belongs to the balance policy, not to a fixed posture
- D007: Kinematic-follow attach as the grasp baseline, not SurfaceGripper
- D008: Dex3-1 hand with a 15-hour time box, Dex1 gripper as fallback
- D009: Detailed kitchen with a pre-opened dishwasher; cabinets as dressing
- D010: ArtVIP and MolmoSpaces for furniture; no PartNet-Mobility
- D011: ArtVIP `dishwasher_2` as the dishwasher base; MolmoSpaces only as static filler
- D012: Stand-off 0.30 m with a commanded waist lean and the IK point at the finger bases (from the paper analysis in Method, not yet measured)
- D013: Compact dishwasher raised on a plinth, dressed as a low cabinet (from the paper analysis in Method, not yet measured)

<!-- TODO(#26): add one line for each decision recorded after D013 and, for D008, the outcome of the time box -->

## Limitations and honest caveats

*What goes here: everything a careful reader should discount, stated before they have to ask; each item says what was simplified and what the effect on the results could be.*

- **The plate has a rim.** *A paper-thin plate is a known failure case of the physics engine, so the plate has a rim of non-zero thickness; say so and say it is not a real plate.* <!-- TODO(#9): the rim thickness and diameters as built -->
- **The kinematic grasp is not a grasp.** *Results with the attach baseline measure planning, not contact physics, and are labelled that way everywhere.* <!-- TODO(#17): where the baseline rows appear -->
- **The dishwasher is pre-opened and modified.** *The door starts open and the rack starts out; the twin was wrapped with new collision and possibly a scale factor, and the tines are authored.* <!-- TODO(#11): the changes actually made -->
- **Object poses come from simulation state.** *There is no camera perception, so the results say nothing about seeing the plate.* <!-- TODO(#22): confirm that the final observation groups contain no rendered input -->
- **Planner constants are provisional.** *Timeouts and tolerances were tuned on the seen split only; the held-out split is used for reporting and never for tuning.* <!-- TODO(#21): the list of tuned constants and the split used to tune them -->
- **The Windows gate.** *State the outcome of the go or no-go decision and what it cost.* <!-- TODO(#4): the outcome and the numbers behind it -->
- **Trial counts limit what can be concluded.** *Say how wide the intervals are and which differences between rows are therefore not established.* <!-- TODO(#26): the sentence, written after the intervals are known -->
- **Missed targets.** *List each target of the evaluation protocol that was missed; targets were not lowered to fit.* <!-- TODO(#26): the list, from the Targets versus measured table -->
- **Simulation only.** *No claim is made about the real robot; the interfaces for it are reserved and nothing more.* <!-- TODO(#26): one sentence pointing at architecture section 13 -->

## Next steps toward the KitchenBot vision

*What goes here: what comes after this version, in order of effort, starting from the seams already reserved in the architecture (walking, door opening, vision, bimanual work, a learned arm policy, imitation from the scripted planner, a real-robot bridge).*

<!-- TODO(#26): the list, written after the failure analysis so that it follows the observed failures -->

## Reproduction

*What goes here: the exact commands that reproduce the results from a fresh clone, in order, each copied from the setup and evaluation documents and kept only after it has been run.*

Check the host, on Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\host_check.ps1
```

Create the environment and install Isaac Lab, Isaac Sim and this package:

```bat
uv venv --python 3.12 C:\venvs\kitchenbot
C:\venvs\kitchenbot\Scripts\activate
set OMNI_KIT_ACCEPT_EULA=YES
uv pip install "isaaclab[isaacsim]==3.0.0" ^
  --overrides "https://raw.githubusercontent.com/isaac-sim/IsaacLab/v3.0.0/tools/wheel_builder/uv-overrides.txt" ^
  --extra-index-url https://pypi.nvidia.com ^
  --torch-backend cu130 ^
  --index-strategy unsafe-best-match
cd <repo>
uv pip install -e .
```

Verify the installation:

```bat
isaacsim isaacsim.exp.compatibility_check
isaacsim
python scripts/smoke/create_empty.py --viz kit
isaaclab list_envs --show_presets
```

Evaluate, plot and update the results table:

```bat
python scripts/evaluate.py --scenario S4 --split unseen --seeds 1000-1049 --num_envs 50 --headless
python scripts/plot_results.py
python scripts/update_experiments.py
```

<!-- TODO(#26): replay every block above from a fresh clone and delete or correct any line that fails -->
<!-- TODO(#4): the two gate commands (pretrained play and the short training run) exactly as they were run, with the flags confirmed by the help output -->
<!-- TODO(#6): the training command through scripts/train_stand.py and the policy export through scripts/export_policy.py -->
<!-- TODO(#26): the evaluator command for every scenario, split, lower-body policy, hand and base used in the tables, with the checkpoint release URL and SHA256 -->
<!-- TODO(#25): the command that records the demo video through scripts/record_demo.py -->

## Traceability

*What goes here: the index that lets a reader check any number in this report: the result file it came from, the commit that produced that file and the SHA256 of the checkpoint that was evaluated.*

| JSON file | Scenario | Split | Lower-body policy | Hand | Base | Commit | Checkpoint SHA256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <!-- TODO(#26): one row per result file used in this report, from the run block of the file --> | | | | | | | |

<!-- TODO(#26): the sentence that tells the reader how to check a number: find the file in the table, open the matching row of experiments.md, rerun the evaluator command -->

## Acknowledgements and licences

*What goes here: every third-party component with its licence and the attribution it requires; the licences of the downloaded assets are generated into `source/kitchenbot/scene/assets/third_party/MANIFEST.md`, and this table must agree with that file.*

| Component | Used for | Licence | Attribution and source |
| --- | --- | --- | --- |
| ArtVIP | Articulated dishwasher and cabinets | Apache-2.0 (dataset card metadata; no separate LICENSE file) | arXiv 2506.04941, <https://huggingface.co/datasets/x-humanoid-robomind/ArtVIP> |
| MolmoSpaces (Ai2) | Optional static filler furniture | CC-BY-4.0 for data, except the Objaverse subset under ODC-BY-1.0 (not used); code Apache-2.0 | Ai2, MolmoSpaces, arXiv 2602.11337, <https://huggingface.co/datasets/allenai/molmospaces>; the dataset card adds research and educational use wording |
| Unitree G1 robot description (`unitree_ros`) | Joint names and limits from the URDF; the Dex1 variant if the fallback is used | BSD-3-Clause | <https://github.com/unitreerobotics/unitree_ros> |
| NVIDIA Isaac Sim and Isaac assets | Simulator, G1 asset | NVIDIA Omniverse licence, per the README | <!-- TODO(#1): licence text read from the installed package --> |
| NVIDIA Isaac Lab | Environment framework, IK controller, task templates | <!-- TODO(#1): licence read from the installed package --> | <https://github.com/isaac-sim/IsaacLab> |
| RSL-RL | PPO training | <!-- TODO(#1): licence read from the installed package --> | <!-- TODO(#1): citation from the package metadata --> |
| NVIDIA Agile locomotion checkpoint | Baseline lower-body policy | <!-- TODO(#7): licence of the checkpoint file as shipped --> | <!-- TODO(#7): source path in the Isaac asset root --> |
| This repository's code | Everything under `source/kitchenbot` | MIT (see `LICENSE`) | <https://github.com/haianhltr/kitchenbot-ai> |

Design references, ideas and not code: HOMIE (arXiv 2502.13013), AMO (arXiv 2505.03738) and NVIDIA's GR00T decoupled whole-body controller. Nothing from PartNet-Mobility is used.

<!-- TODO(#12): state whether any MolmoSpaces asset is visible in the final report or video; it is optional and dropped if the operator reads its licence wording as restrictive -->
