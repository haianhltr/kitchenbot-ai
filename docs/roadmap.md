# Roadmap

Budget: 150 engineering hours at about 20 hours per week, so eight weeks from the first commit.
Hours below are estimates made before any code existed; `docs/status.md` tracks actuals. GPU time is
reported per run in `docs/experiments.md` and is not estimated here.

## Milestones

| Milestone | Owner | Hours | Exit criterion |
| --- | --- | --- | --- |
| M0 Environment | kb-sim | 15 | Windows go/no-go gate passed or Ubuntu fallback decided; G1 Dex3 joint map committed; CI green |
| M1 Standing | kb-controls | 27 | Own lower-body policy meets S0 targets under pushes and scripted arm sweeps; Agile baseline measured alongside |
| M2 Kitchen | kb-sim | 30 | Full kitchen loads with robot and plate; dishwasher door and rack articulate; randomization terms exist |
| M3 Reaching | kb-controls | 22 | S1 targets met with balance policy active, seen and unseen |
| M4 Grasping | kb-manip | 31 | S2 targets met with Dex3 or the Dex1 fallback decision recorded |
| M5 Placement | kb-manip | 16 | S4 runs end to end with automatic success detection |
| M6 Evaluation | kb-eval | 19 | All deliverables in `docs/evaluation.md` section 8 checked |
| **Total** | | **160** | no unallocated buffer; actuals in `docs/status.md` are running far under estimate |

Stretch, outside the 150 hours: **M3b** learned right-arm policy replacing DifferentialIK behind the
same interface, then imitation learning from the scripted planner.

## Issue plan

Planning IDs map one to one onto GitHub issue numbers: P01 is #1, P26 is #26, and the pinned GPU
queue is #27 (P27 is unused). P28, P29, P30, P35, P40, P42, P43, P46 to P49 and P52 were added
on 2026-10-23 as no-simulation work that ran while the Isaac Sim install waited on the operator's
reboot (P31 to P34, P36 to P39, P41, P44, P45, P50, P51 and P53 are unused: those numbers went to
pull requests). `CP` marks the critical path, labelled `milestone-blocker`. "After" lists hard
dependencies, mirrored as "Blocked by" in each issue.

| ID | Milestone | Owner | Title | Hours | After | Flags |
| --- | --- | --- | --- | --- | --- | --- |
| P01 | M0 | kb-sim | Install Isaac Lab 3.0.0 + Isaac Sim 6.2 on Windows with uv; smoke steps 1 to 4 of `docs/setup.md` | 4 | — | CP |
| P02 | M0 | kb-sim | Repository scaffolding: pyproject, CI (lint, unit tests, 5 MB file guard), pytest markers | 2 | — | |
| P03 | M0 | kb-sim | Load G1 Dex3 asset; dump joint and link names; diff against the P28 map and correct `joint_maps.py`; fixed and floating base spawn tests | 2 | P01, P28 | CP |
| P04 | M0 | kb-sim | Windows go/no-go gate: pretrained play and 100-iteration training run; record FPS and VRAM; decision comment | 3 | P01 | CP, gpu |
| P28 | M0 | kb-sim | Provisional G1 Dex3 joint map and limits from the Unitree URDF; partition test; no simulation | 2 | — | |
| P35 | M0 | kb-sim | Windows host check script with `-Fix` for long paths, git option and uv; no simulation | 2 | — | |
| P05 | M1 | kb-controls | Stand environment: run and fix the P46 drafts in simulation; zero-agent and random-agent smoke | 4 | P03, P46 | CP |
| P06 | M1 | kb-controls | Train lower-body policy v1; export TorchScript and joint manifest; S0 metrics | 8 | P05, P04 | CP, gpu |
| P07 | M1 | kb-controls | Lower-body policy action-term wrapper; Agile checkpoint baseline wrapper; S0 comparison under pushes | 6 | P06 | CP |
| P08 | M1 | kb-controls | Robustness test: scripted arm sweeps while standing; fall rate for own policy and Agile | 5 | P07 | |
| P46 | M1 | kb-controls | Stand environment drafts and the Isaac Lab 3.0 API notes, written against the v3.0.0 source; no simulation | 4 | — | |
| P09 | M2 | kb-sim | Plate asset: drop test of the P30 meshes, `plate_cfg.py` factory, YCB check | 2 | P03, P30 | |
| P10 | M2 | kb-sim | Scene config loads the P47 counter and floor layers with the robot | 3 | P03, P47 | |
| P11 | M2 | kb-sim | Articulated dishwasher: wrap the ArtVIP twin P29 recommends, else author USD; door and rack joints, attach the P47 tines to the rack link, door open and rack out by default | 6 | P10, P29, P47 | |
| P12 | M2 | kb-sim | Articulated cabinets from ArtVIP or MolmoSpaces as scene dressing; full kitchen composition; load time and FPS check | 5 | P11 | |
| P13 | M2 | kb-sim | Scene randomization event terms wrapping the P52 sampler | 2 | P09, P11, P52 | |
| P29 | M2 | kb-sim | Asset fetch script, licence manifest, `usd-core` inspection of ArtVIP and MolmoSpaces candidates; no simulation | 3 | — | |
| P30 | M2 | kb-sim | Procedural rimmed plate USD assets, 22, 25 and 27 cm, watertight check; no simulation | 2 | — | |
| P47 | M2 | kb-sim | Rack tines, counter and floor as procedural USD; kitchen dimensions and slot poses in one module; no simulation | 4 | P29, P30 | |
| P52 | M2 | kb-sim | Seeded randomization sampler for `docs/evaluation.md` section 2 and the ten eval YAMLs; no simulation | 3 | P23, P47 | |
| P14 | M3 | kb-controls | Right-arm DifferentialIK term on the P48 pipeline, left-arm posture term, waist hold; fixed pelvis; S1 fixed-base | 5 | P03, P48 | |
| P15 | M3 | kb-controls | Composite environment: lower-body policy plus arm IK plus hand term; pelvis-frame target transform; S1 with balance | 8 | P07, P14 | CP |
| P16 | M3 | kb-controls | Validate the P48 envelope against the real IK; `ik_infeasible` tagging; S1 seen and unseen | 5 | P15, P48 | |
| P48 | M3 | kb-controls | Right-arm target slew filter, URDF forward kinematics, sampled reachability envelope, pelvis-frame transform; no simulation | 4 | P28, P42, P43 | |
| P17 | M4 | kb-manip | KinematicFollowHand on the P42 interface; S2 fixed-base baseline | 5 | P09, P14, P42 | |
| P18 | M4 | kb-manip | Dex3 synergy postures, fingertip contact sensors, grasp tuning; S2 fixed-base; 15-hour time box with fallback decision | 15 | P17 | |
| P19 | M4 | kb-manip | Conditional: Dex1 gripper fallback via URDF conversion and gripper controller | 6 | P18 | only if P18 misses target |
| P20 | M4 | kb-manip | Grasp with balance active; S2 with policy | 3 | P18, P15 | CP |
| P42 | M4 | kb-manip | HandController interface, Dex3 posture tables and synergy hand core from the URDF limits; no simulation | 2 | P28 | |
| P21 | M5 | kb-manip | Plate-to-rack FSM in simulation on the P43 core: slot constants, rack force sensor; S3 fixed-base | 6 | P11, P17, P43 | |
| P22 | M5 | kb-manip | Success detection, rack collision force tagging; full S4 with balance | 7 | P21, P20 | CP |
| P43 | M5 | kb-manip | Plate-to-rack planner core: states, waypoints, timeouts, failure tags against a fake arm and hand; no simulation | 3 | — | |
| P23 | M6 | kb-eval | Evaluator runner, JSON schema, seed splits, failure tags | 6 | — | usable from M3; unblocked 2026-10-23, CI already exists |
| P24 | M6 | kb-eval | Plot script and `docs/experiments.md` generation | 3 | P23 | |
| P40 | M6 | kb-eval | Experiments table rows for continuous metrics: mean, median, bootstrap interval | 1 | P24 | |
| P25 | M6 | kb-eval | Demo video recording with on-screen controller labels | 3 | P22 | CP |
| P26 | M6 | kb-eval | Technical report on the P49 skeleton, README results, reused versus developed table, failure analysis | 5 | P24, P25, P49 | CP |
| P49 | M6 | kb-eval | Technical report skeleton and demo video script with every slot bound to an issue; no simulation | 1 | — | |

Plus one pinned issue, **GPU queue**, which is a lock, not work.

Hours by owner: kb-sim 45, kb-controls 49, kb-manip 47 (41 without the conditional P19), kb-eval 19.

## Week plan (20 hours per week)

| Week | Focus | Issues |
| --- | --- | --- |
| 1 | Environment gate, scaffolding, asset load; no-simulation work while the install waits | P01 P02 P03 P04 P28 P29 P30 P35, P23 P24 P40, P42 P43 P46 P47 P48 P49 P52, start P05 |
| 2 | First policy training; plate and counter | P05 P06, P09 P10 |
| 3 | Policy wrapper and robustness; dishwasher; arm IK fixed-base; evaluator | P07 P08, P11, P14, P23 |
| 4 | Composite env with balance; cabinets; kinematic grasp | P15, P12 P13, P17 |
| 5 | Reach metrics; Dex3 grasp begins; placement FSM fixed-base; plots | P16, P18, P21, P24 |
| 6 | Dex3 decision; grasp and placement with balance | P18 or P19, P20, P22 |
| 7 | Full S4 runs, video, buffer | P22, P25, buffer |
| 8 | Report, README, release artifacts, buffer | P26, buffer |

Parallelism is real only across owners: kb-controls on P05 to P08 while kb-sim builds P09 to P13.
Training runs serialize on the single GPU through the GPU queue; week 2 is when kb-controls holds
the GPU and kb-sim does CPU-only asset work.

## Critical path

P01 → P03 → P05 → P06 → P07 → P15 → P20 → P22 → P25 → P26, 50 hours after the
2026-10-23 prework. Any slip here moves the
end date; slips elsewhere consume the buffer first.

## Decision points

| When | Decision | Owner |
| --- | --- | --- |
| End of P04 | Windows GO or Ubuntu NO-GO | operator, on kb-sim's evidence |
| End of P08 | Own policy or Agile baseline as the MVP's lower-body controller, by S0 numbers | director |
| 15 hours into P18 | Dex3 or Dex1 fallback | director, recorded in `docs/decisions.md` |
| End of week 6 | Attempt M3b stretch or spend remaining hours on robustness | operator |
