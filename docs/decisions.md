# Decision log

One entry per decision that changes the architecture, the scope or the toolchain. Newest last.
Format: context, decision, consequences. Dates are the day the decision was taken.

## D001 — Windows 11 first, Ubuntu 24.04 as a gated fallback (2026-10-22)

Context: Isaac Lab 3.0 documents Windows support as incomplete and recommends Ubuntu 24.04; the
operator works on Windows and would lose the Windows desktop while booted into Ubuntu.
Decision: run M0 on Windows with an explicit go/no-go gate (`docs/setup.md` section 5). On NO-GO,
install Ubuntu 24.04 alongside Windows.
Consequences: Pink IK, Isaac Lab Mimic, XR teleop, Ray and multi-GPU are unavailable (Linux-only);
the architecture avoids all of them.

## D002 — Isaac Lab 3.0.0 + Isaac Sim 6.2, not the Isaac Lab 2.2 / Isaac Sim 5.x stack (2026-10-22)

Context: Unitree's own Isaac Lab repository targets 2.x / 5.x, which upstream no longer supports;
Isaac Sim 4.5 does not run on Blackwell GPUs at all.
Decision: build on the supported 3.0 / 6.2 stack and port what is needed.
Consequences: Unitree's pick-and-place scenes are references, not dependencies.

## D003 — PhysX through full Isaac Sim, not Kit-less Newton (2026-10-22)

Context: Isaac Lab 3.0 defaults to the Newton MJWarp backend; grasping and USD kitchen scenes need
mature contacts and the full Isaac Sim toolset.
Decision: `physics=isaacsim_physx` on every command and config, asserted by a unit test.
Consequences: higher VRAM per environment than Newton; conservative env counts.

## D004 — DifferentialIK for the arms, not Pink IK (2026-10-22)

Context: Isaac Lab's humanoid manipulation examples use Pink, whose dependencies have no Windows
wheels and are declared Linux-only in 3.0.
Decision: `DifferentialIKController` with damped least squares; `OperationalSpaceController` as the
documented fallback if posture control is needed.
Consequences: no null-space posture task; elbow drift is a tracked risk.

## D005 — Train our own lower-body policy; keep NVIDIA's Agile checkpoint as the baseline (2026-10-22)

Context: Isaac Lab 3.0 ships a pretrained Agile lower-body policy inside a contrib loco-manipulation
task; the operator wants to retrain reused components and the career focus is robot learning.
Decision: train a 15-joint (legs + waist) standing policy with randomized upper-body motion; compare
against Agile in every table.
Consequences: M1 carries the GPU time; the comparison is the project's first robot-learning claim.

## D006 — Waist belongs to the balance policy, not to a fixed posture (2026-10-22)

Context: AMO and GR00T's decoupled controller both give the waist to the lower-body policy so the
torso can counter-lean during reaches.
Decision: 15 actions; a reward keeps the waist near zero unless commanded.
Consequences: the arm controller never commands the waist; later waist use is a policy change.

## D007 — Kinematic-follow attach as the grasp baseline, not SurfaceGripper (2026-10-22)

Context: Isaac Lab's official attach mechanism, `SurfaceGripper`, requires CPU simulation in 3.0.
Decision: implement a labelled kinematic-follow baseline on GPU PhysX; use SurfaceGripper only for
cross-checks.
Consequences: the baseline is explicitly not a grasp and is reported as such.

## D008 — Dex3-1 hand with a 15-hour time box, Dex1 gripper as fallback (2026-10-22)

Context: thin plates are a known PhysX failure case; Unitree publishes a 29-DoF G1 URDF with its
own two-jaw Dex1 gripper.
Decision: rimmed plate plus contact-confirmed closing on Dex3; switch to Dex1 if S2 is below target
after 15 hours of effort on P18.
Consequences: a conditional issue (P19) exists and may never be opened.

## D009 — Detailed kitchen with a pre-opened dishwasher; cabinets as dressing (2026-10-22)

Context: the operator chose the detailed kitchen over the simplified one; the budget is 150 hours.
Decision: the dishwasher is articulated and starts with the door open and the rack out; cabinets are
articulated assets that nothing touches in the MVP.
Consequences: door opening is a post-MVP task; about 13 extra hours in M2 relative to a simple scene.

## D010 — ArtVIP and MolmoSpaces for furniture; no PartNet-Mobility (2026-10-22)

Context: PartNet-Mobility is non-commercial, gated, and may refuse non-institutional e-mail; ArtVIP
is Apache-2.0 USD and MolmoSpaces is CC BY 4.0 with Isaac exports.
Decision: ArtVIP first, MolmoSpaces second, authored USD as the guaranteed fallback; rack tines are
authored in every case.
Consequences: the portfolio carries no non-commercial asset licence.

## D011 — ArtVIP `dishwasher_2` as the dishwasher base; MolmoSpaces only as static filler (2026-10-23)

Context: #29 inspected seven ArtVIP dishwashers and the MolmoSpaces THOR set with `usd-core`, before
Isaac Sim was installed. `dishwasher_2` (7 MB, 52k points) has a bottom-hinged door (0 to 90
degrees) and two prismatic racks with 0.2 m travel; every ArtVIP dishwasher uses `sdf` collision
and has no tines. MolmoSpaces has no dishwasher, its THOR cabinets are static, and its dataset card
adds "research and educational use" wording on top of CC BY 4.0.
Decision: #11 wraps `dishwasher_2` (alternate `dishwasher_4`), replaces its collision with convex
shapes, authors the tines, and scales the unit uniformly if a 25 cm plate does not fit; "rack out"
means its full 0.2 m travel, not the 0.4 m first assumed. MolmoSpaces assets are optional static
filler, dropped if the operator reads the licence wording as restrictive.
Consequences: #11 is a wrapping job (7 hours) rather than authoring from scratch; architecture
section 10a and issue #11 now say 0.2 m; one licence line for the operator to read before the demo
video uses a MolmoSpaces asset.
