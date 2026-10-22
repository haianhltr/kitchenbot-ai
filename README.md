# KitchenBot AI — humanoid kitchen manipulation in simulation

A simulated Unitree G1 humanoid that stands on a reinforcement-learning balance policy, grasps a
plate from a kitchen counter with a Dex3-1 hand and places it into an articulated dishwasher rack.
Built on Isaac Lab 3.0 and Isaac Sim 6.2 with PhysX. Real physics, a real controller, no keyframes.

**Status: planning complete, no code yet.** Nothing in this repository has been measured. Every
number in `docs/evaluation.md` is a target, not a result. Results will appear in
`docs/experiments.md` with date, commit, seeds and the exact command that produced them.

## What the MVP does

1. The robot stands in a detailed kitchen on a lower-body policy we train ourselves, whose training
   randomizes arm motion so that reaching does not cause falls.
2. A classical inverse-kinematics controller moves the right arm to a plate at a known pose.
3. A three-finger Dex3-1 hand grasps the plate by friction. A kinematic-attach baseline exists for
   comparison and is always labelled as such.
4. A finite-state planner lifts, carries, aligns and inserts the plate into a free slot of the
   dishwasher's lower rack, then releases and retreats.
5. An evaluator runs the task over randomized seen and held-out configurations and reports success
   rate, fall rate, collision rate and completion time with confidence intervals.

Walking, opening the dishwasher door, camera perception and bimanual work are deliberately out of
scope for the MVP. They are later stages of the KitchenBot vision: identify dirty dishes, rinse,
load and unload a dishwasher, dry with a towel, put dishes away, wipe counters, all as one
long-horizon routine.

## How the hard part is solved

Every one of the robot's 43 joints has exactly one writer, so locomotion and manipulation never
fight over a joint:

| Joints | Controller |
| --- | --- |
| 12 leg + 3 waist | lower-body RL policy, trained here, frozen at deployment |
| 14 arm | DifferentialIK (right arm task-driven, left arm held) |
| 14 hand | Dex3 synergy controller (open, pre-grasp, close) |

The full design, including the training environment, rewards, observation layout, scene assets and
sim-to-real seams, is in [docs/architecture.md](docs/architecture.md).

## Reused versus developed

| Reused | Developed here |
| --- | --- |
| Isaac Sim 6.2, PhysX, the G1 + Dex3 USD asset, Isaac Lab 3.0 framework, `DifferentialIKController`, RSL-RL PPO, NVIDIA's Agile locomotion checkpoint as a baseline, ArtVIP and MolmoSpaces furniture | the standing environment with upper-body randomization and its trained policy, the lower-body policy action term, the upper-body follow action term, hand controllers, plate asset, dishwasher rack geometry, kitchen scene assembly, task planner, evaluator, plots, report |

Design references (ideas, not code): HOMIE and AMO (RSS 2025) and NVIDIA's GR00T decoupled
whole-body controller.

## Repository map

| Path | Purpose |
| --- | --- |
| `docs/architecture.md` | technical design |
| `docs/roadmap.md` | milestones M0 to M6, the issue plan, week plan, critical path |
| `docs/setup.md` | pinned toolchain and install steps for Windows 11, with the Ubuntu fallback gate |
| `docs/evaluation.md` | scenarios, metrics, protocol, targets, failure taxonomy |
| `docs/experiments.md` | measured results only |
| `docs/decisions.md` | decision log |
| `docs/status.md` | current state, rewritten by the director after every merge |
| `AGENTS.md` | how this repository is built: a team of Claude Code sessions coordinated through GitHub issues, plus one human operator |
| `source/kitchenbot/` | the Python package |
| `scripts/`, `configs/`, `tests/` | entry points, pinned run configs, unit and sim tests |

## Getting started

Follow [docs/setup.md](docs/setup.md). Host requirements: Windows 11 or Ubuntu 24.04, an NVIDIA GPU
with 16 GB VRAM or more, 32 GB RAM, Python 3.12.

## How this project is built

This repository is developed by several Claude Code sessions, each with a fixed role, coordinating
only through GitHub issues and pull requests, with a human operator making scope decisions and
reviewing. The protocol is in [AGENTS.md](AGENTS.md). The point is a reproducible engineering
artefact, so every claim must trace to a committed config, a logged run and a script that printed it.

## Licence

Code: MIT (see `LICENSE`). Third-party assets keep their own licences: NVIDIA Isaac assets under the
NVIDIA Omniverse licence, Unitree robot descriptions under BSD-3-Clause, ArtVIP under Apache-2.0,
MolmoSpaces data under CC BY 4.0. Nothing from PartNet-Mobility is used.
