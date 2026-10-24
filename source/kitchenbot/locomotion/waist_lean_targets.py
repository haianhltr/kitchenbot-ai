"""The math of the waist-lean command used while training the balance policy (#68, for #5 and #6).

D012: the planner leans the torso forward during pre-grasp, grasp and lift and stands upright again
from the counter-edge waypoint on, so the lower-body policy is trained from its first run with a
commanded forward waist pitch (docs/architecture.md section 6, Commands row). The Isaac Lab command
term (``commands.py``, an untested draft) converts tensors at the boundary and calls these
functions; this module is numpy only so that it runs on CI.

Rules, all deterministic given a ``numpy.random.Generator``:

- **Targets.** A new target is drawn every 1 to 3 s (uniform). With probability 0.5 the target is
  exactly 0.0, otherwise it is uniform in ``[0, 0.468]`` rad, positive leaning the torso forward.
  ``MAX_LEAN_RAD`` is ``robots.joint_limits.WAIST_LEAN_MAX_RAD``, the soft limit of
  ``waist_pitch_joint`` (0.9 x its 0.52 rad URDF limit, rounded to 3 decimals; D012 as amended
  2026-10-24, #69 finding E8, #77): a policy is not trained to hold a joint at its hard stop. The
  planner's lean target (#67) and the pitch range of the waist envelope (#77) read the same
  constant. The draw is per resample, not per episode: a 20 s episode holds about ten of them.
- **Ramp, not step.** The command the policy observes (and the reward tracks) moves toward the
  target at ``MAX_RATE_RAD_S`` = 1 rad/s, the rate at which the planner's arm target pipeline slews
  (``controllers/target_filter.py``: 0.3 m/s and 1 rad/s). Why: the deployed command comes from a
  slewed planner and does not jump, so a step would train the policy on a transient it never
  meets, and the tracking reward would charge it for the waist's unavoidable lag after a jump (the
  joint is a PD loop, it cannot arrive in one 20 ms step: a 0.468 rad error asks 2,340 N·m of the
  waist actuator, stiffness 5000 N·m/rad, whose limit is 50 N·m, ``unitree.py:506-531``). A full
  0.468 rad lean takes 0.468 s (24 steps of 20 ms), well inside the 1 s minimum between targets,
  so every ramp has arrived before the next target.
  ``max_rate_rad_s=None`` in the config gives the step, for an ablation in #5. The planner
  contract that follows: #21 slews the lean command at no more than 1 rad/s; a commanded jump is
  outside the training distribution.
- **Episode start.** The command starts every episode at 0.0 (the reset puts the waist at its
  default pose, zero) and ramps to the first target.
- **Tracking error.** ``tracking_error`` is the reward's quantity: the pitch error to the command
  plus the absolute yaw and roll, which the policy should keep at zero.

``slew_toward`` and ``tracking_error`` use only arithmetic, ``abs`` and ``.clip``, so they also
accept torch tensors. ``simulate_command`` replays the command term on a numpy clock (Isaac Lab's
``CommandTerm.compute``, v3.0.0 ``managers/command_manager.py:146-161``) so the statistics can be
tested without a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kitchenbot.locomotion.upper_body_targets import sample_durations
from kitchenbot.robots.joint_limits import WAIST_LEAN_MAX_RAD
from kitchenbot.robots.joint_maps import WAIST

#: The joint the command drives.
WAIST_PITCH_JOINT = "waist_pitch_joint"

#: Positions of the three waist joints inside ``joint_maps.WAIST`` (yaw, roll, pitch).
WAIST_YAW_INDEX = WAIST.index("waist_yaw_joint")
WAIST_ROLL_INDEX = WAIST.index("waist_roll_joint")
WAIST_PITCH_INDEX = WAIST.index(WAIST_PITCH_JOINT)

#: Largest commanded forward pitch in rad (D012 as amended): the soft limit of the joint,
#: ``robots.joint_limits.WAIST_LEAN_MAX_RAD``, 0.468 rad = 26.8 degrees.
MAX_LEAN_RAD = WAIST_LEAN_MAX_RAD

#: Probability that a resampled target is exactly zero (D012: the planner stands upright half the
#: time).
ZERO_SHARE = 0.5

#: Seconds between two targets (docs/architecture.md section 6).
RESAMPLE_TIME_RANGE_S: tuple[float, float] = (1.0, 3.0)

#: Rate in rad/s at which the observed command moves toward its target: the planner's slew rate.
MAX_RATE_RAD_S = 1.0

#: Second word of the command's seed sequence, ``default_rng([seed, RNG_STREAM])``, so its draws
#: are independent of the upper-body command's, which uses ``default_rng(seed)``.
RNG_STREAM = 68


def sample_lean_targets(
    rng: np.random.Generator,
    num: int,
    max_lean: float = MAX_LEAN_RAD,
    zero_share: float = ZERO_SHARE,
) -> np.ndarray:
    """``num`` lean targets in rad, shape ``(num,)``: exactly 0.0 with probability ``zero_share``,
    otherwise uniform in ``[0, max_lean]``."""
    if max_lean < 0.0:
        raise ValueError(f"max_lean must be >= 0, got {max_lean}")
    if not 0.0 <= zero_share <= 1.0:
        raise ValueError(f"zero_share must be in [0, 1], got {zero_share}")
    lean = rng.uniform(0.0, max_lean, size=num)
    return np.where(rng.random(num) < zero_share, 0.0, lean)


def slew_toward(command, target, max_step):
    """``command`` moved toward ``target`` by at most ``max_step`` (a non-negative step in rad).

    Arrives exactly when the distance is within ``max_step``. Works on numpy arrays and torch
    tensors alike.
    """
    return command + (target - command).clip(-max_step, max_step)


def tracking_error(pitch, yaw, roll, command):
    """``|pitch - command| + |yaw| + |roll|`` per environment, in rad.

    ``pitch``, ``yaw`` and ``roll`` are the waist joint positions (not relative to a default; the
    waist default is zero), ``command`` the lean command. Works on numpy arrays and torch tensors.
    """
    return abs(pitch - command) + abs(yaw) + abs(roll)


@dataclass(frozen=True)
class CommandTrace:
    """Per-step history of ``simulate_command``, each array of shape ``(num_steps, num_envs)``."""

    command: np.ndarray
    target: np.ndarray
    #: ``True`` at the steps where the target was redrawn.
    resampled: np.ndarray
    dt: float


def simulate_command(
    rng: np.random.Generator,
    num_envs: int,
    num_steps: int,
    dt: float,
    *,
    time_range: tuple[float, float] = RESAMPLE_TIME_RANGE_S,
    max_lean: float = MAX_LEAN_RAD,
    zero_share: float = ZERO_SHARE,
    max_rate: float | None = MAX_RATE_RAD_S,
) -> CommandTrace:
    """Replay the command term for ``num_steps`` environment steps of ``dt`` seconds.

    Same order as the term: at the episode start a first interval and target are drawn and the
    command is 0.0 (``max_rate=None``: the target); at each step the time left drops by ``dt``,
    targets whose time ran out are redrawn together with a new interval (which replaces the time
    left, as ``CommandTerm._resample`` does, ``command_manager.py:179``), then the command moves
    toward its target.
    """
    if num_envs < 1 or num_steps < 1 or dt <= 0.0:
        raise ValueError("need num_envs >= 1, num_steps >= 1 and dt > 0")
    time_left = sample_durations(rng, num_envs, time_range)
    target = sample_lean_targets(rng, num_envs, max_lean, zero_share)
    command = target.copy() if max_rate is None else np.zeros(num_envs)
    commands, targets, due_steps = [], [], []
    for _ in range(num_steps):
        time_left = time_left - dt
        due = time_left <= 0.0
        count = int(due.sum())
        if count:
            time_left[due] = sample_durations(rng, count, time_range)
            target[due] = sample_lean_targets(rng, count, max_lean, zero_share)
        command = target.copy() if max_rate is None else slew_toward(command, target, max_rate * dt)
        commands.append(command)
        targets.append(target.copy())
        due_steps.append(due)
    return CommandTrace(np.array(commands), np.array(targets), np.array(due_steps), dt)
