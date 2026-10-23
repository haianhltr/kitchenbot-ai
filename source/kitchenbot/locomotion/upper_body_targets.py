"""The math of the upper-body command used while training the balance policy (#46, for #5).

During training the arms and hands do not belong to the agent. A command term samples joint
targets for them and an action term with zero agent dimensions applies those targets, so the
balance policy learns to stand while the upper body moves (docs/architecture.md sections 5 and 6;
HOMIE, RSS 2025). This module is that sampler, written with numpy only so it runs on CI; the
Isaac Lab command term (``commands.py``, an untested draft) converts tensors at the boundary and
calls these functions.

Rules, all deterministic given a ``numpy.random.Generator``:

- **Random targets.** Each target is drawn uniformly from ``[lower, upper]`` shrunk about the
  default pose by the range factor ``f``: ``[d + f * (lower - d), d + f * (upper - d)]``, then
  clipped to ``[lower, upper]``. ``f = 1`` is the whole joint range; ``f = 0.2`` is a fifth of it
  around the default pose.
- **Timing.** A new target is drawn every 1 to 3 s (uniform); the command moves linearly from the
  previous target to the new one over exactly that interval.
- **Reach episodes.** 20 percent of episodes ignore the random targets: the right arm moves slowly
  (over 3 to 5 s) from its start pose to one target inside a front-workspace box and holds it; all
  other upper-body joints hold their default. The box is in joint space and **provisional**: the
  sign conventions behind it (negative shoulder pitch swings the arm forward, negative right
  shoulder roll swings it outward) are read from the URDF joint axes, not checked in simulation.
- **Curriculum.** The range factor starts at 0.2 and moves toward 1.0 in fixed steps while the
  smoothed fall rate stays low, and back toward 0.2 when it is high.

Joint order is ``UPPER_BODY_JOINTS``: left arm, right arm, left hand, right hand, each in
``robots.joint_maps`` order. Array arguments are ``float`` arrays of shape ``(J,)`` for per-joint
values and ``(N, J)`` for per-environment values. ``interpolate`` uses only arithmetic and
``.clip``, so it also accepts torch tensors.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from kitchenbot.locomotion.obs_layout import UPPER_BODY_JOINTS
from kitchenbot.robots.joint_maps import ARM_LEFT, ARM_RIGHT

# UPPER_BODY_JOINTS (from obs_layout) is the command order: the 28 arm and hand joints.

#: Position of the right arm's seven joints inside ``UPPER_BODY_JOINTS``.
RIGHT_ARM_SLICE = slice(len(ARM_LEFT), len(ARM_LEFT) + len(ARM_RIGHT))

#: Seconds between two random targets (docs/architecture.md section 6).
RESAMPLE_TIME_RANGE_S: tuple[float, float] = (1.0, 3.0)

#: Fraction of episodes that run the right-arm reach instead of random targets.
REACH_EPISODE_FRACTION = 0.2

#: Seconds the slow reach takes to arrive at its target. Provisional.
REACH_DURATION_RANGE_S: tuple[float, float] = (3.0, 5.0)

#: Front-workspace box for the right arm in joint space (rad). Provisional, see the module
#: docstring; #5 checks in simulation that these poses put the hand in front of the torso.
REACH_BOX: dict[str, tuple[float, float]] = {
    "right_shoulder_pitch_joint": (-1.4, -0.3),
    "right_shoulder_roll_joint": (-0.4, 0.0),
    "right_shoulder_yaw_joint": (-0.3, 0.3),
    "right_elbow_joint": (-0.3, 0.8),
    "right_wrist_roll_joint": (-0.5, 0.5),
    "right_wrist_pitch_joint": (-0.4, 0.4),
    "right_wrist_yaw_joint": (-0.4, 0.4),
}

#: Range factor at the start and at the end of the curriculum.
RANGE_FACTOR_START = 0.2
RANGE_FACTOR_END = 1.0

#: Curriculum rule parameters. Provisional; #5 tunes them from the first training curves.
RANGE_FACTOR_STEP = 0.05
PROMOTE_BELOW_FALL_RATE = 0.10
DEMOTE_ABOVE_FALL_RATE = 0.30
FALL_RATE_SMOOTHING = 0.5


def _check_limits(lower: np.ndarray, upper: np.ndarray, default: np.ndarray) -> None:
    if not (lower.shape == upper.shape == default.shape and lower.ndim == 1):
        raise ValueError("lower, upper and default must be 1-D arrays of the same length")
    if np.any(lower > upper):
        raise ValueError("every lower limit must be <= its upper limit")


def sampling_bounds(
    lower: np.ndarray, upper: np.ndarray, default: np.ndarray, range_factor: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint ``(low, high)`` of the uniform target distribution for ``range_factor``.

    The default pose is first clipped into ``[lower, upper]``, so the interval always lies inside
    the limits and contains the (clipped) default.
    """
    lower, upper, default = (np.asarray(a, dtype=float) for a in (lower, upper, default))
    _check_limits(lower, upper, default)
    if not 0.0 <= range_factor <= 1.0:
        raise ValueError(f"range_factor must be in [0, 1], got {range_factor}")
    centre = np.clip(default, lower, upper)
    return centre + range_factor * (lower - centre), centre + range_factor * (upper - centre)


def sample_targets(
    lower: np.ndarray,
    upper: np.ndarray,
    default: np.ndarray,
    range_factor: float,
    rng: np.random.Generator,
    num: int,
) -> np.ndarray:
    """``num`` random upper-body targets, shape ``(num, J)``, each inside ``[lower, upper]``."""
    low, high = sampling_bounds(lower, upper, default, range_factor)
    targets = rng.uniform(low, high, size=(num, low.size))
    return np.clip(targets, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))


def sample_durations(
    rng: np.random.Generator, num: int, time_range: tuple[float, float] = RESAMPLE_TIME_RANGE_S
) -> np.ndarray:
    """``num`` durations in seconds, uniform in ``time_range``."""
    low, high = time_range
    if not 0.0 < low <= high:
        raise ValueError(f"time_range must satisfy 0 < low <= high, got {time_range}")
    return rng.uniform(low, high, size=num)


def interpolate(previous, target, elapsed, duration):
    """Linear interpolation from ``previous`` to ``target`` after ``elapsed`` of ``duration``.

    ``previous`` and ``target`` have shape ``(N, J)``; ``elapsed`` and ``duration`` have shape
    ``(N,)``. The fraction is clipped to ``[0, 1]``: before the start the result is ``previous``,
    after the end it holds ``target``. Works on numpy arrays and torch tensors alike.
    """
    alpha = (elapsed / duration).clip(0.0, 1.0)[..., None]
    return previous + alpha * (target - previous)


def sample_reach_episodes(
    rng: np.random.Generator, num: int, fraction: float = REACH_EPISODE_FRACTION
) -> np.ndarray:
    """Boolean mask of length ``num``: ``True`` where the episode runs the right-arm reach."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    return rng.random(num) < fraction


def reach_box_arrays(
    joint_names: tuple[str, ...] = UPPER_BODY_JOINTS,
    box: Mapping[str, tuple[float, float]] = REACH_BOX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mask, low, high)`` over ``joint_names``: which joints the reach moves and their box."""
    mask = np.array([name in box for name in joint_names])
    low = np.array([box.get(name, (0.0, 0.0))[0] for name in joint_names], dtype=float)
    high = np.array([box.get(name, (0.0, 0.0))[1] for name in joint_names], dtype=float)
    return mask, low, high


def sample_reach_targets(
    lower: np.ndarray,
    upper: np.ndarray,
    default: np.ndarray,
    rng: np.random.Generator,
    num: int,
    joint_names: tuple[str, ...] = UPPER_BODY_JOINTS,
    box: Mapping[str, tuple[float, float]] = REACH_BOX,
) -> np.ndarray:
    """``num`` reach targets, shape ``(num, J)``.

    Joints named in ``box`` get a uniform value in the box intersected with ``[lower, upper]``;
    every other joint holds its default (clipped into its limits).
    """
    lower, upper, default = (np.asarray(a, dtype=float) for a in (lower, upper, default))
    _check_limits(lower, upper, default)
    if len(joint_names) != lower.size:
        raise ValueError("joint_names must name every joint of lower/upper/default")
    mask, box_low, box_high = reach_box_arrays(joint_names, box)
    low = np.where(mask, np.maximum(box_low, lower), lower)
    high = np.where(mask, np.minimum(box_high, upper), upper)
    if np.any(low[mask] > high[mask]):
        raise ValueError("the reach box does not intersect the joint limits")
    held = np.clip(default, lower, upper)
    targets = np.broadcast_to(held, (num, lower.size)).copy()
    targets[:, mask] = rng.uniform(low[mask], high[mask], size=(num, int(mask.sum())))
    return targets


def update_fall_rate(
    previous: float, num_fell: int, num_ended: int, smoothing: float = FALL_RATE_SMOOTHING
) -> float:
    """Exponentially smoothed fall rate after a batch of ``num_ended`` finished episodes.

    ``num_fell`` of them ended by falling (any termination that is not a time-out). A batch with
    no finished episode leaves the rate unchanged.
    """
    if not 0 <= num_fell <= num_ended:
        raise ValueError("need 0 <= num_fell <= num_ended")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError(f"smoothing must be in (0, 1], got {smoothing}")
    if num_ended == 0:
        return previous
    return (1.0 - smoothing) * previous + smoothing * (num_fell / num_ended)


def next_range_factor(
    current: float,
    fall_rate: float,
    *,
    step: float = RANGE_FACTOR_STEP,
    promote_below: float = PROMOTE_BELOW_FALL_RATE,
    demote_above: float = DEMOTE_ABOVE_FALL_RATE,
    start: float = RANGE_FACTOR_START,
    end: float = RANGE_FACTOR_END,
) -> float:
    """The curriculum rule: one ``step`` toward ``end`` when ``fall_rate < promote_below``, one
    step back toward ``start`` when ``fall_rate > demote_above``, unchanged in between. The result
    always lies in ``[start, end]``."""
    if not start <= end:
        raise ValueError("start must be <= end")
    if not promote_below <= demote_above:
        raise ValueError("promote_below must be <= demote_above")
    if fall_rate < promote_below:
        current += step
    elif fall_rate > demote_above:
        current -= step
    return float(min(max(current, start), end))
