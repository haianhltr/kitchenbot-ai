"""Seeded domain randomization: the implementation of docs/evaluation.md section 2 (#52).

Given a seed, a split and a scenario, every episode's plate pose, plate diameter, rack slot, pelvis
noise, friction and push schedule is one deterministic draw, :func:`sample_episode`. The Isaac Lab
event terms of #13 wrap this module (they read :class:`EpisodeSpec`; they do not re-implement the
draws), the ``configs/eval/*.yaml`` files are generated from :class:`RandomizationRanges` by
``scripts/make_eval_configs.py``, and the seen and unseen splits cannot drift apart because both
come from :meth:`RandomizationRanges.for_split`. numpy and the standard library only: nothing
here imports Isaac Sim, Isaac Lab or torch.

The table of section 2, row by row (field of :class:`RandomizationRanges`: seen / unseen only)
----------------------------------------------------------------------------------------------
* Plate position (x, y): uniform in a 0.20 m x 0.20 m box around ``plate_centre_xy_m``
  (``plate_box_size_m``) / the box shifted 5 cm toward the robot, negative ``y``
  (``plate_box_shift_toward_robot_m``: 0 seen, 0.05 unseen).
* Plate yaw: uniform in [-30, 30] deg (``plate_yaw_half_range_deg``) / same.
* Plate diameter: {0.22, 0.25} m / 0.27 m (``plate_diameters_m``).
* Rack slot: 1, 2, 3, 4 / 5 (``rack_slots``).
* Pelvis pose: +-0.02 m in x and y, +-5 deg yaw (``pelvis_position_noise_m``,
  ``pelvis_yaw_noise_deg``) / same.
* Friction: floor and object, each uniform in [0.6, 1.0] (``friction_range``) / same.
* External push, S0 only: 10 to 30 N every 3 s in a random horizontal direction
  (``push_force_range_n``, ``push_interval_s``, ``push_scenarios``) / 40 N.

The unseen split always uses the held-out diameter and slot (docs/evaluation.md section 1), and
the seen split never does. Interpretations the table leaves open, fixed here:

* "noise" of the pelvis pose is uniform and bounded (a Gaussian would have no range to test);
  ``dx`` and ``dy`` are independent, in the kitchen frame, relative to the nominal standing pose;
* "floor and object friction" are two independent draws, ``floor_friction`` and ``object_friction``
  (the plate is the only object whose friction matters);
* a push is a force magnitude in N with a horizontal direction, one independent draw per push.
  The table says "impulse" and gives newtons; how long the force acts is #13's decision, not a
  sampled quantity;
* pushes belong to S0 only, whose episode is ``push_horizon_s`` = 60 s (the S0 target of section 5
  is 60 s without a fall). They happen at ``push_interval_s``, ``2 * push_interval_s``, ... strictly
  before the horizon: 3, 6, ..., 57 s, 19 pushes. Any other scenario has an empty schedule;
* table rows a scenario does not use (the plate in S0, say) are still drawn, so a spec has the same
  fields for every scenario.

There is no mass row in section 2, so nothing here randomizes mass; the plate mass is the USD's.

Frames
------
Positions are in the kitchen frame of :mod:`kitchenbot.scene.kitchen_dims`: origin on the floor
at the middle of the counter's front edge, ``x`` along the counter (the robot's right, since it
faces ``+y``), ``y`` into the counter, ``z`` up. "Toward the robot" is therefore negative ``y``.
``plate_position_m`` has ``z`` = the counter-top face, which is where the plate USD's origin goes
(the origin of ``plate_*.usda`` is the centre of the underside). ``rack_slot_pose`` is the slot's
``(x, y, z, yaw)`` in the *rack* frame, straight from ``RackSlotLayout.slot_pose``; placing the
rack in the kitchen is #11's.

Where the draw is anchored
--------------------------
The nominal plate point is :data:`PLATE_NOMINAL_XY_M`, a constant of this module, ``provisional``:

* ``y = 0.30 m`` is mid-depth of the counter (``COUNTER_DEPTH_M / 2``), and the nearest to the
  robot that keeps the 27 cm plate on the counter in the shifted unseen box: its front edge is
  then 15 mm behind the counter edge in the worst draw (a test checks every draw);
* ``x = 0.15 m`` is to the right of the sink (the opening ends at ``x = -0.325 m``) and near the
  middle of the right palm's reach in ``y`` (#48: pelvis-frame ``y`` from -0.56 to 0.28 m).

Reproducibility rule (the sub-seed derivation)
----------------------------------------------
:func:`sample_episode` is a pure function of ``(seed, split, scenario, ranges, dims)``. It holds
no generator between calls, so the episode of a seed does not depend on how many other episodes
were sampled, or in which order. The generator is derived per episode::

    entropy = [SEED_DOMAIN, scenario_id, split_id, seed]
    rng = numpy.random.Generator(numpy.random.PCG64(numpy.random.SeedSequence(entropy)))

* ``SEED_DOMAIN`` is the constant 52 (this sampler; never changes);
* ``scenario_id`` is the scenario's number, ``S0`` -> 0 ... ``S4`` -> 4, so the same seed gives
  different draws in different scenarios;
* ``split_id`` is the index in ``kitchenbot.evaluation.seeds.SPLITS``, ``seen`` -> 0,
  ``unseen`` -> 1;
* ``seed`` is the evaluation seed.

That is ``numpy.random.default_rng(SeedSequence(entropy))`` today; ``PCG64`` is named so a future
numpy changing the default bit generator cannot change the draws. Only ``Generator.random``
(uniform doubles in [0, 1), whose stream numpy keeps stable) is used and every transform is plain
IEEE arithmetic, so the specs are reproducible across platforms and numpy versions, with one
exception: the push directions go through ``math.cos`` and ``math.sin`` and may differ in the last
bit between C libraries. The draws, in order, are ten uniforms (plate x, plate y, plate yaw,
diameter, slot, pelvis dx, dy, yaw, floor friction, object friction), then for an S0 episode two
uniforms per push (force, direction). Reordering them, or changing a range, changes every
episode: the ranges are frozen once M4 starts and changing them resets ``docs/experiments.md``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from kitchenbot.evaluation.seeds import SPLITS, default_seeds
from kitchenbot.scene import kitchen_dims as kd

__all__ = [
    "PLATE_NOMINAL_XY_M",
    "SCENARIOS",
    "SEED_DOMAIN",
    "SEEN_DIAMETERS_M",
    "UNSEEN_DIAMETERS_M",
    "EpisodeSpec",
    "RandomizationRanges",
    "SpawnDims",
    "episode_rng",
    "sample_episode",
    "sample_split",
]

#: Scenarios the sampler knows (docs/evaluation.md section 1). ``fake`` is not randomized.
SCENARIOS: tuple[str, ...] = ("S0", "S1", "S2", "S3", "S4")

#: First word of every episode's seed sequence; see the module docstring. Never changes.
SEED_DOMAIN = 52

#: Nominal plate centre on the counter top, kitchen frame, metres. Provisional (see module doc).
PLATE_NOMINAL_XY_M: tuple[float, float] = (0.15, kd.COUNTER_DEPTH_M / 2.0)

#: docs/evaluation.md section 2: 22 and 25 cm are drawn in the seen split, 27 cm is held out.
SEEN_DIAMETERS_M: tuple[float, ...] = (0.22, 0.25)
UNSEEN_DIAMETERS_M: tuple[float, ...] = (0.27,)

_N_FIXED_DRAWS = 10  # plate x, y, yaw, diameter, slot; pelvis dx, dy, yaw; two frictions


def _pair(name: str, value: Iterable[float]) -> tuple[float, float]:
    out = tuple(float(v) for v in value)
    if len(out) != 2:
        raise ValueError(f"{name} must have two values, got {out!r}")
    return out[0], out[1]


@dataclass(frozen=True)
class RandomizationRanges:
    """Every row of docs/evaluation.md section 2 for one split. The defaults are the seen split.

    Use :meth:`for_split` to get the ranges of a split; it is the one place the seen and unseen
    values are written down. All lengths in metres, angles in degrees, forces in newtons.
    ``to_dict`` / ``from_dict`` are the ``randomization`` block of ``configs/eval/*.yaml``.
    """

    split: str = "seen"
    #: Centre of the plate-position box before any shift, kitchen frame (x, y).
    plate_centre_xy_m: tuple[float, float] = PLATE_NOMINAL_XY_M
    #: Side of the square box the plate position is uniform in (20 cm x 20 cm).
    plate_box_size_m: float = 0.20
    #: The box moves this far toward the robot (negative y): 0 seen, 0.05 unseen.
    plate_box_shift_toward_robot_m: float = 0.0
    #: Plate yaw is uniform in [-this, this].
    plate_yaw_half_range_deg: float = 30.0
    #: Diameters drawn uniformly: {0.22, 0.25} seen, {0.27} unseen.
    plate_diameters_m: tuple[float, ...] = SEEN_DIAMETERS_M
    #: Rack slots (1-based) drawn uniformly: 1 to 4 seen, 5 unseen.
    rack_slots: tuple[int, ...] = kd.SEEN_SLOTS
    #: Pelvis x and y offset is uniform in [-this, this], independently.
    pelvis_position_noise_m: float = 0.02
    #: Pelvis yaw offset is uniform in [-this, this].
    pelvis_yaw_noise_deg: float = 5.0
    #: Floor friction and object friction, each uniform in this interval.
    friction_range: tuple[float, float] = (0.6, 1.0)
    #: Push force magnitude, uniform in this interval: (10, 30) seen, (40, 40) unseen.
    push_force_range_n: tuple[float, float] = (10.0, 30.0)
    #: Time between pushes.
    push_interval_s: float = 3.0
    #: Scenarios that get pushes (section 2: S0 only).
    push_scenarios: tuple[str, ...] = ("S0",)
    #: Pushes happen strictly before this time: the S0 episode length (section 5, 60 s).
    push_horizon_s: float = 60.0

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split {self.split!r} not in {SPLITS}")
        for name in ("plate_centre_xy_m", "friction_range", "push_force_range_n"):
            object.__setattr__(self, name, _pair(name, getattr(self, name)))
        object.__setattr__(
            self, "plate_diameters_m", tuple(float(d) for d in self.plate_diameters_m)
        )
        object.__setattr__(self, "rack_slots", tuple(self.rack_slots))
        object.__setattr__(self, "push_scenarios", tuple(self.push_scenarios))

        for name in (
            "plate_box_size_m",
            "pelvis_position_noise_m",
            "pelvis_yaw_noise_deg",
            "plate_yaw_half_range_deg",
            "plate_box_shift_toward_robot_m",
            "push_interval_s",
            "push_horizon_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and not negative, got {value!r}")
        if self.plate_box_size_m <= 0.0 or self.push_interval_s <= 0.0:
            raise ValueError("plate_box_size_m and push_interval_s must be positive")
        for name in ("friction_range", "push_force_range_n"):
            lo, hi = getattr(self, name)
            if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo <= hi):
                raise ValueError(f"{name} must be 0 <= low <= high, got {(lo, hi)!r}")
        if not self.plate_diameters_m or any(
            d not in kd.PLATE_DIAMETERS_M for d in self.plate_diameters_m
        ):
            raise ValueError(
                f"plate_diameters_m must be a non-empty subset of {kd.PLATE_DIAMETERS_M}, "
                f"got {self.plate_diameters_m!r}"
            )
        if not self.rack_slots or any(
            isinstance(s, bool) or not isinstance(s, int) or not 1 <= s <= kd.RACK_SLOTS
            for s in self.rack_slots
        ):
            raise ValueError(
                f"rack_slots must be integers in 1..{kd.RACK_SLOTS}, got {self.rack_slots!r}"
            )
        if any(s not in SCENARIOS for s in self.push_scenarios):
            raise ValueError(f"push_scenarios must be a subset of {SCENARIOS}")

    @classmethod
    def for_split(cls, split: str) -> RandomizationRanges:
        """The ranges of ``split``: ``"seen"`` (development) or ``"unseen"`` (held out)."""
        if split == "seen":
            return cls(split="seen")
        if split == "unseen":
            return cls(
                split="unseen",
                plate_box_shift_toward_robot_m=0.05,
                plate_diameters_m=UNSEEN_DIAMETERS_M,
                rack_slots=kd.UNSEEN_SLOTS,
                push_force_range_n=(40.0, 40.0),
            )
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")

    @property
    def plate_box_centre_xy_m(self) -> tuple[float, float]:
        """Centre of the plate-position box after the shift toward the robot."""
        cx, cy = self.plate_centre_xy_m
        return (cx, cy - self.plate_box_shift_toward_robot_m)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (lists, floats, ints, strings) in field order; JSON and YAML ready."""
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RandomizationRanges:
        """Inverse of :meth:`to_dict`. Unknown keys raise ``ValueError``; missing keys default."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown randomization keys {unknown}")
        return cls(**dict(data))


@dataclass(frozen=True)
class SpawnDims:
    """The scene numbers the draw reads from ``kitchen_dims``: counter height and the rack."""

    counter_height_m: float = kd.COUNTER_HEIGHT_M
    rack: kd.RackSlotLayout = field(default_factory=kd.RackSlotLayout)


@dataclass(frozen=True)
class EpisodeSpec:
    """One episode's randomized values. Plain floats, ints, strings and tuples only.

    Frames and units are in the module docstring. ``push_*`` tuples have equal length and are
    empty unless the scenario gets pushes. ``push_directions_xy`` are unit vectors in the kitchen
    frame's horizontal plane.
    """

    seed: int
    split: str
    scenario: str
    plate_position_m: tuple[float, float, float]
    plate_yaw_deg: float
    plate_diameter_m: float
    rack_slot: int
    rack_slot_pose: tuple[float, float, float, float]
    pelvis_offset_m: tuple[float, float]
    pelvis_yaw_offset_deg: float
    floor_friction: float
    object_friction: float
    push_times_s: tuple[float, ...]
    push_forces_n: tuple[float, ...]
    push_directions_xy: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (lists instead of tuples) that ``json.dumps`` and ``yaml.safe_dump`` take."""
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}


def _plain(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def episode_rng(seed: int, split: str, scenario: str) -> np.random.Generator:
    """The generator of one episode; the sub-seed rule is in the module docstring."""
    _check_seed(seed)
    _check_split(split)
    _check_scenario(scenario)
    entropy = [SEED_DOMAIN, int(scenario[1:]), SPLITS.index(split), int(seed)]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def _check_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")


def _check_scenario(scenario: str) -> None:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")


def _uniform(lo: float, hi: float, u: float) -> float:
    """``lo + (hi - lo) * u`` for ``u`` in [0, 1), held inside [lo, hi] against rounding."""
    return min(max(lo + (hi - lo) * float(u), lo), hi)


def _pick(options: tuple[Any, ...], u: float) -> Any:
    return options[min(int(float(u) * len(options)), len(options) - 1)]


def _push_times(ranges: RandomizationRanges) -> tuple[float, ...]:
    times = []
    k = 1
    while k * ranges.push_interval_s < ranges.push_horizon_s:
        times.append(k * ranges.push_interval_s)
        k += 1
    return tuple(times)


def sample_episode(
    seed: int,
    split: str,
    scenario: str,
    ranges: RandomizationRanges | None = None,
    dims: SpawnDims | None = None,
) -> EpisodeSpec:
    """Draw the episode of ``seed``: same arguments, same :class:`EpisodeSpec`, always.

    ``ranges`` defaults to ``RandomizationRanges.for_split(split)`` and must belong to ``split``;
    ``dims`` defaults to the ``kitchen_dims`` values. Any non-negative ``seed`` is accepted
    (training draws seeds outside the evaluation splits); which seeds belong to a split is the
    evaluator's rule, ``kitchenbot.evaluation.seeds``.
    """
    rng = episode_rng(seed, split, scenario)
    r = RandomizationRanges.for_split(split) if ranges is None else ranges
    if r.split != split:
        raise ValueError(f"ranges are for split {r.split!r}, not {split!r}")
    d = SpawnDims() if dims is None else dims

    u = rng.random(_N_FIXED_DRAWS)
    half_box = r.plate_box_size_m / 2.0
    cx, cy = r.plate_box_centre_xy_m
    x = _uniform(cx - half_box, cx + half_box, u[0])
    y = _uniform(cy - half_box, cy + half_box, u[1])
    yaw = _uniform(-r.plate_yaw_half_range_deg, r.plate_yaw_half_range_deg, u[2])
    diameter = float(_pick(r.plate_diameters_m, u[3]))
    slot = int(_pick(r.rack_slots, u[4]))
    noise = r.pelvis_position_noise_m
    pelvis_dx = _uniform(-noise, noise, u[5])
    pelvis_dy = _uniform(-noise, noise, u[6])
    pelvis_yaw = _uniform(-r.pelvis_yaw_noise_deg, r.pelvis_yaw_noise_deg, u[7])
    floor_friction = _uniform(*r.friction_range, u[8])
    object_friction = _uniform(*r.friction_range, u[9])

    times: tuple[float, ...] = ()
    forces: tuple[float, ...] = ()
    directions: tuple[tuple[float, float], ...] = ()
    if scenario in r.push_scenarios:
        times = _push_times(r)
        p = rng.random((len(times), 2))
        forces = tuple(_uniform(*r.push_force_range_n, v) for v in p[:, 0])
        angles = [math.radians(360.0 * float(v)) for v in p[:, 1]]
        directions = tuple((math.cos(a), math.sin(a)) for a in angles)

    return EpisodeSpec(
        seed=int(seed),
        split=split,
        scenario=scenario,
        plate_position_m=(x, y, float(d.counter_height_m)),
        plate_yaw_deg=yaw,
        plate_diameter_m=diameter,
        rack_slot=slot,
        rack_slot_pose=tuple(float(v) for v in d.rack.slot_pose(slot)),
        pelvis_offset_m=(pelvis_dx, pelvis_dy),
        pelvis_yaw_offset_deg=pelvis_yaw,
        floor_friction=floor_friction,
        object_friction=object_friction,
        push_times_s=times,
        push_forces_n=forces,
        push_directions_xy=directions,
    )


def sample_split(
    split: str,
    scenario: str,
    ranges: RandomizationRanges | None = None,
    dims: SpawnDims | None = None,
    seeds: Iterable[int] | None = None,
) -> tuple[EpisodeSpec, ...]:
    """The specs of a scenario's evaluation seeds, in seed order.

    ``seeds`` defaults to ``kitchenbot.evaluation.seeds.default_seeds(scenario, split)``: seeds
    0 to 99 (0 to 49 for S4) seen, 1000 to 1049 unseen. Each spec is independent of the others.
    """
    _check_split(split)
    _check_scenario(scenario)
    seed_list = default_seeds(scenario, split) if seeds is None else list(seeds)
    return tuple(sample_episode(s, split, scenario, ranges, dims) for s in seed_list)
