"""Grasp and placement judges: docs/evaluation.md section 3 decided from state alone (#57).

The scenario backends feed these judges poses, the rack contact force, the hand's holding flag and
the episode time; the judges decide the ``grasp_success``, ``placement_success``,
``rack_collision`` and ``completion_time_s`` metrics of :mod:`kitchenbot.evaluation.metrics`, and
:func:`episode_outcome` turns those verdicts and the other failure signals into ``(success,
failure_tag)``. **They are the reference implementation of the section 3 definitions**: the
simulation backends (#22) and the CI dry run (#60) both call them, so both mean the same thing by
success. numpy, :mod:`kitchenbot.geometry` and :mod:`kitchenbot.scene.kitchen_dims` only (plus the
planner's config for shared numbers); nothing here imports a simulator.

Each judge is a per-episode scalar object: build it (or ``reset`` it) at episode start, then call
``step`` once per control step with ``t``, the episode time in seconds since reset (time zero),
strictly increasing. One judge watches one environment; whether the composite environment
vectorises the judges across ``num_envs`` is #22's decision, as it is for the planner.

Poses and frames
----------------
- **Plate pose** is the plate's body frame as ``scripts/make_plate_usd.py`` authors it (#30): origin
  at the centre of the underside, local ``z`` the plate normal (the planner's plate convention,
  :mod:`kitchenbot.geometry`), quaternions ``(w, x, y, z)``. It is *not* the planner's
  ``plate_pose``, which is the grasp frame on the top rim (docs/architecture.md section 9); a
  backend converts with ``kitchenbot.planning.geometry.body_pose_from_grasp`` (#67). With the
  planner's grasp frame at its slot entry, :func:`plate_in_slot`'s lowest rim point is at the tine
  tops, centred in the gap (``tests/test_planner_hand_contract.py``).
- **Rack frame** is the one of :mod:`kitchenbot.scene.kitchen_dims`: ``x`` across the tines, ``y``
  along the tine rows, ``z`` up, origin at the centre of the underside of the base frame. A seated
  plate stands vertically with its normal along ``x``. :func:`plate_in_slot` takes the plate pose
  already in the rack frame; :meth:`PlacementJudge.step` takes the world poses of the plate and of
  the rack frame and converts with :func:`kitchenbot.geometry.to_frame`. If the rack link of the
  dishwasher asset has other axes, #22 composes the fixed offset before calling.
- :class:`GraspJudge` works in the world frame (``z`` up).

Released
--------
``released`` means the hand reports not holding the plate (pass ``not hand_holding``). The release
is the first step at which it is true **after a step at which the hand held the plate**, so in S4
the plate lying on the counter before the grasp is not "released". A drop and an intentional
release look the same here; telling them apart is the caller's (``dropped`` in
:func:`episode_outcome`). ``rack_collision`` counts the contact force only on steps before the
release (the release step itself is not counted, as in the planner's ``release`` state); a bump
after it is ignored.

Thresholds
----------
Every number is in :data:`THRESHOLDS` with its source. The ones that are also planner numbers are
read from :class:`~kitchenbot.planning.plate_to_rack.PlannerConfig` (same quantity, one value); the
slot volume comes from :class:`~kitchenbot.scene.kitchen_dims.RackSlotLayout` (see
:func:`plate_in_slot`); the two stability tolerances are provisional, to be checked in #22 against
a plate resting in the simulated rack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from kitchenbot.evaluation.failure_tags import FailureTag, primary_failure_tag
from kitchenbot.evaluation.metrics import METRICS
from kitchenbot.geometry import (
    Pose,
    position_distance,
    quat_angle,
    quat_apply,
    tilt_from_vertical,
    to_frame,
)
from kitchenbot.planning.plate_to_rack import EVAL_3, PROVISIONAL, PlannerConfig
from kitchenbot.scene.kitchen_dims import PLATE_HEIGHT_M, RackSlotLayout

__all__ = [
    "GRASP_HOLD_S",
    "GRASP_LIFT_HEIGHT_M",
    "PLACEMENT_STABLE_S",
    "PLACEMENT_VERTICAL_TOL_DEG",
    "RACK_COLLISION_FORCE_N",
    "STABLE_ANG_TOL_RAD_S",
    "STABLE_LIN_TOL_M_S",
    "THRESHOLDS",
    "GraspJudge",
    "PlacementJudge",
    "SlotFit",
    "StabilityWindow",
    "Threshold",
    "episode_outcome",
    "plate_in_slot",
    "vertical_error_deg",
]

_EPS = 1e-9  # float slack for accumulated time and height comparisons, as in the planner


@dataclass(frozen=True)
class Threshold:
    """One judge number: its value, unit, where it comes from and what it means."""

    value: float
    unit: str
    source: str
    doc: str


_PLANNER_DEFAULTS = PlannerConfig()
_PLANNER_FIELDS = {f.name: f for f in fields(PlannerConfig)}


def _from_planner(name: str, unit: str, doc: str) -> Threshold:
    """A threshold that is the same quantity as a :class:`PlannerConfig` field: read, not copied."""
    source = f"PlannerConfig.{name} ({_PLANNER_FIELDS[name].metadata['source']})"
    return Threshold(float(getattr(_PLANNER_DEFAULTS, name)), unit, source, doc)


#: Every number the judges use, by name. The slot volume is geometry, see :func:`plate_in_slot`.
THRESHOLDS: dict[str, Threshold] = {
    "grasp_lift_height_m": _from_planner(
        "lift_height_m",
        "m",
        "grasp success: plate lifted 15 cm (docs/evaluation.md section 3); the planner's lift",
    ),
    "grasp_hold_s": Threshold(
        2.0, "s", EVAL_3, "grasp success: held for 2 s, without counter contact, after the lift"
    ),
    "placement_vertical_tol_deg": _from_planner(
        "vertical_tolerance_deg", "deg", "placement success: within 15 deg of vertical"
    ),
    "placement_stable_s": _from_planner(
        "hold_check_s", "s", "placement success: stable for 2 s after release"
    ),
    "rack_collision_force_n": _from_planner(
        "rack_collision_force_n", "N", "collision: rack contact force above 20 N before release"
    ),
    "stable_lin_tol_m_s": Threshold(
        0.01,
        "m/s",
        PROVISIONAL,
        "stable: plate speed at most 1 cm/s between steps (2 cm of creep over the 2 s window)",
    ),
    "stable_ang_tol_rad_s": Threshold(
        0.1,
        "rad/s",
        PROVISIONAL,
        "stable: plate turn rate at most 0.1 rad/s (5.7 deg/s) between steps",
    ),
}

GRASP_LIFT_HEIGHT_M = THRESHOLDS["grasp_lift_height_m"].value
GRASP_HOLD_S = THRESHOLDS["grasp_hold_s"].value
PLACEMENT_VERTICAL_TOL_DEG = THRESHOLDS["placement_vertical_tol_deg"].value
PLACEMENT_STABLE_S = THRESHOLDS["placement_stable_s"].value
RACK_COLLISION_FORCE_N = THRESHOLDS["rack_collision_force_n"].value
STABLE_LIN_TOL_M_S = THRESHOLDS["stable_lin_tol_m_s"].value
STABLE_ANG_TOL_RAD_S = THRESHOLDS["stable_ang_tol_rad_s"].value


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value}")
    return value


class _Clock:
    """Checks that a judge's timestamps are finite, non-negative and strictly increasing."""

    def __init__(self) -> None:
        self.last: float | None = None

    def tick(self, t: float) -> float:
        t = float(t)
        if not math.isfinite(t) or t < 0.0:
            raise ValueError(f"t must be a finite episode time >= 0, got {t}")
        if self.last is not None and t <= self.last:
            raise ValueError(f"t must increase strictly: got {t} after {self.last}")
        self.last = t
        return t


# ---------------------------------------------------------------------------------------------
# Geometry: vertical error and the slot volume
# ---------------------------------------------------------------------------------------------


def vertical_error_deg(plate_quat) -> float:
    """Angle in degrees between the plate's plane and the vertical, from its ``(w, x, y, z)`` quat.

    The plate normal is the rotated ``z`` axis (the planner's convention); the plate is vertical
    when the normal is horizontal, so the error is ``asin(|n_z|)``: 0 for a seated plate (normal
    along the rack ``x`` axis), 90 for a plate lying flat. It is
    :func:`kitchenbot.geometry.tilt_from_vertical`, the check the planner's ``align`` state applies
    with the same 15 deg, so the planner and the judge agree on "vertical". Pass the quaternion in
    the rack frame (as :class:`PlacementJudge` does): "vertical" is then the rack's ``z``, the world
    up while the rack is level. Turning the plate about the vertical (its normal no longer across
    the tines) is not counted here: below the tine tops the two tine rows keep a plate facing
    across them, and :func:`plate_in_slot` places the plate.
    """
    return math.degrees(tilt_from_vertical(plate_quat))


@dataclass(frozen=True)
class SlotFit:
    """Where a plate's lowest rim point is relative to one slot, rack frame, metres.

    The point is the lowest point of the plate's rim circle on its mid-plane (see
    :func:`plate_in_slot`). ``across_m`` (rack ``x``) and ``along_m`` (rack ``y``) are its signed
    offsets from the slot centre of :meth:`RackSlotLayout.slot_pose`; ``rim_bottom_m`` is its
    height above ``slot_entry_z`` (negative: below the tine tops). ``in_slot`` is true when the
    point lies in the slot volume: ``|across_m| <= across_limit_m``, ``|along_m| <= along_limit_m``
    and ``-depth_limit_m <= rim_bottom_m < 0``.
    """

    slot_index: int
    across_m: float
    along_m: float
    rim_bottom_m: float
    across_limit_m: float
    along_limit_m: float
    depth_limit_m: float
    in_slot: bool

    def explain(self) -> str:
        """One line: ``in slot 3`` or which bound the rim point is outside of, with numbers."""
        problems = []
        if abs(self.across_m) > self.across_limit_m:
            problems.append(f"across {self.across_m:+.3f} m beyond +-{self.across_limit_m:.3f} m")
        if abs(self.along_m) > self.along_limit_m:
            problems.append(f"along {self.along_m:+.3f} m beyond +-{self.along_limit_m:.3f} m")
        if self.rim_bottom_m >= 0.0:
            problems.append(f"rim bottom {self.rim_bottom_m:+.3f} m, not below the slot entry")
        elif self.rim_bottom_m < -self.depth_limit_m:
            problems.append(
                f"rim bottom {self.rim_bottom_m:+.3f} m, below the rack base "
                f"(-{self.depth_limit_m:.3f} m)"
            )
        if not problems:
            return f"in slot {self.slot_index}"
        return f"not in slot {self.slot_index}: " + "; ".join(problems)


def plate_in_slot(
    plate_pose_rack: Pose,
    slot_index: int,
    diameter_m: float,
    layout: RackSlotLayout,
    *,
    thickness_m: float = PLATE_HEIGHT_M,
) -> SlotFit:
    """Whether a plate is inside the volume of slot ``slot_index`` (1-based), and by how much.

    ``plate_pose_rack`` is the plate's body pose (origin at the centre of the underside, ``z`` the
    normal, #30) **in the rack frame**. The plate is modelled as a disc of ``diameter_m`` whose
    mid-plane lies ``thickness_m / 2`` along the normal from the origin (``PLATE_HEIGHT_M``, the
    20 mm the plate occupies standing in a slot, as in :meth:`RackSlotLayout.plate_fit`). The point
    tested is the lowest point of the rim circle on that mid-plane: the part of the plate that goes
    deepest into the slot. For a plate lying flat (no lowest point) it is the mid-plane centre.

    The slot volume is the slot of :class:`RackSlotLayout`, no extra number: across, the gap
    between tine ``i`` and tine ``i + 1`` (``|x - x_i| <= gap_width / 2``, the tine surfaces);
    along, between the two tine rows (``|y| <= row_spacing / 2``); in height, from the underside of
    the base frame (rack ``z = 0``) up to, not including, the slot entry (``slot_entry_z``, the tine
    tops). A plate resting on the tine tops, in the next slot, beside the rack or on the floor is
    outside it. The tines themselves are not modelled: in simulation they keep a plate that is in
    the volume upright and facing across them; a synthetic trajectory has to do that itself.
    """
    sx, sy, sz, _ = layout.slot_pose(slot_index)  # validates the index
    radius = _positive("diameter_m", diameter_m) / 2.0
    half_thickness = _positive("thickness_m", thickness_m) / 2.0
    normal = quat_apply(plate_pose_rack.quat_wxyz, (0.0, 0.0, 1.0))
    centre = plate_pose_rack.position + half_thickness * normal
    # Downward direction within the plate's plane: -z minus its component along the normal.
    down = np.array([0.0, 0.0, -1.0]) + normal[2] * normal
    norm = float(np.linalg.norm(down))
    bottom = centre + (radius / norm) * down if norm > 1e-9 else centre
    across = float(bottom[0] - sx)
    along = float(bottom[1] - sy)
    rim_bottom = float(bottom[2] - sz)
    across_limit = layout.gap_width / 2.0
    along_limit = layout.row_spacing / 2.0
    depth_limit = layout.slot_entry_z
    return SlotFit(
        slot_index=slot_index,
        across_m=across,
        along_m=along,
        rim_bottom_m=rim_bottom,
        across_limit_m=across_limit,
        along_limit_m=along_limit,
        depth_limit_m=depth_limit,
        in_slot=bool(
            abs(across) <= across_limit
            and abs(along) <= along_limit
            and -depth_limit <= rim_bottom < 0.0
        ),
    )


# ---------------------------------------------------------------------------------------------
# Stability window
# ---------------------------------------------------------------------------------------------


class StabilityWindow:
    """True once a pose has stayed still for ``duration_s``, judged from timestamped samples.

    Between two consecutive samples the speed is the distance over the time step and the turn rate
    is the rotation angle over the time step. An interval faster than ``lin_tol_m_s`` or
    ``ang_tol_rad_s`` restarts the window at its later sample, i.e. where the motion stopped; the
    first sample after construction or :meth:`reset` starts it. The window closes at the first
    sample at least ``duration_s`` after its start (``closed_at_s``) and stays closed while the
    samples stay still. A plate that settles at ``t_s`` closes the window at ``t_s + duration_s``.
    """

    def __init__(self, duration_s: float, lin_tol_m_s: float, ang_tol_rad_s: float) -> None:
        self.duration_s = _positive("duration_s", duration_s)
        self.lin_tol_m_s = _positive("lin_tol_m_s", lin_tol_m_s)
        self.ang_tol_rad_s = _positive("ang_tol_rad_s", ang_tol_rad_s)
        self.reset()

    def reset(self) -> None:
        """Forget every sample: the next one starts a new window."""
        self._last: tuple[float, Pose] | None = None
        self._since: float | None = None
        self._closed_at: float | None = None
        self.linear_speed_m_s = 0.0
        self.angular_speed_rad_s = 0.0

    def step(self, t: float, pose: Pose) -> bool:
        """Add ``pose`` sampled at ``t`` (after the previous sample); return :attr:`stable`."""
        t = float(t)
        if not math.isfinite(t):
            raise ValueError(f"t must be finite, got {t}")
        if self._last is None:
            self._since = t
            self.linear_speed_m_s = self.angular_speed_rad_s = 0.0
        else:
            t_prev, prev = self._last
            dt = t - t_prev
            if dt <= 0.0:
                raise ValueError(f"t must increase strictly: got {t} after {t_prev}")
            self.linear_speed_m_s = position_distance(prev.position, pose.position) / dt
            self.angular_speed_rad_s = quat_angle(prev.quat_wxyz, pose.quat_wxyz) / dt
            if (
                self.linear_speed_m_s > self.lin_tol_m_s
                or self.angular_speed_rad_s > self.ang_tol_rad_s
            ):
                self._since = t
                self._closed_at = None
        self._last = (t, pose)
        if self._closed_at is None and t - self._since >= self.duration_s - _EPS:
            self._closed_at = t
        return self.stable

    @property
    def stable(self) -> bool:
        """The samples have been still for at least ``duration_s`` up to the latest one."""
        return self._closed_at is not None

    @property
    def stable_since_s(self) -> float | None:
        """Start of the current still run (``None`` before the first sample)."""
        return self._since

    @property
    def still_for_s(self) -> float:
        """How long the current still run has lasted at the latest sample."""
        if self._last is None or self._since is None:
            return 0.0
        return self._last[0] - self._since

    @property
    def closed_at_s(self) -> float | None:
        """Time of the sample at which the current run reached ``duration_s``, else ``None``."""
        return self._closed_at


# ---------------------------------------------------------------------------------------------
# Grasp judge
# ---------------------------------------------------------------------------------------------


class GraspJudge:
    """Grasp success (docs/evaluation.md section 3) for one episode.

    The definition: plate lifted 15 cm and held for 2 s without contact with the counter.
    ``reset(plate_start_pose)`` records the plate's world pose at episode start; ``step(t,
    plate_pose, holding)`` then follows the plate's world pose and the hand's holding flag. The
    lift is the rise of the plate origin above its start height. The hold window opens at the first
    step at which the hand holds the plate, it is not in contact with the counter and it is at
    least ``lift_height_m`` up; it closes, and the grasp succeeds, at the first step ``hold_s``
    later through which the hand kept holding and the counter was never touched. The plate may go
    lower again during the hold (in S4 the transport descends towards the rack); losing it or
    touching the counter closes the window without success, and a later lift can open a new one.
    Once the grasp has succeeded the verdict is final and later steps change nothing.

    ``dropped`` follows the planner's rule (docs/architecture.md section 9): the hand had reported
    holding and then stopped, before the grasp succeeded. A plate that was never held is a grasp
    miss, not a drop. Drops after the grasp has succeeded (during transport) are the caller's to
    report to :func:`episode_outcome`.

    ``counter_contact`` is the plate-counter contact the caller reads from the scene (#22); it
    defaults to no contact, which is all a dry run without a counter model can say.
    """

    def __init__(
        self, lift_height_m: float = GRASP_LIFT_HEIGHT_M, hold_s: float = GRASP_HOLD_S
    ) -> None:
        self.lift_height_m = _positive("lift_height_m", lift_height_m)
        self.hold_s = _positive("hold_s", hold_s)
        self._start_z: float | None = None
        self._clear()

    def _clear(self) -> None:
        self._clock = _Clock()
        self._holding = False
        self._held_at: float | None = None
        self._lifted_at: float | None = None
        self._grasped_at: float | None = None
        self._grasp_lift_at: float | None = None
        self._dropped_at: float | None = None
        self._max_lift = -math.inf
        self._last_t: float | None = None

    def reset(self, plate_start_pose: Pose) -> None:
        """Start an episode with the plate at ``plate_start_pose`` (world frame)."""
        self._start_z = float(plate_start_pose.position[2])
        self._clear()

    def step(
        self, t: float, plate_pose: Pose, holding: bool, *, counter_contact: bool = False
    ) -> None:
        """Advance to episode time ``t`` with the plate's world pose and the hand's holding flag."""
        if self._start_z is None:
            raise RuntimeError("call reset(plate_start_pose) before step()")
        t = self._clock.tick(t)
        self._last_t = t
        if self._grasped_at is not None:
            return
        holding, counter_contact = bool(holding), bool(counter_contact)
        lift = float(plate_pose.position[2]) - self._start_z
        if holding:
            self._max_lift = max(self._max_lift, lift)
            if self._held_at is None:
                self._held_at = t
        elif self._holding and self._dropped_at is None:
            self._dropped_at = t
        self._holding = holding
        if not holding or counter_contact:
            self._lifted_at = None
            return
        if self._lifted_at is None and lift >= self.lift_height_m - _EPS:
            self._lifted_at = t
        if self._lifted_at is not None and t - self._lifted_at >= self.hold_s - _EPS:
            self._grasped_at = t
            self._grasp_lift_at = self._lifted_at

    @property
    def grasp_success(self) -> bool:
        return self._grasped_at is not None

    @property
    def grasped_at_s(self) -> float | None:
        """Episode time at which the 2 s hold completed (the grasp was confirmed), else ``None``."""
        return self._grasped_at

    @property
    def lifted_at_s(self) -> float | None:
        """When the hold window of the success opened, or of the open window; else ``None``."""
        return self._grasp_lift_at if self._grasped_at is not None else self._lifted_at

    @property
    def held_at_s(self) -> float | None:
        """First step at which the hand reported holding the plate, else ``None``."""
        return self._held_at

    @property
    def held(self) -> bool:
        """The hand reported holding the plate at some step (``held`` of ``episode_outcome``)."""
        return self._held_at is not None

    @property
    def dropped(self) -> bool:
        """Holding was lost after it had been established, before the grasp succeeded."""
        return self._dropped_at is not None

    @property
    def dropped_at_s(self) -> float | None:
        return self._dropped_at

    def why(self) -> str:
        """One line explaining the verdict so far."""
        if self._start_z is None:
            return "not started: call reset(plate_start_pose)"
        if self._grasped_at is not None:
            return (
                f"grasped: lifted {self.lift_height_m:.3f} m at {self._grasp_lift_at:.2f} s and "
                f"held {self.hold_s:.1f} s, confirmed at {self._grasped_at:.2f} s"
            )
        if self._held_at is None:
            return "no grasp: the hand never reported holding the plate"
        parts = [f"no grasp: held from {self._held_at:.2f} s"]
        if self._dropped_at is not None:
            parts.append(f"dropped at {self._dropped_at:.2f} s")
        if self._lifted_at is not None and self._last_t is not None:
            parts.append(
                f"lifted at {self._lifted_at:.2f} s, held {self._last_t - self._lifted_at:.2f} s "
                f"of {self.hold_s:.1f} s so far"
            )
        else:
            parts.append(
                f"lifted at most {max(self._max_lift, 0.0):.3f} m of {self.lift_height_m:.3f} m"
            )
        return "; ".join(parts)


# ---------------------------------------------------------------------------------------------
# Placement judge
# ---------------------------------------------------------------------------------------------


class PlacementJudge:
    """Placement success and rack collision (docs/evaluation.md section 3) for one slot.

    Placement success: after the release, the plate is in the volume of ``slot_index``
    (:func:`plate_in_slot`), within ``vertical_tol_deg`` of vertical (:func:`vertical_error_deg`)
    and still (:class:`StabilityWindow`, in the rack frame) for ``stable_s``, all three at every
    step of the window. A step that breaks any of them restarts the window, so a plate that wobbles
    and then settles succeeds ``stable_s`` after it settled. The verdict is final once the window
    closes; ``completion_time_s`` is the episode time of that step (from episode time zero, the
    "seconds from episode start to stable placement" of section 3), ``None`` until then.

    ``rack_collision``: a contact force above ``rack_force_n`` on any step before the release (see
    the module docstring for "released"). It is a metric, not yet a failure tag: a trial can collide
    and still be placed, and :func:`episode_outcome` turns it into the tag only when the placement
    failed. ``rack_contact_force_n`` is the largest robot-rack or plate-rack contact force of the
    step, in newtons, as #22 reads it from the rack's contact sensor.

    Re-grasping after the release restarts the window; collisions stay counted only before the
    first release.
    """

    def __init__(
        self,
        layout: RackSlotLayout,
        slot_index: int,
        diameter_m: float,
        vertical_tol_deg: float = PLACEMENT_VERTICAL_TOL_DEG,
        stable_s: float = PLACEMENT_STABLE_S,
        rack_force_n: float = RACK_COLLISION_FORCE_N,
        *,
        lin_tol_m_s: float = STABLE_LIN_TOL_M_S,
        ang_tol_rad_s: float = STABLE_ANG_TOL_RAD_S,
        thickness_m: float = PLATE_HEIGHT_M,
    ) -> None:
        layout.slot_pose(slot_index)  # validates the index against the layout
        self.layout = layout
        self.slot_index = slot_index
        self.diameter_m = _positive("diameter_m", diameter_m)
        self.thickness_m = _positive("thickness_m", thickness_m)
        self.vertical_tol_deg = _positive("vertical_tol_deg", vertical_tol_deg)
        self.stable_s = _positive("stable_s", stable_s)
        self.rack_force_n = _positive("rack_force_n", rack_force_n)
        self._window = StabilityWindow(stable_s, lin_tol_m_s, ang_tol_rad_s)
        self.reset()

    def reset(self) -> None:
        """Start a new episode with the same slot and plate."""
        self._clock = _Clock()
        self._window.reset()
        self._was_held = False
        self._holding_now = False
        self._released_at: float | None = None
        self._collision_at: float | None = None
        self._max_force = 0.0
        self._completion: float | None = None
        self._fit: SlotFit | None = None
        self._vertical_deg: float | None = None
        self._last_t: float | None = None

    def step(
        self,
        t: float,
        plate_pose_world: Pose,
        rack_pose_world: Pose,
        rack_contact_force_n: float,
        released: bool,
    ) -> None:
        """Advance to episode time ``t``.

        ``plate_pose_world`` is the plate's body pose and ``rack_pose_world`` the pose of the rack
        frame, both in the world; ``released`` is true when the hand reports not holding the plate.
        """
        t = self._clock.tick(t)
        force = float(rack_contact_force_n)
        if not math.isfinite(force) or force < 0.0:
            raise ValueError(f"rack_contact_force_n must be finite and >= 0, got {force}")
        self._last_t = t
        if self._completion is not None:
            return
        released = bool(released)
        self._holding_now = not released
        if not released:
            self._was_held = True
        if self._released_at is None:
            if not (released and self._was_held):  # before the release: count the contact
                self._max_force = max(self._max_force, force)
                if force > self.rack_force_n and self._collision_at is None:
                    self._collision_at = t
                return
            self._released_at = t
        if not released:  # held again after the release
            self._window.reset()
            return
        plate_rack = to_frame(plate_pose_world, rack_pose_world)
        self._fit = plate_in_slot(
            plate_rack, self.slot_index, self.diameter_m, self.layout, thickness_m=self.thickness_m
        )
        self._vertical_deg = vertical_error_deg(plate_rack.quat_wxyz)
        if not self._fit.in_slot or self._vertical_deg > self.vertical_tol_deg:
            self._window.reset()
            return
        if self._window.step(t, plate_rack):
            self._completion = self._window.closed_at_s

    @property
    def placement_success(self) -> bool:
        return self._completion is not None

    @property
    def completion_time_s(self) -> float | None:
        """Episode time at which the stability window closed, ``None`` unless placed."""
        return self._completion

    @property
    def rack_collision(self) -> bool:
        """Contact force above ``rack_force_n`` at some step before the release."""
        return self._collision_at is not None

    @property
    def rack_collision_at_s(self) -> float | None:
        return self._collision_at

    @property
    def max_force_before_release_n(self) -> float:
        return self._max_force

    @property
    def released(self) -> bool:
        """The plate has been released (``released`` of :func:`episode_outcome`)."""
        return self._released_at is not None

    @property
    def released_at_s(self) -> float | None:
        return self._released_at

    @property
    def slot_fit(self) -> SlotFit | None:
        """The :class:`SlotFit` of the latest step after the release."""
        return self._fit

    @property
    def last_vertical_error_deg(self) -> float | None:
        """:func:`vertical_error_deg` of the latest step after the release."""
        return self._vertical_deg

    def why(self) -> str:
        """One line explaining the verdict so far, with the rack collision if there was one."""
        line = self._placement_line()
        if self._collision_at is not None:
            line += (
                f"; rack contact {self._max_force:.1f} N over {self.rack_force_n:.1f} N before "
                f"release, first at {self._collision_at:.2f} s"
            )
        return line

    def _placement_line(self) -> str:
        slot = self.slot_index
        if self._completion is not None:
            return (
                f"placed in slot {slot}: {self._vertical_deg:.1f} deg from vertical, still "
                f"{self.stable_s:.1f} s, at {self._completion:.2f} s"
            )
        if self._released_at is None:
            if not self._was_held:
                return "not placed: the plate was never held, so never released"
            return "not placed: not released, the hand still holds the plate"
        head = f"not placed: released at {self._released_at:.2f} s"
        if self._holding_now:
            return f"{head}, held again"
        fit, vertical = self._fit, self._vertical_deg
        if fit is None or vertical is None:  # set on every released step; kept for the types
            return head
        problems = []
        if not fit.in_slot:
            problems.append(fit.explain())
        if vertical > self.vertical_tol_deg:
            problems.append(f"{vertical:.1f} deg from vertical, over {self.vertical_tol_deg:.1f}")
        if problems:
            return f"{head}; at {self._last_t:.2f} s " + "; ".join(problems)
        return (
            f"{head}; in slot {slot}, {vertical:.1f} deg from vertical, still "
            f"{self._window.still_for_s:.2f} s of {self.stable_s:.1f} s"
        )


# ---------------------------------------------------------------------------------------------
# Episode outcome
# ---------------------------------------------------------------------------------------------

_GRASP_SCENARIOS = METRICS["grasp_success"].scenarios  # S2, S4, fake and dry
_PLACEMENT_SCENARIOS = METRICS["placement_success"].scenarios  # S3, S4, fake and dry


def episode_outcome(
    scenario: str,
    *,
    fell: bool,
    ik_infeasible: bool,
    grasp: bool | None,
    placement: bool | None,
    dropped: bool,
    rack_collision_before_release: bool,
    timed_out: bool,
    held: bool | None = None,
    released: bool | None = None,
) -> tuple[bool, FailureTag]:
    """``(success, failure_tag)`` of one S2, S3 or S4 trial (``fake`` and ``dry`` mirror S4).

    Success follows ``metrics.SUCCESS_DEFINITIONS``: S2 needs ``grasp`` (the
    :attr:`GraspJudge.grasp_success`), S3 needs ``placement`` (:attr:`PlacementJudge.
    placement_success`), S4 needs both; and every one of them needs no fall and no timeout. S4's
    definition says so; for S2 and S3 it is the same rule, because both are also run with the
    balance policy active and the fixed-pelvis comparison of docs/evaluation.md section 6 only
    means something if a fall counts as a failure there. ``grasp`` is ignored in S3 (the plate
    starts in the hand) and ``placement`` in S2; either may be ``None`` there.

    A failed trial's tag is :func:`~kitchenbot.evaluation.failure_tags.primary_failure_tag` of the
    causes observed, i.e. the first of them in ``FAILURE_PRECEDENCE`` (fall, ik_infeasible,
    grasp_miss, drop, rack_collision, misplace, timeout). The causes:

    - ``fall`` if ``fell``; ``ik_infeasible`` if ``ik_infeasible``; ``drop`` if ``dropped`` (the
      plate was lost after the grasp, as the caller tells it: the judges cannot tell a drop from a
      release); ``timeout`` if ``timed_out`` (the time budget ran out before success).
    - ``grasp_miss`` when the grasp failed (S2, S4), the plate was not dropped and it was never
      held. ``held`` is :attr:`GraspJudge.held`; left ``None`` it counts as never held, so a failed
      grasp without a drop is a miss. A plate that was held but not lifted 15 cm for 2 s is not a
      miss; that trial ends in another cause, usually ``timeout``.
    - ``rack_collision`` when ``rack_collision_before_release`` (:attr:`PlacementJudge.
      rack_collision`) **and the placement failed** (S3, S4). With a successful placement the
      collision stays a metric (the collision rate counts it) and the trial succeeds: a trial can
      touch the rack above 20 N and still succeed (``metrics`` module docstring; docs/evaluation.md
      sections 3 and 7). This is a rule of the judges, not of the planner, which aborts on such a
      contact.
    - ``misplace`` when the placement failed after the grasp phase succeeded (always in S3), the
      plate was not dropped and it was released: placed outside the slot volume, beyond 15 deg, or
      never still for 2 s. ``released`` is :attr:`PlacementJudge.released`; left ``None`` it counts
      as released. A plate still in the hand is not misplaced; that trial ends in another cause.

    Raises ``ValueError`` for a scenario without a grasp or placement phase (S0, S1), for a missing
    ``grasp`` or ``placement`` the scenario needs, and for a failed trial with no cause (say
    ``timed_out=True`` if the episode simply ran out of time).
    """
    has_grasp = scenario in _GRASP_SCENARIOS
    has_placement = scenario in _PLACEMENT_SCENARIOS
    if not (has_grasp or has_placement):
        raise ValueError(f"episode_outcome judges S2, S3, S4, fake and dry, not {scenario!r}")
    if has_grasp and grasp is None:
        raise ValueError(f"{scenario} has a grasp phase: pass grasp=GraspJudge.grasp_success")
    if has_placement and placement is None:
        raise ValueError(
            f"{scenario} has a placement phase: pass placement=PlacementJudge.placement_success"
        )
    grasp_failed = has_grasp and not grasp
    placement_failed = has_placement and not placement
    if not (grasp_failed or placement_failed or fell or timed_out):
        return True, FailureTag.SUCCESS

    causes: list[FailureTag] = []
    if fell:
        causes.append(FailureTag.FALL)
    if ik_infeasible:
        causes.append(FailureTag.IK_INFEASIBLE)
    if grasp_failed and not dropped and not held:
        causes.append(FailureTag.GRASP_MISS)
    if dropped:
        causes.append(FailureTag.DROP)
    if placement_failed and rack_collision_before_release:
        causes.append(FailureTag.RACK_COLLISION)
    if placement_failed and not grasp_failed and not dropped and released is not False:
        causes.append(FailureTag.MISPLACE)
    if timed_out:
        causes.append(FailureTag.TIMEOUT)
    if not causes:
        raise ValueError(
            f"{scenario} trial failed (grasp={grasp}, placement={placement}) with no cause "
            "observed: pass timed_out=True if the episode ran out of time"
        )
    return False, primary_failure_tag(causes)
