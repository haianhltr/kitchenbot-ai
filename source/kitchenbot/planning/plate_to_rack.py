"""Plate-to-rack task planner: the finite-state machine of docs/architecture.md section 9.

``idle -> pre_grasp -> grasp -> lift -> transport -> align -> insert -> release -> done``, with a
``failed`` sink. The planner is single-environment pure Python (numpy, no simulator import): it
never calls the arm or the hand. :meth:`PlateToRackPlanner.step` reads a
:class:`PlannerObservation` and returns a :class:`PlannerCommand` holding a palm pose target for the
arm controller (section 7) and a :class:`HandAction` whose value is the name of the
``HandController`` method the composite environment calls (section 8).

What the poses in the observation mean
--------------------------------------
- ``plate_pose`` is the plate's **grasp frame**: the point on the rim the palm closes on, with the
  plate's orientation (``z`` is the plate normal). Turning the plate's body pose and diameter into
  that point is the environment layer's job (#21), so the planner needs no plate geometry.
- ``slot_pose`` is where that grasp frame sits when the plate's lower edge is at the top of the
  slot (the slot entry), with the slot's yaw. The plate is seated ``insertion_depth_m`` below it.
- ``counter_edge_pose`` is the counter-top edge the plate is carried over on its way to the rack.
- ``palm_pose`` is the controlled palm frame; at the grasp it coincides with the grasp frame.

Failure tags
------------
Each step checks the current state's failure conditions first, then its exit condition. Within one
step several causes may be observed; the tag is :func:`primary_failure_tag` of those, i.e. the
precedence of docs/evaluation.md section 7. Across steps, the first cause observed wins: the planner
moves to ``failed`` and returns the same command forever. Falls are not the planner's to detect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any

import numpy as np

from kitchenbot.evaluation.failure_tags import FailureTag, primary_failure_tag
from kitchenbot.geometry import (
    Pose,
    plate_vertical_quat,
    position_distance,
    quat_angle,
    tilt_from_vertical,
)

__all__ = [
    "HandAction",
    "PlannerCommand",
    "PlannerConfig",
    "PlannerObservation",
    "PlannerState",
    "PlateToRackPlanner",
    "Pose",
    "plan_waypoints",
]

#: Source tags for :class:`PlannerConfig` field metadata.
ARCH_7 = "docs/architecture.md section 7"
ARCH_9 = "docs/architecture.md section 9"
EVAL_3 = "docs/evaluation.md section 3"
PROVISIONAL = "provisional"

_EPS = 1e-9  # float slack for accumulated time comparisons


class PlannerState(StrEnum):
    """FSM states in execution order. The index in this order is the one-hot position."""

    IDLE = "idle"
    PRE_GRASP = "pre_grasp"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    ALIGN = "align"
    INSERT = "insert"
    RELEASE = "release"
    DONE = "done"
    FAILED = "failed"


TERMINAL_STATES = frozenset({PlannerState.DONE, PlannerState.FAILED})
_ACTIVE_ORDER: tuple[PlannerState, ...] = (
    PlannerState.IDLE,
    PlannerState.PRE_GRASP,
    PlannerState.GRASP,
    PlannerState.LIFT,
    PlannerState.TRANSPORT,
    PlannerState.ALIGN,
    PlannerState.INSERT,
    PlannerState.RELEASE,
)
_CARRYING = frozenset(
    {PlannerState.LIFT, PlannerState.TRANSPORT, PlannerState.ALIGN, PlannerState.INSERT}
)


class HandAction(StrEnum):
    """Hand command. Each value is the name of a ``HandController`` method (#42, section 8)."""

    OPEN = "open"
    PRE_GRASP = "pre_grasp"
    CLOSE = "close"


def _src(default: float, source: str, doc: str) -> Any:
    return field(default=default, metadata={"source": source, "doc": doc})


@dataclass(frozen=True)
class PlannerConfig:
    """Every number the planner uses. ``metadata["source"]`` cites a doc or says ``provisional``.

    Provisional values are this issue's (#43) first guesses, to be tuned in #21 against S3. The
    per-state timeouts are provisional; with the arm slewing at 0.3 m/s and 1 rad/s
    (docs/architecture.md section 7) each leaves at least twice the nominal fake-run time, and they
    sum to 27 s, inside the 30 s budget.
    """

    # Waypoint geometry
    pre_grasp_height_m: float = _src(0.08, ARCH_9, "palm 8 cm above the rim grasp point")
    lift_height_m: float = _src(
        0.15,
        ARCH_9,
        "lift 15 cm; also the grasp-success height in docs/evaluation.md section 3; the waypoint "
        "is aimed one position tolerance higher so the exit height is at least this",
    )
    counter_edge_clearance_m: float = _src(
        0.15, PROVISIONAL, "height above the counter edge for the first transport waypoint"
    )
    above_slot_height_m: float = _src(0.25, ARCH_9, "transport ends 25 cm above the slot")
    insertion_depth_m: float = _src(
        0.05, PROVISIONAL, "descent below the slot entry at which the plate counts as seated"
    )
    retreat_height_m: float = _src(0.10, ARCH_9, "palm retreats 10 cm up after opening")

    # Tolerances
    position_tolerance_m: float = _src(
        0.02, PROVISIONAL, "palm-to-waypoint distance that counts as reached; under the 3 cm "
        "hand-target error of docs/evaluation.md section 3"
    )
    orientation_tolerance_deg: float = _src(
        10.0, PROVISIONAL, "palm-to-waypoint rotation angle that counts as reached"
    )
    vertical_tolerance_deg: float = _src(
        15.0, EVAL_3, "plate within 15 deg of vertical (placement success) before insert"
    )

    # Forces
    rack_collision_force_n: float = _src(
        20.0, EVAL_3, "rack contact force above 20 N before release is a collision"
    )
    seat_contact_force_n: float = _src(
        2.0, PROVISIONAL, "rack contact at or above this (and not above 20 N) ends insert"
    )

    # Time
    hold_check_s: float = _src(
        2.0, EVAL_3, "placement must be stable for 2 s after release; checked at its end"
    )
    total_budget_s: float = _src(30.0, EVAL_3, "the 30 s task timeout of task success")
    timeout_idle_s: float = _src(0.5, PROVISIONAL, "idle exits on the first step")
    timeout_pre_grasp_s: float = _src(4.0, PROVISIONAL, "reach to 8 cm above the rim")
    timeout_grasp_s: float = _src(2.5, PROVISIONAL, "8 cm descent plus a 0.5 s close")
    timeout_lift_s: float = _src(2.0, PROVISIONAL, "15 cm lift")
    timeout_transport_s: float = _src(6.0, PROVISIONAL, "counter edge, then above the slot")
    timeout_align_s: float = _src(5.0, PROVISIONAL, "up to a half turn at 1 rad/s")
    timeout_insert_s: float = _src(3.0, PROVISIONAL, "30 cm descent at 0.3 m/s")
    timeout_release_s: float = _src(4.0, PROVISIONAL, "2 s hold check plus the retreat")

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{f.name} must be a positive finite number, got {value}")
        if self.seat_contact_force_n >= self.rack_collision_force_n:
            raise ValueError("seat_contact_force_n must be below rack_collision_force_n")
        if self.hold_check_s >= self.timeout_release_s:
            raise ValueError("timeout_release_s must leave room for hold_check_s")

    def timeout_s(self, state: PlannerState) -> float:
        """The provisional timeout of a non-terminal state."""
        if state in TERMINAL_STATES:
            raise ValueError(f"{state} has no timeout")
        return float(getattr(self, f"timeout_{state.value}_s"))

    def state_timeouts(self) -> dict[PlannerState, float]:
        """Timeouts of every non-terminal state, in execution order."""
        return {s: self.timeout_s(s) for s in _ACTIVE_ORDER}


@dataclass(frozen=True)
class PlannerObservation:
    """What the planner reads each step. Poses are in one frame (see the module docstring)."""

    palm_pose: Pose
    plate_pose: Pose
    slot_pose: Pose
    counter_edge_pose: Pose
    hand_holding: bool
    rack_contact_force: float
    arm_infeasible: bool
    plate_in_slot: bool


@dataclass(frozen=True)
class PlannerCommand:
    """What the planner asks for: a palm pose target and a hand verb."""

    arm_target: Pose
    hand: HandAction
    state: PlannerState
    failure: FailureTag | None = None


def plan_waypoints(
    plate_pose: Pose, slot_pose: Pose, counter_edge_pose: Pose, config: PlannerConfig
) -> dict[str, Pose]:
    """Palm waypoints of one episode, computed once from the poses recorded at reset.

    Until ``align`` the palm keeps the grasp frame's orientation (the plate is carried flat);
    ``aligned`` and ``insertion`` use :func:`plate_vertical_quat` of the slot yaw.

    The ``lift`` waypoint is aimed ``lift_height_m + position_tolerance_m`` above the grasp frame,
    not ``lift_height_m``: the ``lift`` state exits once the palm is within ``position_tolerance_m``
    of it, so a palm that stops a full tolerance short still has the plate at least
    ``lift_height_m`` up, the 15 cm the grasp judge asks for (docs/evaluation.md section 3). The
    next waypoint, ``counter_edge``, is at least as high as ``lift``, so the plate is carried above
    that height from then on.
    """
    cfg = config
    grasp = Pose(plate_pose.position, plate_pose.quat_wxyz)
    lift = grasp.raised(cfg.lift_height_m + cfg.position_tolerance_m)
    edge_z = max(
        float(lift.position[2]), float(counter_edge_pose.position[2]) + cfg.counter_edge_clearance_m
    )
    counter_edge = Pose(
        np.array([counter_edge_pose.position[0], counter_edge_pose.position[1], edge_z]),
        grasp.quat_wxyz,
    )
    vertical = plate_vertical_quat(slot_pose.yaw)
    above_slot = Pose(slot_pose.position, grasp.quat_wxyz).raised(cfg.above_slot_height_m)
    return {
        "pre_grasp": grasp.raised(cfg.pre_grasp_height_m),
        "grasp": grasp,
        "lift": lift,
        "counter_edge": counter_edge,
        "above_slot": above_slot,
        "aligned": above_slot.with_quat(vertical),
        "insertion": Pose(slot_pose.position, vertical).raised(-cfg.insertion_depth_m),
    }


class PlateToRackPlanner:
    """The section 9 FSM for one environment.

    Usage: ``reset(obs)`` once per episode, then ``step(obs, dt)`` at the control rate. The planner
    is a pure function of the observations it was given and the elapsed time.
    """

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config if config is not None else PlannerConfig()
        self._ready = False
        self._state = PlannerState.IDLE
        self._time_in_state = 0.0
        self._elapsed = 0.0
        self._failure: FailureTag | None = None
        self._failed_in: PlannerState | None = None
        self._terminal_command: PlannerCommand | None = None
        self._waypoints: dict[str, Pose] = {}
        self._hand = HandAction.OPEN
        self._closing = False
        self._released = False
        self._transport_index = 0

    # ------------------------------------------------------------------ read-only state

    @property
    def state(self) -> PlannerState:
        return self._state

    @property
    def time_in_state(self) -> float:
        """Seconds since the current state was entered (frozen once terminal)."""
        return float(self._time_in_state)

    @property
    def elapsed(self) -> float:
        """Seconds since ``reset``."""
        return float(self._elapsed)

    @property
    def failure(self) -> FailureTag | None:
        return self._failure

    @property
    def failed_in(self) -> PlannerState | None:
        """The state that was active when the planner failed."""
        return self._failed_in

    @property
    def waypoints(self) -> dict[str, Pose]:
        """The episode's waypoints; ``retreat`` appears once ``release`` is entered."""
        return dict(self._waypoints)

    def state_one_hot(self) -> tuple[float, ...]:
        """The ``planner`` observation group (docs/architecture.md section 11)."""
        return tuple(1.0 if s is self._state else 0.0 for s in PlannerState)

    # ------------------------------------------------------------------ episode

    def reset(self, obs: PlannerObservation) -> None:
        """Record the plate, slot and counter-edge poses and compute the waypoints."""
        self._state = PlannerState.IDLE
        self._time_in_state = 0.0
        self._elapsed = 0.0
        self._failure = None
        self._failed_in = None
        self._terminal_command = None
        self._hand = HandAction.OPEN
        self._closing = False
        self._released = False
        self._transport_index = 0
        self._waypoints = plan_waypoints(
            obs.plate_pose, obs.slot_pose, obs.counter_edge_pose, self.config
        )
        self._waypoints["idle"] = obs.palm_pose
        self._ready = True

    def step(self, obs: PlannerObservation, dt: float) -> PlannerCommand:
        """Advance ``dt`` seconds: the state's failures first, then its exit, then the command."""
        if not self._ready:
            raise RuntimeError("call reset(obs) before step()")
        if self._terminal_command is not None:
            return self._terminal_command
        if not dt > 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        self._elapsed += dt
        self._time_in_state += dt

        causes = self._failure_causes(obs)
        if causes:
            return self._fail(primary_failure_tag(causes), obs)

        self._update_progress(obs)
        if self._exit_condition(obs):
            self._enter(self._next_state(), obs)
        return self._command()

    # ------------------------------------------------------------------ internals

    def _reached(self, palm: Pose, target: Pose) -> bool:
        cfg = self.config
        return position_distance(palm.position, target.position) <= cfg.position_tolerance_m and (
            math.degrees(quat_angle(palm.quat_wxyz, target.quat_wxyz))
            <= cfg.orientation_tolerance_deg
        )

    def _transport_targets(self) -> tuple[Pose, Pose]:
        return self._waypoints["counter_edge"], self._waypoints["above_slot"]

    def _hold_check_passed(self, obs: PlannerObservation) -> bool:
        return (
            self._released
            and self._time_in_state >= self.config.hold_check_s - _EPS
            and self._reached(obs.palm_pose, self._waypoints["retreat"])
        )

    def _failure_causes(self, obs: PlannerObservation) -> list[FailureTag]:
        cfg, s = self.config, self._state
        causes: list[FailureTag] = []
        if s is not PlannerState.IDLE and obs.arm_infeasible:
            causes.append(FailureTag.IK_INFEASIBLE)
        if s in _CARRYING and not obs.hand_holding:
            causes.append(FailureTag.DROP)
        if s is not PlannerState.RELEASE and obs.rack_contact_force > cfg.rack_collision_force_n:
            causes.append(FailureTag.RACK_COLLISION)
        if s is PlannerState.RELEASE and self._hold_check_passed(obs) and not obs.plate_in_slot:
            causes.append(FailureTag.MISPLACE)
        if self._time_in_state > cfg.timeout_s(s) + _EPS:
            grasp_missed = s is PlannerState.GRASP and not obs.hand_holding
            causes.append(FailureTag.GRASP_MISS if grasp_missed else FailureTag.TIMEOUT)
        if self._elapsed > cfg.total_budget_s + _EPS:
            causes.append(FailureTag.TIMEOUT)
        return causes

    def _update_progress(self, obs: PlannerObservation) -> None:
        s = self._state
        if s is PlannerState.GRASP and not self._closing:
            self._closing = self._reached(obs.palm_pose, self._waypoints["grasp"])
        elif s is PlannerState.TRANSPORT and self._transport_index == 0:
            if self._reached(obs.palm_pose, self._transport_targets()[0]):
                self._transport_index = 1
        elif s is PlannerState.RELEASE and not self._released:
            self._released = not obs.hand_holding

    def _exit_condition(self, obs: PlannerObservation) -> bool:
        s, wp, palm = self._state, self._waypoints, obs.palm_pose
        if s is PlannerState.IDLE:
            return True
        if s is PlannerState.PRE_GRASP:
            return self._reached(palm, wp["pre_grasp"])
        if s is PlannerState.GRASP:
            return obs.hand_holding
        if s is PlannerState.LIFT:
            return self._reached(palm, wp["lift"])
        if s is PlannerState.TRANSPORT:
            return self._transport_index == 1 and self._reached(palm, wp["above_slot"])
        if s is PlannerState.ALIGN:
            vertical = math.degrees(tilt_from_vertical(obs.plate_pose.quat_wxyz))
            return self._reached(palm, wp["aligned"]) and (
                vertical <= self.config.vertical_tolerance_deg
            )
        if s is PlannerState.INSERT:
            seated = obs.rack_contact_force >= self.config.seat_contact_force_n
            return seated or self._reached(palm, wp["insertion"])
        if s is PlannerState.RELEASE:
            return self._hold_check_passed(obs)  # plate_in_slot holds: misplace was checked first
        return False

    def _next_state(self) -> PlannerState:
        return list(PlannerState)[list(PlannerState).index(self._state) + 1]

    def _enter(self, state: PlannerState, obs: PlannerObservation) -> None:
        self._state = state
        self._time_in_state = 0.0
        if state is PlannerState.RELEASE:
            entry = obs.palm_pose
            self._waypoints["release"] = entry
            self._waypoints["retreat"] = entry.raised(self.config.retreat_height_m)

    def _command(self) -> PlannerCommand:
        s, wp = self._state, self._waypoints
        if s is PlannerState.IDLE:
            target, hand = wp["idle"], HandAction.OPEN
        elif s is PlannerState.PRE_GRASP:
            target, hand = wp["pre_grasp"], HandAction.PRE_GRASP
        elif s is PlannerState.GRASP:
            target = wp["grasp"]
            hand = HandAction.CLOSE if self._closing else HandAction.PRE_GRASP
        elif s is PlannerState.LIFT:
            target, hand = wp["lift"], HandAction.CLOSE
        elif s is PlannerState.TRANSPORT:
            target, hand = self._transport_targets()[self._transport_index], HandAction.CLOSE
        elif s is PlannerState.ALIGN:
            target, hand = wp["aligned"], HandAction.CLOSE
        elif s is PlannerState.INSERT:
            target, hand = wp["insertion"], HandAction.CLOSE
        elif s is PlannerState.RELEASE:
            target = wp["retreat"] if self._released else wp["release"]
            hand = HandAction.OPEN
        else:  # DONE
            target, hand = wp["retreat"], HandAction.OPEN
        self._hand = hand
        command = PlannerCommand(arm_target=target, hand=hand, state=s)
        if s is PlannerState.DONE:
            self._terminal_command = command
        return command

    def _fail(self, tag: FailureTag, obs: PlannerObservation) -> PlannerCommand:
        """Hold the palm where it is, keep the hand doing what it was doing, and stop."""
        self._failed_in = self._state
        self._failure = tag
        self._state = PlannerState.FAILED
        self._time_in_state = 0.0
        self._terminal_command = PlannerCommand(
            arm_target=obs.palm_pose, hand=self._hand, state=PlannerState.FAILED, failure=tag
        )
        return self._terminal_command
