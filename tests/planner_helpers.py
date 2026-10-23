"""Fake arm, hand, plate and rack for the plate-to-rack planner tests. No simulation.

The fakes are deliberately simple: they exist to drive every planner transition and failure path
deterministically, not to model physics. Numbers that come from the docs cite them; the scene
coordinates are made up (one frame, metres, robot near the origin facing +x).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kitchenbot.planning.geometry import (
    Pose,
    position_distance,
    quat_angle,
    quat_apply,
    quat_conjugate,
    quat_from_yaw,
    quat_multiply,
    quat_slerp,
)
from kitchenbot.planning.plate_to_rack import (
    PlannerCommand,
    PlannerConfig,
    PlannerObservation,
    PlannerState,
    PlateToRackPlanner,
)

#: Slew limits of the arm target filter, docs/architecture.md section 7.
ARM_LINEAR_SPEED_MPS = 0.3
ARM_ANGULAR_SPEED_RPS = 1.0
#: KinematicFollowHand attaches within 3 cm of the rim, docs/architecture.md section 8.
ATTACH_RADIUS_M = 0.03
#: Dex3SynergyHand closes over 0.5 s, docs/architecture.md section 8.
CLOSE_DURATION_S = 0.5

SHOULDER = (0.0, -0.20, 1.15)
PALM_START = Pose((0.30, -0.20, 1.00))
PLATE_GRASP = Pose((0.45, -0.15, 0.93), quat_from_yaw(math.radians(10.0)))
COUNTER_EDGE = Pose((0.40, -0.40, 0.90))
SLOT_ENTRY = Pose((0.40, -0.60, 0.60), quat_from_yaw(math.radians(90.0)))


@dataclass
class FakeArm:
    """Moves the palm straight toward the commanded target at the section 7 slew limits.

    A target farther than ``reach_radius`` from ``shoulder`` is infeasible: the palm does not move
    and ``infeasible`` is set until a feasible target is commanded.
    """

    palm: Pose
    linear_speed: float = ARM_LINEAR_SPEED_MPS
    angular_speed: float = ARM_ANGULAR_SPEED_RPS
    shoulder: tuple[float, float, float] = SHOULDER
    reach_radius: float = math.inf
    infeasible: bool = False

    def step(self, target: Pose, dt: float) -> None:
        self.infeasible = position_distance(target.position, self.shoulder) > self.reach_radius
        if self.infeasible:
            return
        delta = target.position - self.palm.position
        dist = float(np.linalg.norm(delta))
        max_move = self.linear_speed * dt
        position = target.position if dist <= max_move else self.palm.position + delta * (
            max_move / dist
        )
        angle = quat_angle(self.palm.quat_wxyz, target.quat_wxyz)
        max_turn = self.angular_speed * dt
        quat = (
            target.quat_wxyz
            if angle <= max_turn
            else quat_slerp(self.palm.quat_wxyz, target.quat_wxyz, max_turn / angle)
        )
        self.palm = Pose(position, quat)


@dataclass
class FakeHand:
    """Honours the ``HandAction`` verbs as methods, like the ``HandController`` of section 8.

    ``close`` takes ``close_duration`` seconds; the plate is held at the end of it if
    ``plate_in_reach`` is true then. ``slip_after_s`` makes a held plate slip out after that many
    seconds of holding (the drop test).
    """

    close_duration: float = CLOSE_DURATION_S
    plate_in_reach: bool = False
    slip_after_s: float | None = None
    posture: str = "open"
    holding: bool = False
    _closing_s: float = 0.0
    _held_s: float = 0.0
    _slipped: bool = False

    def open(self) -> None:
        self.posture = "open"

    def pre_grasp(self) -> None:
        self.posture = "pre_grasp"

    def close(self) -> None:
        self.posture = "close"

    def is_holding(self) -> bool:
        return self.holding

    def update(self, dt: float) -> None:
        if self.posture != "close":
            self._closing_s = 0.0
            self.holding = False
            return
        if not self.holding and not self._slipped:
            if self._closing_s < self.close_duration:
                self._closing_s += dt
                if self._closing_s >= self.close_duration - 1e-9 and self.plate_in_reach:
                    self.holding = True
                    self._held_s = 0.0
            return
        if self.holding:
            self._held_s += dt
            if self.slip_after_s is not None and self._held_s >= self.slip_after_s - 1e-9:
                self.holding = False
                self._slipped = True


@dataclass
class FakeRack:
    """One slot as an axis-aligned box below ``slot_entry``, with a crude contact model.

    Inside the rack footprint and below the slot entry, a plate outside the slot's half-width is
    pressing on a tine (``tine_force_n``); inside the slot and below the floor it is pressing on
    the floor (a spring of ``floor_stiffness``). ``in_slot`` is the box test.
    """

    slot_entry: Pose = field(default_factory=lambda: SLOT_ENTRY)
    half_width_m: float = 0.02
    floor_depth_m: float = 0.08
    floor_slack_m: float = 0.01
    floor_stiffness_n_per_m: float = 1000.0
    rack_half_extent_m: float = 0.15
    tine_force_n: float = 50.0

    def _offsets(self, position: np.ndarray) -> tuple[float, float]:
        d = position - self.slot_entry.position
        return max(abs(float(d[0])), abs(float(d[1]))), float(d[2])

    def contact_force(self, position: np.ndarray) -> float:
        lateral, dz = self._offsets(position)
        if dz >= 0.0 or lateral > self.rack_half_extent_m:
            return 0.0
        if lateral > self.half_width_m:
            return self.tine_force_n
        return max(0.0, -dz - self.floor_depth_m) * self.floor_stiffness_n_per_m

    def in_slot(self, position: np.ndarray) -> bool:
        lateral, dz = self._offsets(position)
        bottom = -(self.floor_depth_m + self.floor_slack_m)
        return lateral <= self.half_width_m and bottom <= dz <= 0.0


@dataclass
class FakePlate:
    """The plate's grasp frame. Follows the palm rigidly while held, stays put otherwise."""

    pose: Pose = field(default_factory=lambda: PLATE_GRASP)
    _in_palm: tuple[np.ndarray, np.ndarray] | None = None

    def advance(self, palm: Pose, holding: bool) -> None:
        if not holding:
            self._in_palm = None
            return
        if self._in_palm is None:  # capture the palm-to-plate offset at the grasp
            inv = quat_conjugate(palm.quat_wxyz)
            self._in_palm = (
                quat_apply(inv, self.pose.position - palm.position),
                quat_multiply(inv, self.pose.quat_wxyz),
            )
        offset, rel = self._in_palm
        self.pose = Pose(
            palm.position + quat_apply(palm.quat_wxyz, offset),
            quat_multiply(palm.quat_wxyz, rel),
        )


@dataclass
class Fakes:
    """Everything the planner observes. Observation offsets make the planner see a wrong pose.

    ``hand`` is any object with the ``HandController`` verbs, ``is_holding()`` and ``update(dt)``,
    e.g. the #42 ``MockHand``. With ``hand_reach_from_geometry`` the harness sets the hand's
    ``plate_in_reach`` from the palm-to-plate distance each step; off keeps the hand's own.
    ``plate_follows_palm`` lets ``FakePlate`` carry the plate with the palm while the hand holds;
    turn it off for a hand that writes the plate pose itself (the #58 ``KinematicFollowHand``).
    """

    arm: FakeArm = field(default_factory=lambda: FakeArm(PALM_START))
    hand: Any = field(default_factory=FakeHand)
    hand_reach_from_geometry: bool = True
    plate: FakePlate = field(default_factory=FakePlate)
    rack: FakeRack = field(default_factory=FakeRack)
    counter_edge_pose: Pose = field(default_factory=lambda: COUNTER_EDGE)
    plate_observation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    slot_observation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    plate_follows_palm: bool = True

    def observe(self) -> PlannerObservation:
        plate = Pose(self.plate.pose.position + self.plate_observation_offset,
                     self.plate.pose.quat_wxyz)
        slot = Pose(self.rack.slot_entry.position + self.slot_observation_offset,
                    self.rack.slot_entry.quat_wxyz)
        return PlannerObservation(
            palm_pose=self.arm.palm,
            plate_pose=plate,
            slot_pose=slot,
            counter_edge_pose=self.counter_edge_pose,
            hand_holding=self.hand.is_holding(),
            rack_contact_force=self.rack.contact_force(self.plate.pose.position),
            arm_infeasible=self.arm.infeasible,
            plate_in_slot=self.rack.in_slot(self.plate.pose.position),
        )

    def apply(self, command: PlannerCommand, dt: float) -> None:
        """Send the command the way the composite environment will, then advance the fakes."""
        self.arm.step(command.arm_target, dt)
        getattr(self.hand, command.hand.value)()  # HandAction values are method names
        if self.hand_reach_from_geometry:
            palm_to_plate = position_distance(self.arm.palm.position, self.plate.pose.position)
            self.hand.plate_in_reach = palm_to_plate <= ATTACH_RADIUS_M
        self.hand.update(dt)
        if self.plate_follows_palm:
            self.plate.advance(self.arm.palm, self.hand.is_holding())


@dataclass
class EpisodeLog:
    """What one fake episode produced."""

    visited: list[tuple[float, PlannerState]]
    commands: list[PlannerCommand]
    planner: PlateToRackPlanner

    @property
    def states(self) -> list[PlannerState]:
        return [s for _, s in self.visited]

    @property
    def final(self) -> PlannerCommand:
        return self.commands[-1]

    def entered_at(self, state: PlannerState) -> float:
        return next(t for t, s in self.visited if s is state)


def run_episode(
    planner: PlateToRackPlanner | None = None,
    config: PlannerConfig | None = None,
    fakes: Fakes | None = None,
    hz: float = 50.0,
    max_s: float = 30.0,
) -> EpisodeLog:
    """Run the planner against the fakes until it is terminal or ``max_s`` has passed.

    ``planner`` defaults to a new planner built from ``config`` (or the default config).
    ``visited`` holds each state with the time it was entered; ``idle`` is entered at 0.
    """
    if planner is None:
        planner = PlateToRackPlanner(config)
    fakes = fakes if fakes is not None else Fakes()
    dt = 1.0 / hz
    planner.reset(fakes.observe())
    visited = [(0.0, planner.state)]
    commands: list[PlannerCommand] = []
    for k in range(1, int(round(max_s * hz)) + 1):
        command = planner.step(fakes.observe(), dt)
        commands.append(command)
        if planner.state is not visited[-1][1]:
            visited.append((round(k * dt, 6), planner.state))
        if command.state in (PlannerState.DONE, PlannerState.FAILED):
            break
        fakes.apply(command, dt)
    return EpisodeLog(visited=visited, commands=commands, planner=planner)
