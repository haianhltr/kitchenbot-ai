"""Test doubles for the plate-to-rack planner: fake arm, hand, plate and rack. No simulation.

**These are stand-ins, not models.** They exist to drive every planner transition and failure path
deterministically (the #43 unit tests, ``tests/planner_helpers.py`` re-exports them) and, since
#60, to run the whole evaluation chain on CI without a simulator
(:mod:`kitchenbot.evaluation.dry_run`). #21 replaces them with the simulated arm, hand, plate and
rack of the S3 environment, and #22 with the composite S4 environment; the planner's observation
and command (``PlannerObservation``, ``PlannerCommand``) and the ``HandController`` verbs are the
seam that stays. Numbers that come from the docs cite them.

Frames (used by :meth:`Fakes.from_episode`; the hand-built defaults below are in a made-up frame)
-------------------------------------------------------------------------------------------------
- **World frame = the kitchen frame** of :mod:`kitchenbot.scene.kitchen_dims`: origin on the floor
  at the middle of the counter's front edge, ``x`` along the counter (the robot's right, since it
  faces ``+y``), ``y`` into the counter, ``z`` up. The fakes, the planner and the judges all work
  in it. ``EpisodeSpec.plate_position_m`` is already in it, so no conversion is needed.
- **Pelvis frame** (the frame of the reachability envelope): the pelvis link, ``x`` forward, ``y``
  left, ``z`` up. The nominal pelvis is ``(0, -ROBOT_STANDOFF_M, PELVIS_HEIGHT_M)`` facing ``+y``
  (yaw 90 degrees); ``EpisodeSpec.pelvis_offset_m`` moves it in the kitchen ``x`` and ``y`` and
  ``pelvis_yaw_offset_deg`` turns it, as the sampler defines. The pelvis does not move during an
  episode (no balance model), so the pelvis pose is a constant of the episode.
- **Rack frame** (:class:`~kitchenbot.scene.kitchen_dims.RackSlotLayout`): ``x`` across the tines,
  ``y`` along the rows, ``z`` up. Where it sits in the kitchen is #11's decision;
  :data:`DRY_RUN_RACK_FRAME` is a placeholder for the dry run: the provisional pose derived in
  ``docs/reach_gap_analysis.md`` (#56) from architecture section 10, a floor-standing
  ``dishwasher_2`` to the robot's right with its front flush with the counter's front edge and the
  lower rack pulled out, yaw 0. #59 adds the dishwasher pose to ``kitchen_dims`` and #11 checks it;
  then this constant goes.
- **Plate poses.** ``FakePlate.pose`` is the plate's **grasp frame** on the rim, what the planner
  observes (docs/architecture.md section 9). The judges (:mod:`kitchenbot.evaluation.judges`) take
  the plate **body** pose, origin at the centre of the underside. The conversion is
  :func:`body_pose_from_grasp` / :func:`grasp_pose_from_body`: the grasp frame sits a radius along
  the body's local ``+x``, same orientation. **That is the convention of #58's tests**
  (``tests/test_planner_kinematic_follow.py``, ``CENTRE_IN_GRASP``), not a measured rule: it puts
  the grasp frame on the underside plane of the rim, and #21's environment layer decides the real
  rim point. Two consequences to reconcile there: the planner's quarter turn to vertical
  (``plate_vertical_quat``) sends the local ``+x`` rim point to the *bottom* of the plate, so with
  this convention the slot pose is at the tine tops; and ``docs/reach_gap_analysis.md`` (#56)
  assumes another one, the rim point nearest the robot, 2 cm up, with the slot pose a diameter
  above the tine tops (the hand on the top rim).
- **Slot entry.** :meth:`FakeRack.from_layout` puts the grasp frame at the slot entry shifted by
  half a plate thickness against the slot's ``x`` so that the plate stands centred in the gap once
  the planner has turned it vertical (its normal then points along ``+x``; the judge's mid-plane is
  half a thickness along the normal from the body origin).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from kitchenbot.controllers.reachability import ReachabilityEnvelope
from kitchenbot.controllers.target_filter import ArmTargetPipeline
from kitchenbot.geometry import (
    Pose,
    pose_compose,
    position_distance,
    quat_angle,
    quat_apply,
    quat_conjugate,
    quat_from_yaw,
    quat_multiply,
    quat_slerp,
    to_frame,
)
from kitchenbot.planning.plate_to_rack import (
    PlannerCommand,
    PlannerConfig,
    PlannerObservation,
    PlannerState,
    PlateToRackPlanner,
)
from kitchenbot.scene import kitchen_dims as kd

if TYPE_CHECKING:
    from kitchenbot.scene.randomization import EpisodeSpec

__all__ = [
    "ARM_ANGULAR_SPEED_RPS",
    "ARM_LINEAR_SPEED_MPS",
    "ATTACH_RADIUS_M",
    "CLOSE_DURATION_S",
    "COUNTER_EDGE",
    "DRY_RUN_RACK_FRAME",
    "PALM_START",
    "PALM_START_IN_PELVIS",
    "PELVIS_HEIGHT_M",
    "PLATE_GRASP",
    "SHOULDER",
    "SLOT_ENTRY",
    "EpisodeLog",
    "FakeArm",
    "FakeHand",
    "FakePlate",
    "FakeRack",
    "Fakes",
    "body_pose_from_grasp",
    "grasp_pose_from_body",
    "pelvis_pose",
    "run_episode",
]

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

#: Standing pelvis height, m: the base-height target of docs/architecture.md section 6 and the
#: ``init_state`` of ``G1_29DOF_CFG`` (docs/isaaclab_3_api_notes.md).
PELVIS_HEIGHT_M = 0.75
#: Palm at the start of an episode, in the pelvis frame: in front of the right hip, elbow bent,
#: turned so that its world heading is that of the world ``x`` axis at the nominal pelvis yaw.
#: Provisional (#60); inside the committed envelope, a test checks it.
PALM_START_IN_PELVIS = Pose((0.25, -0.20, 0.25), quat_from_yaw(-math.pi / 2.0))
#: Rack frame in the kitchen frame for the dry run: ``(x, y, z) = (0.55, 0.0, 0.09)``, yaw 0, the
#: provisional pose of ``PROVISIONAL_RACK_POSE`` in ``scripts/reach_gap_analysis.py`` (#56). Not a
#: measurement; see the module docstring.
DRY_RUN_RACK_FRAME = Pose((0.55, 0.0, 0.09))


def body_pose_from_grasp(grasp: Pose, diameter_m: float) -> Pose:
    """The plate body pose (origin at the underside centre) from its rim grasp frame.

    The body origin lies a radius against the grasp frame's local ``x``; same orientation. This is
    the convention of #58's tests, see the module docstring.
    """
    return pose_compose(grasp, Pose((-diameter_m / 2.0, 0.0, 0.0)))


def grasp_pose_from_body(body: Pose, diameter_m: float) -> Pose:
    """Inverse of :func:`body_pose_from_grasp`: the rim grasp frame of a plate body pose."""
    return pose_compose(body, Pose((diameter_m / 2.0, 0.0, 0.0)))


def pelvis_pose(
    spec: EpisodeSpec, stand_off_m: float = kd.ROBOT_STANDOFF_M, height_m: float = PELVIS_HEIGHT_M
) -> Pose:
    """The episode's pelvis pose in the kitchen frame (module docstring): nominal plus the noise."""
    dx, dy = spec.pelvis_offset_m
    yaw = math.pi / 2.0 + math.radians(spec.pelvis_yaw_offset_deg)
    return Pose((dx, -stand_off_m + dy, height_m), quat_from_yaw(yaw))


@dataclass
class FakeArm:
    """Moves the palm straight toward the commanded target at the section 7 slew limits.

    Without ``pipeline``: a target farther than ``reach_radius`` from ``shoulder`` is infeasible,
    the palm does not move and ``infeasible`` is set until a feasible target is commanded.

    With ``pipeline`` (an :class:`~kitchenbot.controllers.target_filter.ArmTargetPipeline`, #48)
    and ``pelvis`` (the pelvis pose in the planner's frame): the target goes through the
    pipeline's world-to-pelvis transform, reachability gate and slew filter, ``infeasible`` is the
    gate's verdict (the planner's ``arm_infeasible``), and the palm is the filtered pelvis-frame
    target back in the world frame, i.e. an IK that tracks exactly. The pipeline keeps steering to
    its last feasible target while the gate refuses. ``linear_speed`` and ``angular_speed`` then
    belong to the pipeline's filter (``Fakes.from_episode`` builds it with the same numbers).
    ``refusals`` counts the steps on which the arm reported ``infeasible``.
    """

    palm: Pose
    linear_speed: float = ARM_LINEAR_SPEED_MPS
    angular_speed: float = ARM_ANGULAR_SPEED_RPS
    shoulder: tuple[float, float, float] = SHOULDER
    reach_radius: float = math.inf
    infeasible: bool = False
    pipeline: ArmTargetPipeline | None = None
    pelvis: Pose | None = None
    refusals: int = 0

    def __post_init__(self) -> None:
        if self.pipeline is not None:
            if self.pelvis is None:
                raise ValueError("a pipeline needs the pelvis pose")
            self.pipeline.reset(to_frame(self.palm, self.pelvis), self.pelvis)

    def step(self, target: Pose, dt: float) -> None:
        if self.pipeline is not None:
            out = self.pipeline.step(target, self.pelvis, dt)
            self.infeasible = out.infeasible
            self.palm = pose_compose(self.pelvis, out.pose)
            self.refusals += int(self.infeasible)
            return
        self.infeasible = position_distance(target.position, self.shoulder) > self.reach_radius
        if self.infeasible:
            self.refusals += 1
            return
        delta = target.position - self.palm.position
        dist = float(np.linalg.norm(delta))
        max_move = self.linear_speed * dt
        position = (
            target.position if dist <= max_move else self.palm.position + delta * (max_move / dist)
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

    ``frame``, ``slot_index`` and ``layout`` are set by :meth:`from_layout`: the world pose of the
    rack frame (what the placement judge needs) and the slot and layout the box was built from.
    """

    slot_entry: Pose = field(default_factory=lambda: SLOT_ENTRY)
    half_width_m: float = 0.02
    floor_depth_m: float = 0.08
    floor_slack_m: float = 0.01
    floor_stiffness_n_per_m: float = 1000.0
    rack_half_extent_m: float = 0.15
    tine_force_n: float = 50.0
    frame: Pose | None = None
    slot_index: int | None = None
    layout: kd.RackSlotLayout | None = None

    @classmethod
    def from_layout(cls, layout: kd.RackSlotLayout, slot_index: int, frame: Pose) -> FakeRack:
        """The fake rack of slot ``slot_index`` (1-based) of ``layout`` placed at ``frame``.

        ``slot_entry`` is the planner's slot pose (module docstring, "Slot entry"); the box is the
        gap between two tines, narrowed to the clearance a plate leaves on each side
        (``(gap_width - thickness) / 2``, 12 mm for the default layout), as deep as the tines are
        tall; the footprint is the rack's larger half dimension. Nothing here models a lean.
        """
        sx, sy, sz, yaw = layout.slot_pose(slot_index)  # validates the index
        thickness = kd.PLATE_HEIGHT_M
        entry = pose_compose(frame, Pose((sx - thickness / 2.0, sy, sz), quat_from_yaw(yaw)))
        return cls(
            slot_entry=entry,
            half_width_m=(layout.gap_width - thickness) / 2.0,
            floor_depth_m=layout.tine_height,
            rack_half_extent_m=max(layout.width, layout.depth) / 2.0,
            frame=frame,
            slot_index=slot_index,
            layout=layout,
        )

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
    """The plate's grasp frame. Follows the palm rigidly while held, stays put otherwise.

    ``diameter_m`` is only needed to report the body pose (:meth:`body_pose`); :meth:`from_spec`
    sets it.
    """

    pose: Pose = field(default_factory=lambda: PLATE_GRASP)
    _in_palm: tuple[np.ndarray, np.ndarray] | None = None
    diameter_m: float | None = None

    @classmethod
    def from_spec(cls, spec: EpisodeSpec) -> FakePlate:
        """The plate of an episode: lying flat on the counter at the sampled pose.

        ``spec.plate_position_m`` (kitchen frame, ``z`` the counter-top face) is the body origin,
        the yaw turns it about ``z``; the grasp frame is derived from them.
        """
        body = Pose(spec.plate_position_m, quat_from_yaw(math.radians(spec.plate_yaw_deg)))
        return cls(
            pose=grasp_pose_from_body(body, spec.plate_diameter_m), diameter_m=spec.plate_diameter_m
        )

    def body_pose(self) -> Pose:
        """The plate body pose for the judges (module docstring); needs ``diameter_m``."""
        if self.diameter_m is None:
            raise ValueError("the body pose needs diameter_m (build the plate with from_spec)")
        return body_pose_from_grasp(self.pose, self.diameter_m)

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

    @classmethod
    def from_episode(
        cls,
        spec: EpisodeSpec,
        layout: kd.RackSlotLayout,
        *,
        gate: bool,
        envelope: ReachabilityEnvelope | None = None,
        stand_off_m: float = kd.ROBOT_STANDOFF_M,
        rack_frame: Pose = DRY_RUN_RACK_FRAME,
    ) -> Fakes:
        """The fakes of one sampled episode, all in the kitchen frame (module docstring).

        The plate, its grasp frame, the slot (``spec.rack_slot`` of ``layout`` at ``rack_frame``)
        and the counter edge come from the spec; the counter-edge waypoint is the front edge of
        the counter (``y = 0``, counter-top height) above the slot's ``x``. The arm starts at
        :data:`PALM_START_IN_PELVIS` of the episode's pelvis. ``gate`` true routes the arm
        through an :class:`ArmTargetPipeline` on ``envelope`` (the committed right-palm envelope
        by default), so a target outside the pelvis-frame envelope shows up as ``arm_infeasible``;
        false leaves the arm unconstrained (it reaches anything at 0.3 m/s and 1 rad/s).
        ``stand_off_m`` is the pelvis distance from the counter edge, a knob of the reach-gap
        question of docs/status.md. The hand is a :class:`FakeHand` whose reach follows the
        geometry.
        """
        pelvis = pelvis_pose(spec, stand_off_m)
        palm = pose_compose(pelvis, PALM_START_IN_PELVIS)
        pipeline = (
            ArmTargetPipeline(envelope, ARM_LINEAR_SPEED_MPS, ARM_ANGULAR_SPEED_RPS)
            if gate
            else None
        )
        rack = FakeRack.from_layout(layout, spec.rack_slot, rack_frame)
        edge = Pose((float(rack.slot_entry.position[0]), 0.0, kd.COUNTER_HEIGHT_M))
        return cls(
            arm=FakeArm(palm, pipeline=pipeline, pelvis=pelvis),
            hand=FakeHand(),
            plate=FakePlate.from_spec(spec),
            rack=rack,
            counter_edge_pose=edge,
        )

    def observe(self) -> PlannerObservation:
        plate = Pose(
            self.plate.pose.position + self.plate_observation_offset, self.plate.pose.quat_wxyz
        )
        slot = Pose(
            self.rack.slot_entry.position + self.slot_observation_offset,
            self.rack.slot_entry.quat_wxyz,
        )
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
