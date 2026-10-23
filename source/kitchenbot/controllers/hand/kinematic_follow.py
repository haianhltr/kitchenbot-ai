"""Kinematic-follow hand: a labelled BASELINE, NOT A GRASP (docs/architecture.md section 8, M4a).

**Read this before quoting a result obtained with this hand.** ``KinematicFollowHand`` does not
close fingers on anything. On ``close()`` within 3 cm of the plate rim it switches the plate to a
kinematic body and from then on writes the plate's pose from the palm's pose every control step,
so the plate rides along rigidly no matter what a real hand could do. There is no finger-plate
contact, no friction, no force closure and no slip, and the plate does not react to the world
while attached (a kinematic body is moved through the rack tines, not stopped by them). Anything
measured with it is "the kinematic attach baseline" (the S2 target of at least 95 percent in
docs/evaluation.md section 5) and must be labelled that way; it says nothing about whether the
Dex3-1 hand, ``Dex3SynergyHand``, can grasp a plate. Per AGENTS.md section 7, never describe it as
a grasp controller result. Its purpose is to let the arm, the planner and the rack insertion be
developed and measured while the real grasp (#18) is still being tuned.

Pure logic, no simulator (the pattern of ``Dex3SynergyHand``). Sensing and writing are injected as
callables, so the Isaac Lab wrapper of #17 only has to supply them. All poses are world-frame
:class:`kitchenbot.geometry.Pose` objects (quaternions ``(w, x, y, z)``):

- ``palm_pose() -> Pose``: the palm point, see "Palm point" below.
- ``palm_velocity() -> (linear, angular)``: world-frame velocity of the palm point, m/s and rad/s,
  each a 3-sequence.
- ``plate_pose() -> Pose``: the plate's root pose, i.e. the centre of the disc, with local ``z`` the
  plate normal (the plate frame of ``kitchenbot.geometry``). This is NOT the grasp frame on the rim
  that the planner observes (docs/architecture.md section 9); the environment layer owns that
  conversion.
- ``set_plate_kinematic(bool)``: switch the plate between a kinematic and a dynamic rigid body.
- ``write_plate_pose(Pose)``: teleport the plate's root pose.
- ``set_plate_velocity(linear, angular)``: set the plate's velocity (called once, on release).

Attach rule (one behaviour, tested): ``close()`` attaches if and only if :func:`rim_distance` of
the palm position to the plate is at most ``attach_distance_m`` at the moment of the call. The
decision is made once, when the posture changes to closed. A ``close()`` that misses leaves the
posture closed with nothing attached and ``is_holding()`` false, which is what makes the planner's
grasp timeout tag ``grasp_miss``; neither a repeated ``close()`` nor ``update`` re-checks and
attaches later, because the planner sends ``close`` on every step and a silent late attach would
hide the miss. To try again, command ``open()`` or ``pre_grasp()`` first. The captured offset is
the plate pose in the palm frame, ``pose_compose(pose_inverse(palm), plate)``, and ``update(dt)``
writes ``pose_compose(palm, offset)`` while attached.

Release: ``open()`` or ``pre_grasp()`` while attached (any posture other than closed means the
plate is not held, like ``MockHand`` and the synergy hand) calls ``set_plate_kinematic(False)``
and then ``set_plate_velocity(*palm_velocity())``. The plate stays at the pose of the last
``update``. Leaving the closed posture while not attached only changes the posture.

Palm point: the palm point is the origin of the palm link whose pose ``palm_pose`` returns, with no
offset. #14's ``body_offset`` for the IK may move the controlled point away from that origin; if
it does, the callable must return the same point the 3 cm tolerance is meant for, or the rim
distance is measured from the wrong place.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from kitchenbot.controllers.hand.base import HandController, HandPosture
from kitchenbot.geometry import Pose, pose_compose, pose_inverse, to_frame

#: Attach tolerance, docs/architecture.md section 8: the palm within 3 cm of the rim.
DEFAULT_ATTACH_DISTANCE_M = 0.03

SIDES: tuple[str, ...] = ("left", "right")


def rim_distance(point, plate_pose: Pose, diameter_m: float) -> float:
    """Distance in metres from ``point`` to the plate's rim circle.

    The rim is the ring of radius ``diameter_m / 2`` in the plate plane, the local ``xy`` plane of
    ``plate_pose`` through its origin (the plate centre), with local ``z`` the plate normal. The
    point is expressed in the plate frame; with ``radial`` its distance from the plate axis and
    ``axial`` its height along it, the result is ``hypot(radial - radius, axial)``. So it is zero
    on the rim, the radius at the centre of the disc (the centre is far from the rim), and the
    height above or below the rim for a point over it. Both sides of the plate count alike.
    """
    if not (math.isfinite(diameter_m) and diameter_m > 0.0):
        raise ValueError(f"diameter_m must be positive and finite, got {diameter_m!r}")
    local = to_frame(Pose(point), plate_pose).position
    radial = math.hypot(float(local[0]), float(local[1]))
    return math.hypot(radial - 0.5 * diameter_m, float(local[2]))


class KinematicFollowHand(HandController):
    """The kinematic-attach BASELINE hand for one side. It is not a grasp; see the module docstring.

    Starts open and not attached; constructing it calls nothing. ``posture`` is the commanded
    posture. ``is_holding()`` is true exactly while the plate is attached. See the module
    docstring for the attach rule, the release, the frames and the palm point assumption.
    ``update(dt)`` only writes while attached; the follow is instantaneous, ``dt`` is validated
    and otherwise unused.
    """

    def __init__(
        self,
        side: str,
        *,
        palm_pose: Callable[[], Pose],
        palm_velocity: Callable[[], tuple[Sequence[float], Sequence[float]]],
        plate_pose: Callable[[], Pose],
        plate_diameter_m: float,
        set_plate_kinematic: Callable[[bool], None],
        write_plate_pose: Callable[[Pose], None],
        set_plate_velocity: Callable[[Sequence[float], Sequence[float]], None],
        attach_distance_m: float = DEFAULT_ATTACH_DISTANCE_M,
    ) -> None:
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        if not (math.isfinite(attach_distance_m) and attach_distance_m >= 0.0):
            raise ValueError(
                f"attach_distance_m must be non-negative and finite, got {attach_distance_m!r}"
            )
        self._side = side
        self._palm_pose = palm_pose
        self._palm_velocity = palm_velocity
        self._plate_pose = plate_pose
        self._set_plate_kinematic = set_plate_kinematic
        self._write_plate_pose = write_plate_pose
        self._set_plate_velocity = set_plate_velocity
        self._attach_distance_m = float(attach_distance_m)
        self._plate_diameter_m = self._checked_diameter(plate_diameter_m)
        self._posture = HandPosture.OPEN
        self._offset: Pose | None = None

    @staticmethod
    def _checked_diameter(diameter_m: float) -> float:
        if not (math.isfinite(diameter_m) and diameter_m > 0.0):
            raise ValueError(f"plate_diameter_m must be positive and finite, got {diameter_m!r}")
        return float(diameter_m)

    @property
    def side(self) -> str:
        return self._side

    @property
    def attach_distance_m(self) -> float:
        return self._attach_distance_m

    @property
    def plate_diameter_m(self) -> float:
        """Diameter used by the next ``close()``. Settable: it is randomized per episode."""
        return self._plate_diameter_m

    @plate_diameter_m.setter
    def plate_diameter_m(self, value: float) -> None:
        self._plate_diameter_m = self._checked_diameter(value)

    @property
    def offset(self) -> Pose | None:
        """The captured plate pose in the palm frame while attached, else ``None``."""
        return self._offset

    @property
    def posture(self) -> HandPosture:
        return self._posture

    def open(self) -> None:
        self._leave_closed(HandPosture.OPEN)

    def pre_grasp(self) -> None:
        self._leave_closed(HandPosture.PRE_GRASP)

    def close(self) -> None:
        """Command closed; attach iff the palm is within ``attach_distance_m`` of the rim now.

        Repeating ``close()`` while already commanded closed does nothing: it neither attaches
        after a miss nor recaptures the offset after a hit.
        """
        if self._posture is HandPosture.CLOSED:
            return
        palm = self._palm_pose()
        plate = self._plate_pose()
        if rim_distance(palm.position, plate, self._plate_diameter_m) <= self._attach_distance_m:
            offset = pose_compose(pose_inverse(palm), plate)
            self._set_plate_kinematic(True)
            self._offset = offset
        self._posture = HandPosture.CLOSED

    def is_holding(self) -> bool:
        return self._offset is not None

    def update(self, dt: float) -> None:
        if dt < 0.0:
            raise ValueError(f"dt must be non-negative, got {dt!r}")
        if self._offset is not None:
            self._write_plate_pose(pose_compose(self._palm_pose(), self._offset))

    def _leave_closed(self, posture: HandPosture) -> None:
        if self._offset is not None:
            self._set_plate_kinematic(False)
            self._set_plate_velocity(*self._palm_velocity())
            self._offset = None
        self._posture = posture
