"""The plate grasp convention (#67), plus a re-export of :mod:`kitchenbot.geometry`.

Re-export
---------
The poses and quaternion helpers moved to :mod:`kitchenbot.geometry` in #48 so that planning and
the arm controllers share them; every name there is still importable from here, so
``from kitchenbot.planning.geometry import ...`` keeps working for #43's planner tests. New code
that needs only those helpers imports :mod:`kitchenbot.geometry`.

The grasp frame (decided in #67)
--------------------------------
The planner observes the plate as its **grasp frame** (``PlannerObservation.plate_pose``,
docs/architecture.md section 9); the judges and the simulator hold the plate **body** pose (#30:
origin at the centre of the underside, local ``z`` the plate normal). The rule below is the only
place that connects the two. The fakes, the dry run, the kinematic-follow wiring and the tests call
these functions; nothing else restates the rule.

- **Point: the top rim point nearest the robot.** The plate is a solid of revolution, so its yaw
  about its own normal is free. In the body frame turned so that its ``+x`` points toward the
  robot (:func:`approach_body_pose`), the point is ``(+D/2, 0, PLATE_HEIGHT_M)``: a radius out
  toward the robot, at the top of the rim. That is the outer top corner of the profile of
  ``scripts/make_plate_usd.py`` (radius ``D/2``, 20 mm up), the edge the hand closes on from
  above.
- **Orientation.** ``z`` is the plate normal and ``x`` points from the rim point toward the plate
  centre, away from the robot. So the grasp frame is the turned body frame shifted to that point
  and given a half turn about ``z``: :func:`grasp_in_body` is
  ``Pose((D/2, 0, PLATE_HEIGHT_M), yaw pi)``.
- **Toward the robot** is a world-frame direction, projected into the plate's plane. In the kitchen
  frame of :mod:`kitchenbot.scene.kitchen_dims` the robot stands in front of the counter's front
  edge facing ``+y``, so the default :data:`TOWARD_ROBOT_KITCHEN` is ``(0, -1, 0)``: the rim point
  nearest the counter's front edge. It is the point #56's reach analysis used
  (``scripts/reach_gap_analysis.py``, ``near_rim``). It is not the point nearest the pelvis: the
  pelvis noise and the plate's lateral offset do not turn it.

Consequences, each derived from the rule rather than restated:

1. **The quarter turn to vertical.** :func:`kitchenbot.geometry.plate_vertical_quat` turns the
   frame a quarter turn about its own ``y``, which tips its ``x`` straight down. The plate centre
   lies along the grasp frame's ``+x``, so it ends up a radius *below* the grasp point: the grasp
   point is the top of the vertical plate and the hand stays above the tines. The normal ends up
   along the slot's ``+x``.
2. **The slot entry.** The planner's ``slot_pose`` is the grasp frame when the plate's lowest point
   is at the top of the tines, centred in the gap (docs/architecture.md section 9).
   :func:`slot_entry_grasp_pose` derives it from ``RackSlotLayout.slot_pose(i)``, which keeps its
   meaning (gap centre at the tine tops). It works out to one diameter above the tine tops and
   ``PLATE_HEIGHT_M / 2`` along the slot's ``x``, because the grasp point is on the plate's top
   face and the plate's mid-plane is half a thickness behind it. ``PlannerConfig.insertion_depth_m``
   keeps its meaning: the grasp frame descending that far below its slot entry puts the plate's
   lowest point that far below the tine tops.
3. **The kinematic-follow attach.** ``KinematicFollowHand`` measures
   :func:`~kitchenbot.controllers.hand.kinematic_follow.rim_distance` to the rim circle in the
   plane of the pose it is given. Given the plate's root (body) pose, that circle lies in the
   underside plane, so a palm exactly at the grasp point is ``PLATE_HEIGHT_M`` = 2 cm from it.
   That is inside the 3 cm attach distance with 1 cm to spare, and a palm sitting low attaches
   more easily than one sitting high. #17 decides whether the wrapper passes the root pose or a
   pose raised to the top rim.
4. **The judges.** ``judges.plate_in_slot`` tests the body pose's lowest rim point on the
   mid-plane. With the grasp frame at its slot entry, that point is at the tine tops, centred
   (``across = along = rim_bottom = 0``, not yet in the slot). After ``insertion_depth_m`` it is
   inside the slot volume.

``tests/test_planner_hand_contract.py`` asserts the rule and the four consequences in one place.
numpy only; no simulator import.
"""

from __future__ import annotations

import math

import numpy as np

from kitchenbot.geometry import *  # noqa: F403
from kitchenbot.geometry import (
    Pose,
    plate_vertical_quat,
    pose_compose,
    pose_inverse,
    quat_apply,
    quat_conjugate,
    quat_from_yaw,
    quat_multiply,
)
from kitchenbot.geometry import __all__ as _GEOMETRY_ALL
from kitchenbot.scene.kitchen_dims import PLATE_HEIGHT_M, RackSlotLayout

__all__ = [
    *_GEOMETRY_ALL,
    "GRASP_RIM_HEIGHT_M",
    "TOWARD_ROBOT_KITCHEN",
    "approach_body_pose",
    "body_pose_from_grasp",
    "grasp_in_body",
    "grasp_pose_from_body",
    "slot_entry_grasp_pose",
]

#: Height of the grasp point above the plate's underside, body frame: the top of the rim (#30's
#: 20 mm plate height), because the hand closes on the top rim (#67).
GRASP_RIM_HEIGHT_M: float = PLATE_HEIGHT_M

#: The horizontal direction from the plate toward the robot in the kitchen frame of
#: ``kitchen_dims``: the robot stands in front of the counter's front edge and faces ``+y``.
TOWARD_ROBOT_KITCHEN: tuple[float, float, float] = (0.0, -1.0, 0.0)


def _radius(diameter_m: float) -> float:
    diameter_m = float(diameter_m)
    if not math.isfinite(diameter_m) or diameter_m <= 0.0:
        raise ValueError(f"diameter_m must be a positive finite number, got {diameter_m!r}")
    return diameter_m / 2.0


def grasp_in_body(diameter_m: float) -> Pose:
    """The grasp frame in the approach body frame (:func:`approach_body_pose`), see the docstring.

    ``Pose((D/2, 0, GRASP_RIM_HEIGHT_M), yaw pi)``: the top rim point on the body's ``+x`` side
    (toward the robot), with ``x`` pointing back at the plate centre and ``z`` the plate normal.
    """
    return Pose((_radius(diameter_m), 0.0, GRASP_RIM_HEIGHT_M), quat_from_yaw(math.pi))


def approach_body_pose(body: Pose, toward_robot=TOWARD_ROBOT_KITCHEN) -> Pose:
    """The same plate body pose turned about its own normal so that its ``+x`` faces the robot.

    ``toward_robot`` is a world-frame direction (need not be unit length). Its component in the
    plate's plane gives the turn; the plate is a solid of revolution, so the result is the same
    plate. Raises ``ValueError`` if the direction is along the plate normal (no nearest rim point).
    """
    toward = np.asarray(toward_robot, dtype=np.float64).reshape(3)
    local = quat_apply(quat_conjugate(body.quat_wxyz), toward)
    in_plane = math.hypot(float(local[0]), float(local[1]))
    if not math.isfinite(in_plane) or in_plane < 1e-9 * max(1.0, float(np.linalg.norm(toward))):
        raise ValueError(
            f"toward_robot {tuple(toward)} is along the plate normal: no rim point is nearest"
        )
    turn = math.atan2(float(local[1]), float(local[0]))
    return Pose(body.position, quat_multiply(body.quat_wxyz, quat_from_yaw(turn)))


def grasp_pose_from_body(
    body: Pose, diameter_m: float, toward_robot=TOWARD_ROBOT_KITCHEN
) -> Pose:
    """The grasp frame of a plate body pose: the top rim point nearest the robot (docstring).

    Independent of the body's yaw about its normal, as it should be for a solid of revolution.
    """
    return pose_compose(approach_body_pose(body, toward_robot), grasp_in_body(diameter_m))


def body_pose_from_grasp(grasp: Pose, diameter_m: float) -> Pose:
    """The plate body pose of a grasp frame, in the approach body frame (``+x`` toward the robot).

    Inverse of :func:`grasp_pose_from_body` up to the body's yaw about its normal, which the grasp
    frame does not record (the plate looks the same at every yaw). The judges do not depend on that
    yaw.
    """
    return pose_compose(grasp, pose_inverse(grasp_in_body(diameter_m)))


def slot_entry_grasp_pose(
    layout: RackSlotLayout,
    slot_index: int,
    diameter_m: float,
    rack_frame: Pose | None = None,
    *,
    thickness_m: float = PLATE_HEIGHT_M,
) -> Pose:
    """The planner's ``slot_pose``: the grasp frame when the plate stands at the slot entry.

    "At the slot entry" means vertical (:func:`plate_vertical_quat` of the slot yaw, the planner's
    ``aligned`` orientation), with the lowest point of its rim circle on the mid-plane at
    ``layout.slot_pose(slot_index)``: the gap centre at the tine tops. The mid-plane is
    ``thickness_m / 2`` along the normal from the body origin, as in ``judges.plate_in_slot``. The
    offset from that point to the grasp point is computed from :func:`body_pose_from_grasp`, not
    hard-coded: it is ``(thickness_m / 2, 0, D)`` in the slot frame with the default thickness.
    The result is in the rack frame, or in the frame ``rack_frame`` is given in (the world) if
    ``rack_frame`` is the rack frame's pose. Its orientation is the slot yaw only, as the planner
    reads it.
    """
    sx, sy, sz, yaw = layout.slot_pose(slot_index)  # validates the index
    vertical = Pose((0.0, 0.0, 0.0), plate_vertical_quat(yaw))
    body = body_pose_from_grasp(vertical, diameter_m)
    normal = quat_apply(body.quat_wxyz, (0.0, 0.0, 1.0))
    mid_centre = body.position + 0.5 * float(thickness_m) * normal
    lowest = mid_centre - np.array([0.0, 0.0, _radius(diameter_m)])  # the normal is horizontal
    entry = Pose(np.array([sx, sy, sz]) - lowest, quat_from_yaw(yaw))
    return entry if rack_frame is None else pose_compose(rack_frame, entry)
