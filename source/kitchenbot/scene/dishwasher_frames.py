"""Where the ArtVIP dishwasher, its lower rack and the slot row are, as numbers (#59, no simulator).

The dishwasher drafts (``scene/dishwasher_cfg.py``, ``scene/kitchen_scene_cfg.py``,
``scripts/smoke/load_kitchen.py``), the overrides generator
(``scripts/make_dishwasher_overrides.py``) and the tests read this module, so the slot poses the
planner gets, the tine attachment Isaac Lab authors and the composed USD that ``--check`` opens all
come from one calculation. Standard library only: nothing here imports ``pxr``, Isaac Sim, Isaac
Lab, omni or torch. Every number is a ``kitchen_dims`` constant.

Frames (see the ``kitchen_dims`` docstring)
-------------------------------------------
* kitchen frame: origin on the floor at the middle of the counter's front edge, ``x`` along the
  counter, ``y`` into the counter, ``z`` up. An Isaac Lab environment's origin is this origin.
* root-link frame: the frame of ``E_body_5``, the articulation root link; the origin of
  ``dishwasher_2_overrides.usda``. ``DISHWASHER_POSE_KITCHEN`` places it in the kitchen frame.
* lower-rack link frame: ``E_shelf_1_04``. At joint position ``q`` (metres, negative pulls the
  rack out) its origin is ``DISHWASHER_LOWER_RACK_LINK_M + (0, q, 0)`` in the root-link frame.
* rack frame: ``RackSlotLayout``'s frame and the origin of ``rack_tines.usda``, at
  ``RACK_TINES_POSE_IN_LINK`` in the lower-rack link.

Scale: the ArtVIP asset is scaled uniformly by ``scale`` (``RACK_SCALE``); the authored tines and
the plates are not (see ``kitchen_dims.rack_fits_plate``). The rack joint position ``q`` is taken
in world metres, which assumes PhysX does not scale prismatic limits with the asset; #11 checks it.

Robot: the G1 stands at ``(0, -ROBOT_STANDOFF_M)`` facing ``+y`` (``G1_29DOF_CFG.init_state.rot``
is a +90 deg yaw, Isaac Lab ``unitree.py:436-445``; docs/isaaclab_3_api_notes.md section 2) with
the pelvis at ``PELVIS_HEIGHT_M``. The pelvis frame has ``x`` forward, ``y`` to the robot's left,
``z`` up, so the robot's right is kitchen ``+x``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from kitchenbot.scene import kitchen_dims as kd

__all__ = [
    "ARTVIP_DISHWASHER_2",
    "BODY_LINK",
    "DOOR_DAMPING_PER_RAD",
    "DOOR_JOINT",
    "DOOR_LINK",
    "DOOR_STIFFNESS_PER_RAD",
    "LOWER_RACK_JOINT",
    "LOWER_RACK_LINK",
    "PELVIS_HEIGHT_M",
    "RACK_DAMPING",
    "RACK_STIFFNESS",
    "UPPER_RACK_JOINT",
    "UPPER_RACK_LINK",
    "footprints_kitchen",
    "kitchen_to_pelvis",
    "min_rack_travel",
    "plinth_box_kitchen",
    "rack_frame_in_root",
    "rack_tines_scale_in_link",
    "root_to_kitchen",
    "slot_layout",
    "slot_poses_kitchen",
    "slot_poses_root",
    "slot_row_from_pelvis",
    "yaw_quat_xyzw",
]

# --------------------------------------------------------------------------------------------
# ArtVIP dishwasher_2 names (#29 inspection; prim names as composed by usd-core)
# --------------------------------------------------------------------------------------------

#: The ArtVIP file, relative to ``scene/assets/third_party/`` (``MANIFEST.md``).
ARTVIP_DISHWASHER_2 = (
    "artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_2/model_dishwasher_2.usda"
)

#: Links (rigid bodies). #29 names the joints after these, but the joint prims, which are what
#: Isaac Lab's ``joint_names`` and the actuator regexes match, have their own names below.
BODY_LINK = "E_body_5"  # fixed to the world, ArticulationRootAPI
DOOR_LINK = "E_door_4"
UPPER_RACK_LINK = "E_shelf_03"
LOWER_RACK_LINK = "E_shelf_1_04"

DOOR_JOINT = "RevoluteJoint_dishwasher_2_middle"  # under E_door_4, axis X, 0..90 deg
UPPER_RACK_JOINT = "PrismaticJoint_dishwasher_2_up"  # under E_shelf_03, axis Y, -0.2..0 m
LOWER_RACK_JOINT = "PrismaticJoint_dishwasher_2_down"  # under E_shelf_1_04, axis Y, -0.2..0 m

# --------------------------------------------------------------------------------------------
# Drives (#29: UsdPhysics.DriveAPI stiffness and damping only, drive type "acceleration")
# --------------------------------------------------------------------------------------------

#: The USD angular drive is per degree; Isaac Lab's actuator gains are per radian
#: (``schemas_cfg.py:377``, ``:389`` convert the other way when writing USD), so the door's
#: k = 20 / deg and d = 10 / deg become these. The prismatic racks need no conversion.
DOOR_STIFFNESS_PER_RAD = 20.0 * 180.0 / math.pi
DOOR_DAMPING_PER_RAD = 10.0 * 180.0 / math.pi
RACK_STIFFNESS = 10.0  # per metre
RACK_DAMPING = 10.0  # per metre per second

#: G1 pelvis height at the default pose (``G1_29DOF_CFG.init_state.pos`` z, #46 notes section 2).
PELVIS_HEIGHT_M = 0.75

Vec3 = tuple[float, float, float]
Pose4 = tuple[float, float, float, float]


def yaw_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    """A yaw about ``z`` as Isaac Lab's (x, y, z, w) quaternion (``asset_base_cfg.py:42``)."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _rotate_yaw(x: float, y: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * x - s * y, s * x + c * y)


def _check_scale(scale: float) -> None:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be positive and finite, got {scale!r}")


def rack_tines_scale_in_link(scale: float = kd.RACK_SCALE) -> Vec3:
    """Local scale of the tines prim under the lower-rack link so the tines keep their size.

    It undoes the link's own x scale (``DISHWASHER_LOWER_RACK_LINK_SCALE_X``) and the asset scale,
    so the composed scale of every capsule is exactly 1 (PhysX capsules do not take a
    non-uniform scale).
    """
    _check_scale(scale)
    return (1.0 / (scale * kd.DISHWASHER_LOWER_RACK_LINK_SCALE_X), 1.0 / scale, 1.0 / scale)


def rack_frame_in_root(scale: float = kd.RACK_SCALE, rack_pos: float = 0.0) -> Vec3:
    """Origin of the rack frame in the root-link frame with the lower rack at ``rack_pos``."""
    _check_scale(scale)
    lx, ly, lz = kd.DISHWASHER_LOWER_RACK_LINK_M
    tx, ty, tz, _ = kd.RACK_TINES_POSE_IN_LINK
    return (
        scale * (lx + tx * kd.DISHWASHER_LOWER_RACK_LINK_SCALE_X),
        scale * (ly + ty) + rack_pos,
        scale * (lz + tz),
    )


def root_to_kitchen(p: Sequence[float], pose: Pose4 = kd.DISHWASHER_POSE_KITCHEN) -> Vec3:
    """A point given in the (already scaled) root-link frame, in the kitchen frame."""
    x, y, z, yaw = pose
    dx, dy = _rotate_yaw(float(p[0]), float(p[1]), yaw)
    return (x + dx, y + dy, z + float(p[2]))


def slot_layout(scale: float = kd.RACK_SCALE) -> kd.RackSlotLayout:
    """The tines on the scaled ArtVIP rack: scaled outline, authored pitch and height."""
    _check_scale(scale)
    return kd.RackSlotLayout(
        rack_width_m=kd.RACK_WIDTH_M * scale, rack_depth_m=kd.RACK_DEPTH_M * scale
    )


def slot_poses_root(scale: float = kd.RACK_SCALE, rack_pos: float = 0.0) -> tuple[Pose4, ...]:
    """``(x, y, z, yaw)`` of slots 1 to 5 in the root-link frame (yaw 0: the rack is not turned)."""
    ox, oy, oz = rack_frame_in_root(scale, rack_pos)
    return tuple((ox + x, oy + y, oz + z, yaw) for x, y, z, yaw in slot_layout(scale).slot_poses())


def slot_poses_kitchen(
    scale: float = kd.RACK_SCALE,
    rack_pos: float = -kd.DISHWASHER_RACK_TRAVEL_M,
    pose: Pose4 = kd.DISHWASHER_POSE_KITCHEN,
) -> tuple[Pose4, ...]:
    """``(x, y, z, yaw)`` of slots 1 to 5 in the kitchen frame: the planner's slot poses.

    ``yaw`` is the rack's yaw in the kitchen, so a seated plate's normal points along
    ``(cos yaw, sin yaw)``. Add an environment's origin for Isaac Lab world coordinates. The
    default ``rack_pos`` is the travel the pose assumes, not today's -0.2 m limit.
    """
    out = []
    for x, y, z, yaw in slot_poses_root(scale, rack_pos):
        kx, ky, kz = root_to_kitchen((x, y, z), pose)
        out.append((kx, ky, kz, pose[3] + yaw))
    return tuple(out)


def kitchen_to_pelvis(
    p: Sequence[float],
    standoff: float = kd.ROBOT_STANDOFF_M,
    pelvis_height: float = PELVIS_HEIGHT_M,
) -> Vec3:
    """A kitchen-frame point as (forward, left, up) from the pelvis of the robot at ``standoff``."""
    dx, dy, dz = float(p[0]), float(p[1]) + standoff, float(p[2]) - pelvis_height
    return (dy, -dx, dz)


def slot_row_from_pelvis(
    standoff: float = kd.ROBOT_STANDOFF_M,
    scale: float = kd.RACK_SCALE,
    rack_pos: float = -kd.DISHWASHER_RACK_TRAVEL_M,
    pose: Pose4 = kd.DISHWASHER_POSE_KITCHEN,
    pelvis_height: float = PELVIS_HEIGHT_M,
) -> tuple[tuple[float, float, float, float], ...]:
    """Per slot: (forward, left, up, distance) of the slot entry from the pelvis, metres."""
    out = []
    for x, y, z, _ in slot_poses_kitchen(scale, rack_pos, pose):
        f, left, u = kitchen_to_pelvis((x, y, z), standoff, pelvis_height)
        out.append((f, left, u, math.sqrt(f * f + left * left + u * u)))
    return tuple(out)


def min_rack_travel(
    scale: float = kd.RACK_SCALE,
    diameter_m: float = max(kd.PLATE_DIAMETERS_M),
    margin_m: float = 0.0,
) -> float:
    """Smallest pull-out at which a plate standing in the slot row is wholly in front of the body.

    In front of the front face (root-link ``y = 0``) nothing is above the plate, so it can be
    lowered in vertically; behind it the upper rack and the tub's top are in the way (the plate is
    taller than the 0.119 m between the racks at every scale up to 1.5).
    """
    if diameter_m <= 0.0 or margin_m < 0.0:
        raise ValueError("diameter_m must be positive and margin_m not negative")
    row_y = rack_frame_in_root(scale, 0.0)[1]
    return row_y + diameter_m / 2.0 + margin_m


def footprints_kitchen(
    scale: float = kd.RACK_SCALE,
    rack_pos: float = -kd.DISHWASHER_RACK_TRAVEL_M,
    pose: Pose4 = kd.DISHWASHER_POSE_KITCHEN,
) -> dict[str, tuple[tuple[tuple[float, float], ...], tuple[float, float]]]:
    """Kitchen-frame outlines of the body, the open door and the lower rack.

    Each value is ``(corners, (z_min, z_max))``: four ``(x, y)`` corners of the part's rectangle
    and its height range. The door is at ``DISHWASHER_DOOR_OPEN_DEG`` (90 deg, a flat plank in
    front of the front face); the rack at ``rack_pos``.
    """
    _check_scale(scale)
    s = scale
    bx, by, bz = kd.DISHWASHER_BODY_SIZE_M
    hx, hy, hz = kd.DISHWASHER_DOOR_HINGE_M
    t0, t1 = kd.DISHWASHER_DOOR_THICKNESS_M
    lx, ly, lz = kd.DISHWASHER_LOWER_RACK_LINK_M
    rack_h = kd.DISHWASHER_RACK_HEIGHT_M
    parts = {
        "body": ((0.0, bx), (0.0, by), (0.0, bz)),
        "door": (
            kd.DISHWASHER_DOOR_SPAN_X_M,
            (hy - kd.DISHWASHER_DOOR_LENGTH_M, hy),
            (hz + t0, hz + t1),
        ),
        "rack": ((lx, lx + kd.RACK_WIDTH_M), (ly, ly + kd.RACK_DEPTH_M), (lz, lz + rack_h)),
    }
    out = {}
    for name, ((x0, x1), (y0, y1), (z0, z1)) in parts.items():
        shift = rack_pos if name == "rack" else 0.0
        corners = []
        for cx, cy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            kx, ky, _ = root_to_kitchen((s * cx, s * cy + shift, 0.0), pose)
            corners.append((kx, ky))
        out[name] = (tuple(corners), (pose[2] + s * z0, pose[2] + s * z1))
    return out


def plinth_box_kitchen(
    scale: float = kd.RACK_SCALE, pose: Pose4 = kd.DISHWASHER_POSE_KITCHEN
) -> tuple[Vec3, Vec3]:
    """``(centre, size)`` of the axis-aligned plinth from the floor to the body's underside.

    The body outline must be axis-aligned in the kitchen, i.e. the yaw a multiple of 90 deg.
    """
    corners, _ = footprints_kitchen(scale, 0.0, pose)["body"]
    quarter = pose[3] / (math.pi / 2.0)
    if abs(quarter - round(quarter)) > 1e-9:
        raise ValueError(
            "the plinth is axis-aligned: the dishwasher yaw must be a multiple of 90 deg"
        )
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    height = pose[2]
    if height <= 0.0:
        raise ValueError("the dishwasher stands on the floor; there is no plinth")
    centre = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, height / 2.0)
    return centre, (max(xs) - min(xs), max(ys) - min(ys), height)
