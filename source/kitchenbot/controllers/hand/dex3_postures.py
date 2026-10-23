"""Dex3-1 hand postures: closing fractions per joint, resolved to radians through the URDF limits.

Each posture gives every hand joint a *closing fraction* f in [0, 1]: f = 0 is the joint's *open
end*, f = 1 its *closed end*. Both ends are joint limits from ``robots.joint_limits.DEX3_LIMITS``
and the target is the point a fraction f of the way from the open end to the closed end, so any
fraction in [0, 1] gives a target inside the limits. Every provisional number is in
``POSTURE_FRACTIONS``; #18 tunes them in simulation.

Open and closed ends per joint. Geometry is from the URDF joint origins (same file and commit as
``DEX3_LIMITS``), in the palm frame: fingers extend along +x, the index finger sits at
z = +0.0285 m and the middle finger at z = -0.0285 m, the left thumb points along -y and the right
thumb along +y.

- ``thumb_2``, ``middle_0``, ``middle_1``, ``index_0``, ``index_1`` (flexion, ``FLEXION_ROLES``):
  one limit is 0, the joint straight; that is the open end and the other limit is the closed end.
  Read from the limits, not chosen: the left fingers close toward negative angles and the left
  ``thumb_2`` toward positive, which curls the fingers toward -y and the thumb tip toward +x, so
  they meet in front of the palm. The right hand is the mirror image with every sign flipped.
- ``thumb_1`` (thumb base flexion, range crosses zero: left [-0.611, +1.047] rad, right mirrored):
  closes toward the same sign as ``thumb_2`` on the same hand, so the left one toward its upper
  limit. Both joints turn about z in series, and the longer half of the range is on that side.
- ``thumb_0`` (±1.047 rad on both hands): turns about y, the thumb's own long axis, so it does not
  close the thumb; it tilts the plane in which ``thumb_1`` and ``thumb_2`` curl it. At 0 rad,
  fraction 0.5, that plane is z = 0, midway between index and middle: a three-point pinch. So
  pre-grasp and closed hold it at 0.5 and only the open posture moves it, to its open end.

**Unverified until #18 runs in simulation:** the closing directions of ``thumb_0`` and ``thumb_1``
are reasoned from the URDF, not observed in a grasp. They live in ``LEFT_THUMB_CLOSING_SIGN`` and
nowhere else; flipping a value there flips that joint on both hands. With the current table the
``thumb_0`` sign only decides which way the open posture rotates the thumb about its long axis.

The right hand is the left mirrored through the palm's x-z plane (the URDF negates every y offset),
so a joint turning about z closes with the opposite sign on the right hand, and ``thumb_0``, turning
about y, with the same sign. This module imports only the standard library and ``kitchenbot``.
"""

from __future__ import annotations

from kitchenbot.controllers.hand.base import HandPosture
from kitchenbot.robots.joint_limits import DEX3_LIMITS

SIDES: tuple[str, ...] = ("left", "right")

#: Joint roles in ``joint_maps.HAND_LEFT`` / ``HAND_RIGHT`` order; every posture tuple uses it.
ROLES: tuple[str, ...] = (
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
)

#: Roles with one limit at 0 (straight): open end 0, closing direction read from the limits.
FLEXION_ROLES: tuple[str, ...] = ("thumb_2", "middle_0", "middle_1", "index_0", "index_1")

#: UNVERIFIED until #18 runs in simulation. Closing direction of the LEFT hand's two thumb joints
#: whose range crosses zero: +1 = the closed end is the upper limit, -1 = the lower limit. The right
#: hand follows by the mirror rule in ``closing_direction``. Flip a value here and nothing else.
LEFT_THUMB_CLOSING_SIGN: dict[str, int] = {"thumb_0": +1, "thumb_1": +1}

#: Roles that turn about the palm's y axis and so close with the same sign on both hands. Every
#: other role turns about z and closes with the opposite sign on the right hand.
_SAME_SIGN_ON_BOTH_HANDS = frozenset({"thumb_0"})

#: Closing fraction per posture, in ``ROLES`` order. Provisional (#42), tuned in simulation by #18.
#: Open: every joint at its open end. Pre-grasp: thumb opposed (0.5) and fingers part-way (0.35),
#: meant to leave a gap around a 10 mm plate rim; not measured. Closed: 0.8, short of the limits so
#: the position controller keeps squeezing against the rim instead of resting on a joint stop.
#: ``thumb_0`` stays at 0.5 (0 rad, the pinch plane) once the hand leaves open; see the docstring.
POSTURE_FRACTIONS: dict[HandPosture, tuple[float, ...]] = {
    #                      thumb_0 thumb_1 thumb_2 middle_0 middle_1 index_0 index_1
    HandPosture.OPEN:      (0.0,   0.0,    0.0,    0.0,     0.0,     0.0,    0.0),
    HandPosture.PRE_GRASP: (0.5,   0.5,    0.35,   0.35,    0.35,    0.35,   0.35),
    HandPosture.CLOSED:    (0.5,   0.8,    0.8,    0.8,     0.8,     0.8,    0.8),
}


def _check_side(side: str) -> str:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return side


def _check_role(role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    return role


def joint_name(side: str, role: str) -> str:
    """Full joint name, e.g. ``joint_name("right", "index_1") == "right_hand_index_1_joint"``."""
    return f"{_check_side(side)}_hand_{_check_role(role)}_joint"


def closing_direction(side: str, role: str) -> int:
    """+1 if the joint closes toward its upper limit, -1 if toward its lower limit."""
    name = joint_name(side, role)
    if role in LEFT_THUMB_CLOSING_SIGN:
        sign = LEFT_THUMB_CLOSING_SIGN[role]
        if sign not in (1, -1):
            raise ValueError(f"LEFT_THUMB_CLOSING_SIGN[{role!r}] must be +1 or -1, got {sign!r}")
        if side == "right" and role not in _SAME_SIGN_ON_BOTH_HANDS:
            sign = -sign
        return sign
    limit = DEX3_LIMITS[name]
    if limit.lower == 0.0:
        return 1
    if limit.upper == 0.0:
        return -1
    raise ValueError(f"{name} is not a flexion joint with one limit at 0: {limit}")


def closed_end(side: str, role: str) -> float:
    """The joint limit at fraction 1."""
    limit = DEX3_LIMITS[joint_name(side, role)]
    return limit.upper if closing_direction(side, role) > 0 else limit.lower


def open_end(side: str, role: str) -> float:
    """The joint limit at fraction 0."""
    limit = DEX3_LIMITS[joint_name(side, role)]
    return limit.lower if closing_direction(side, role) > 0 else limit.upper


def fraction_to_angle(side: str, role: str, fraction: float) -> float:
    """Joint target in radians for a closing fraction in [0, 1]."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"closing fraction must be in [0, 1], got {fraction!r}")
    start, end = open_end(side, role), closed_end(side, role)
    angle = (1.0 - fraction) * start + fraction * end
    return DEX3_LIMITS[joint_name(side, role)].clamp(angle)


def posture_targets(side: str, posture: HandPosture | str) -> tuple[float, ...]:
    """The 7 joint targets in radians of a posture, in ``joint_maps.HAND_<SIDE>`` order."""
    fractions = POSTURE_FRACTIONS[HandPosture(posture)]
    return tuple(
        fraction_to_angle(side, role, f) for role, f in zip(ROLES, fractions, strict=True)
    )
