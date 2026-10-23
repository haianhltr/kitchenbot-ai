"""Joint limits of the two Dex3-1 hands of the Unitree G1.

**Provisional**, like ``joint_maps``. Every value is the ``<limit>`` element of Unitree's URDF
``robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf`` (``unitreerobotics/unitree_ros``,
BSD-3-Clause, commit and date recorded in ``tests/fixtures/g1_dex3_joints_urdf.json``, issue #28),
copied as written there; ``tests/test_dex3_postures.py`` asserts equality with that fixture.
Issue #3 checks the Isaac Sim USD and turns any difference into a fix here.

Units: ``lower`` and ``upper`` in rad, ``effort`` in N·m, ``velocity`` in rad/s.

The right hand is the left hand mirrored through the palm's x-z plane (the URDF negates every y
offset). Joints that turn about the palm's z axis (all but ``thumb_0``) therefore have sign-mirrored
limits, and ``thumb_0``, which turns about y, has the same limits on both hands. This module imports
only the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointLimit:
    """Position range and actuator limits of one revolute joint."""

    lower: float
    upper: float
    effort: float
    velocity: float

    def contains(self, position: float) -> bool:
        """True if ``position`` lies in ``[lower, upper]``."""
        return self.lower <= position <= self.upper

    def clamp(self, position: float) -> float:
        """``position`` limited to ``[lower, upper]``."""
        return min(max(position, self.lower), self.upper)


#: The 14 Dex3 joints in ``joint_maps.HAND_LEFT + joint_maps.HAND_RIGHT`` order.
#: Columns: lower (rad), upper (rad), effort (N·m), velocity (rad/s).
DEX3_LIMITS: dict[str, JointLimit] = {
    "left_hand_thumb_0_joint": JointLimit(-1.04719755, 1.04719755, 2.45, 3.14),
    "left_hand_thumb_1_joint": JointLimit(-0.61086523, 1.04719755, 1.4, 12.0),
    "left_hand_thumb_2_joint": JointLimit(0.0, 1.74532925, 1.4, 12.0),
    "left_hand_middle_0_joint": JointLimit(-1.57079632, 0.0, 1.4, 12.0),
    "left_hand_middle_1_joint": JointLimit(-1.74532925, 0.0, 1.4, 12.0),
    "left_hand_index_0_joint": JointLimit(-1.57079632, 0.0, 1.4, 12.0),
    "left_hand_index_1_joint": JointLimit(-1.74532925, 0.0, 1.4, 12.0),
    "right_hand_thumb_0_joint": JointLimit(-1.04719755, 1.04719755, 2.45, 3.14),
    "right_hand_thumb_1_joint": JointLimit(-1.04719755, 0.61086523, 1.4, 12.0),
    "right_hand_thumb_2_joint": JointLimit(-1.74532925, 0.0, 1.4, 12.0),
    "right_hand_middle_0_joint": JointLimit(0.0, 1.57079632, 1.4, 12.0),
    "right_hand_middle_1_joint": JointLimit(0.0, 1.74532925, 1.4, 12.0),
    "right_hand_index_0_joint": JointLimit(0.0, 1.57079632, 1.4, 12.0),
    "right_hand_index_1_joint": JointLimit(0.0, 1.74532925, 1.4, 12.0),
}
