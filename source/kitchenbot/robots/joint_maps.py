"""Joint groups and link names of the Unitree G1 (29-DoF body) with Dex3-1 hands.

**Provisional.** Every name here is derived from Unitree's URDF
``robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf`` (``unitreerobotics/unitree_ros``,
BSD-3-Clause, commit and date recorded in ``tests/fixtures/g1_dex3_joints_urdf.json``, issue #28).
Issue #3 verifies this map against the Isaac Sim USD (``Robots/Unitree/G1/g1.usd``) and turns any
difference into a fix here. Until then treat the names, and the order within each group, as
unconfirmed: the USD may list joints in a different order than the URDF file does.

Order inside every group is the order of the ``<joint>`` elements in the URDF file. For the hands
that is thumb 0, 1, 2, then middle 0, 1, then index 0, 1. The left and right arm and hand lists
follow the same pattern, so ``ARM_LEFT[i]`` and ``ARM_RIGHT[i]`` are mirror joints.

Movable joints are the revolute, prismatic and continuous ones; the URDF's fixed joints (sensor
mounts, ``*_hand_palm_joint`` and so on) are not in any group. The six groups partition the 43
movable joints: 12 + 3 + 7 + 7 + 7 + 7.

Each group has a name tuple and a compiled regex. The regexes are anchored with ``^`` and ``$`` so
``match``, ``search`` and ``fullmatch`` agree, and ``.pattern`` can be handed to an Isaac Lab action
term's ``joint_names``. This module imports only the standard library.
"""

from __future__ import annotations

import re

LEGS: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

WAIST: tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

ARM_LEFT: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)

ARM_RIGHT: tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

HAND_LEFT: tuple[str, ...] = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
)

HAND_RIGHT: tuple[str, ...] = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
)

LEGS_RE = re.compile(r"^(left|right)_(hip_(pitch|roll|yaw)|knee|ankle_(pitch|roll))_joint$")
WAIST_RE = re.compile(r"^waist_(yaw|roll|pitch)_joint$")
ARM_LEFT_RE = re.compile(r"^left_(shoulder_(pitch|roll|yaw)|elbow|wrist_(roll|pitch|yaw))_joint$")
ARM_RIGHT_RE = re.compile(r"^right_(shoulder_(pitch|roll|yaw)|elbow|wrist_(roll|pitch|yaw))_joint$")
HAND_LEFT_RE = re.compile(r"^left_hand_(thumb_[012]|middle_[01]|index_[01])_joint$")
HAND_RIGHT_RE = re.compile(r"^right_hand_(thumb_[012]|middle_[01]|index_[01])_joint$")

#: Joint names per group, keyed by group name.
GROUPS: dict[str, tuple[str, ...]] = {
    "legs": LEGS,
    "waist": WAIST,
    "arm_left": ARM_LEFT,
    "arm_right": ARM_RIGHT,
    "hand_left": HAND_LEFT,
    "hand_right": HAND_RIGHT,
}

#: Compiled regex per group, same keys as ``GROUPS``.
GROUP_REGEX: dict[str, re.Pattern[str]] = {
    "legs": LEGS_RE,
    "waist": WAIST_RE,
    "arm_left": ARM_LEFT_RE,
    "arm_right": ARM_RIGHT_RE,
    "hand_left": HAND_LEFT_RE,
    "hand_right": HAND_RIGHT_RE,
}

#: All 43 movable joints: legs, waist, left arm, right arm, left hand, right hand.
ALL_JOINTS: tuple[str, ...] = tuple(name for names in GROUPS.values() for name in names)

#: Last link of each arm before the hand (the child of the wrist yaw joint).
WRIST_LINKS: dict[str, str] = {
    "left": "left_wrist_yaw_link",
    "right": "right_wrist_yaw_link",
}

#: Dex3 palm link of each hand, fixed to the wrist yaw link; the finger joints hang off it.
PALM_LINKS: dict[str, str] = {
    "left": "left_hand_palm_link",
    "right": "right_hand_palm_link",
}


def group_of(joint_name: str) -> str | None:
    """Return the group a movable joint name belongs to, or ``None`` if it is in no group."""
    for group, regex in GROUP_REGEX.items():
        if regex.fullmatch(joint_name):
            return group
    return None
