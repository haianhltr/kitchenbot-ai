"""Forward kinematics of a serial URDF chain, built from a joint dump. numpy only, no simulator.

**Provisional**, like ``joint_maps``: the chain is built from the joint list that
``scripts/smoke/dump_urdf_joints.py`` writes (a list of dicts with ``name``, ``type``, ``parent``,
``child``, ``origin_xyz``, ``origin_rpy``, ``axis``, ``lower``, ``upper``), today the dump of
Unitree's ``g1_29dof_with_hand_rev_1_0.urdf`` committed as
``tests/fixtures/g1_dex3_joints_urdf.json`` (#28, origins added in #48). Issue #3 checks the chain
against the Isaac Sim USD.

The joint list is an argument, never read from ``tests/`` by this package: the chain is only
needed offline (the reachability envelope script and the tests), the fixture stays the one copy of
the URDF facts, and the same functions take #3's USD dump unchanged once it exists.

URDF convention: a joint's ``<origin xyz rpy>`` is the transform from the parent link frame to the
joint frame, which is also the child link frame; ``rpy`` is fixed-axis roll, pitch, yaw, so the
rotation is ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``. The joint motion (rotation about ``axis`` for
revolute and continuous joints, translation along it for prismatic ones) is applied after the
origin, in the child frame. Fixed joints contribute their origin only, and they are part of the
chain: the right palm hangs off the wrist yaw link through the fixed ``right_hand_palm_joint``.

Quaternions are ``(w, x, y, z)`` with ``w >= 0`` (:func:`kitchenbot.geometry.matrix_to_quat`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from kitchenbot.geometry import matrix_to_quat
from kitchenbot.robots import joint_maps

__all__ = [
    "ChainJoint",
    "UrdfChain",
    "right_arm_chain",
    "rpy_to_matrix",
]

_ROTARY = ("revolute", "continuous")
_SUPPORTED = (*_ROTARY, "prismatic", "fixed")


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF fixed-axis roll, pitch, yaw to a 3x3 rotation matrix ``Rz(y) @ Ry(p) @ Rx(r)``."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


@dataclass(frozen=True, eq=False)
class ChainJoint:
    """One joint of a chain: its origin transform, motion axis and limits."""

    name: str
    type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rot: np.ndarray
    axis: np.ndarray | None
    lower: float | None
    upper: float | None

    @property
    def movable(self) -> bool:
        return self.type != "fixed"

    @classmethod
    def from_dump(cls, joint: Mapping) -> ChainJoint:
        jtype = joint["type"]
        if jtype not in _SUPPORTED:
            raise ValueError(f"joint {joint['name']!r}: unsupported type {jtype!r}")
        if "origin_xyz" not in joint or "origin_rpy" not in joint:
            raise ValueError(
                f"joint {joint['name']!r} has no origin; regenerate the dump with "
                "scripts/smoke/dump_urdf_joints.py from #48 or later"
            )
        axis = None
        if jtype != "fixed":
            axis = np.asarray(joint["axis"], dtype=np.float64).reshape(3)
            axis = axis / np.linalg.norm(axis)
        return cls(
            name=joint["name"],
            type=jtype,
            parent=joint["parent"],
            child=joint["child"],
            origin_xyz=np.asarray(joint["origin_xyz"], dtype=np.float64).reshape(3),
            origin_rot=rpy_to_matrix(joint["origin_rpy"]),
            axis=axis,
            lower=joint.get("lower"),
            upper=joint.get("upper"),
        )


def _axis_rotations(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Batched Rodrigues formula: ``(N, 3, 3)`` rotations by ``angles`` about the unit ``axis``."""
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
    )
    s = np.sin(angles)[:, None, None]
    c = np.cos(angles)[:, None, None]
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


class UrdfChain:
    """The serial chain of joints from ``base_link`` to ``tip_link``, fixed joints included.

    ``joint_names`` lists the chain's movable joints from base to tip. ``active_joints`` are the
    ones :meth:`fk` takes in ``q``, in the order given (all movable joints by default); the
    remaining movable joints, ``held_joints`` in chain order, are inputs too but default to zero.
    """

    def __init__(
        self,
        joints: Sequence[ChainJoint],
        base_link: str,
        tip_link: str,
        active: Sequence[str] | None = None,
    ) -> None:
        self.joints: tuple[ChainJoint, ...] = tuple(joints)
        self.base_link = base_link
        self.tip_link = tip_link
        self.joint_names: tuple[str, ...] = tuple(j.name for j in self.joints if j.movable)
        self.active_joints: tuple[str, ...] = self.joint_names if active is None else tuple(active)
        unknown = [n for n in self.active_joints if n not in self.joint_names]
        if unknown or len(set(self.active_joints)) != len(self.active_joints):
            raise ValueError(f"active joints must be distinct movable chain joints: {unknown}")
        self.held_joints: tuple[str, ...] = tuple(
            n for n in self.joint_names if n not in self.active_joints
        )
        by_name = {j.name: j for j in self.joints}
        self.lower = np.array([by_name[n].lower for n in self.active_joints], dtype=np.float64)
        self.upper = np.array([by_name[n].upper for n in self.active_joints], dtype=np.float64)

    @classmethod
    def from_joint_dump(
        cls,
        joints: Sequence[Mapping],
        base_link: str,
        tip_link: str,
        active: Sequence[str] | None = None,
    ) -> UrdfChain:
        """Walk the dump's joints from ``tip_link`` up to ``base_link`` and build the chain."""
        by_child: dict[str, Mapping] = {}
        for joint in joints:
            if joint["child"] in by_child:
                raise ValueError(f"link {joint['child']!r} is the child of two joints")
            by_child[joint["child"]] = joint
        chain: list[ChainJoint] = []
        link = tip_link
        while link != base_link:
            if link not in by_child:
                raise ValueError(f"no chain from {base_link!r} to {tip_link!r}: stuck at {link!r}")
            joint = by_child[link]
            chain.append(ChainJoint.from_dump(joint))
            link = joint["parent"]
        chain.reverse()
        return cls(chain, base_link, tip_link, active)

    @property
    def dof(self) -> int:
        """Number of active joints, the length of ``q``."""
        return len(self.active_joints)

    def _all_positions(self, q: np.ndarray, held: Sequence[float] | None) -> dict[str, np.ndarray]:
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != self.dof:
            raise ValueError(f"q must have shape (N, {self.dof}), got {q.shape}")
        values = {name: q[:, i] for i, name in enumerate(self.active_joints)}
        held_q = np.zeros(len(self.held_joints)) if held is None else np.asarray(held, float)
        if held_q.shape != (len(self.held_joints),):
            raise ValueError(f"held must have {len(self.held_joints)} values {self.held_joints}")
        for name, value in zip(self.held_joints, held_q, strict=True):
            values[name] = np.full(q.shape[0], float(value))
        return values

    def fk_batch(
        self, q: np.ndarray, held: Sequence[float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tip positions ``(N, 3)`` and rotation matrices ``(N, 3, 3)`` in the base frame.

        ``q`` has shape ``(N, dof)`` in ``active_joints`` order; ``held`` gives the
        ``held_joints`` positions shared by all ``N`` rows (zeros if ``None``).
        """
        values = self._all_positions(q, held)
        n = np.asarray(q).shape[0]
        rot = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        pos = np.zeros((n, 3))
        for joint in self.joints:
            pos = pos + rot @ joint.origin_xyz
            rot = rot @ joint.origin_rot
            if joint.type in _ROTARY:
                rot = rot @ _axis_rotations(joint.axis, values[joint.name])
            elif joint.type == "prismatic":
                pos = pos + (rot @ joint.axis) * values[joint.name][:, None]
        return pos, rot

    def fk(self, q: Sequence[float], held: Sequence[float] | None = None):
        """``(position, quat_wxyz)`` of the tip link frame in the base frame for one ``q``."""
        q_row = np.asarray(q, dtype=np.float64).reshape(1, -1)
        pos, rot = self.fk_batch(q_row, held)
        return pos[0], matrix_to_quat(rot[0])


def right_arm_chain(joints: Sequence[Mapping]) -> UrdfChain:
    """The right arm from ``pelvis`` to ``right_hand_palm_link``, through the waist.

    ``q`` is the seven ``joint_maps.ARM_RIGHT`` joints in that order; the three
    ``joint_maps.WAIST`` joints are ``held`` inputs that default to zero, because the balance
    policy, not the arm controller, owns the waist (docs/architecture.md section 5). ``joints`` is
    a dump's ``"joints"`` list, e.g. ``tests/fixtures/g1_dex3_joints_urdf.json``.
    """
    chain = UrdfChain.from_joint_dump(
        joints, "pelvis", joint_maps.PALM_LINKS["right"], active=joint_maps.ARM_RIGHT
    )
    if chain.held_joints != joint_maps.WAIST:
        raise ValueError(f"expected the waist between pelvis and arm, got {chain.held_joints}")
    return chain
