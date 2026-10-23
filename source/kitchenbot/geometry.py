"""Poses and quaternion helpers shared by planning and controllers. numpy only, no simulator import.

Created for the task planner in #43 as ``kitchenbot.planning.geometry`` and moved here in #48 so
the arm controllers can use it without importing the planner; ``kitchenbot.planning.geometry``
re-exports every name. ``kitchenbot.scene`` imports nothing from here.

Quaternions are ``(w, x, y, z)``, this repository's order (Isaac Lab 3.0 itself stores ``(x, y, z,
w)``, see docs/isaaclab_3_api_notes.md; convert at the simulator boundary), and are treated as unit
rotations: every helper normalises its inputs. The planner keeps all its poses in one frame (the
world). The frame helpers added in #48 move poses between frames for the arm pipeline: a ``Pose`` is
read as the transform from its own frame to the frame it is expressed in, :func:`pose_compose`
chains two such transforms, :func:`pose_inverse` reverses one, and :func:`to_frame` expresses a
world pose in a moving frame such as the pelvis (docs/architecture.md section 5, "Frames").

Frame conventions used by :mod:`kitchenbot.planning.plate_to_rack`:

- plate frame: local ``z`` is the plate normal, so a plate lying flat on the counter has its ``z``
  axis pointing up;
- slot frame: yaw only; local ``x`` is the direction a seated plate's normal points, i.e. across
  the tines. A plate is vertical in the slot when its normal lies along the slot's ``x`` axis,
  which :func:`plate_vertical_quat` builds from the slot yaw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Pose",
    "matrix_to_quat",
    "plate_vertical_quat",
    "pose_compose",
    "pose_inverse",
    "position_distance",
    "quat_angle",
    "quat_apply",
    "quat_conjugate",
    "quat_from_axis_angle",
    "quat_from_yaw",
    "quat_multiply",
    "quat_slerp",
    "quat_to_matrix",
    "tilt_from_vertical",
    "to_frame",
    "yaw_from_quat",
]

_IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _unit_quat(q: np.ndarray | tuple | list) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64).reshape(-1)
    if arr.shape != (4,):
        raise ValueError(f"quaternion must have 4 elements (w, x, y, z), got shape {arr.shape}")
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"quaternion must be finite and non-zero, got {arr}")
    return arr / norm


def quat_multiply(a, b) -> np.ndarray:
    """Hamilton product ``a * b`` (apply ``b`` first, then ``a``), both ``(w, x, y, z)``."""
    aw, ax, ay, az = _unit_quat(a)
    bw, bx, by, bz = _unit_quat(b)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conjugate(q) -> np.ndarray:
    """Inverse rotation of a unit quaternion."""
    u = _unit_quat(q)
    return np.array([u[0], -u[1], -u[2], -u[3]])


def quat_apply(q, v) -> np.ndarray:
    """Rotate the 3-vector ``v`` by ``q``."""
    vec = np.asarray(v, dtype=np.float64).reshape(3)
    u = _unit_quat(q)
    w, r = u[0], u[1:]
    t = 2.0 * np.cross(r, vec)
    return vec + w * t + np.cross(r, t)


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    """Rotation of ``angle`` radians about ``axis`` (need not be unit length)."""
    ax = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(ax))
    if norm < 1e-12:
        raise ValueError("rotation axis must be non-zero")
    half = 0.5 * float(angle)
    return np.concatenate(([math.cos(half)], math.sin(half) * ax / norm))


def quat_from_yaw(yaw: float) -> np.ndarray:
    """Rotation of ``yaw`` radians about the world ``z`` axis."""
    return quat_from_axis_angle((0.0, 0.0, 1.0), yaw)


def yaw_from_quat(q) -> float:
    """Heading of the rotated ``x`` axis in the horizontal plane, radians in ``(-pi, pi]``."""
    w, x, y, z = _unit_quat(q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_angle(a, b) -> float:
    """Angle in radians, in ``[0, pi]``, of the rotation that takes ``a`` to ``b``."""
    r = quat_multiply(quat_conjugate(a), b)
    return 2.0 * math.atan2(float(np.linalg.norm(r[1:])), abs(float(r[0])))


def quat_slerp(a, b, t: float) -> np.ndarray:
    """Spherical interpolation from ``a`` (``t = 0``) to ``b`` (``t = 1``) along the short arc."""
    qa, qb = _unit_quat(a), _unit_quat(b)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb, dot = -qb, -dot
    if dot > 1.0 - 1e-12:
        return _unit_quat(qa + t * (qb - qa))
    theta = math.acos(min(dot, 1.0))
    s = math.sin(theta)
    return _unit_quat((math.sin((1.0 - t) * theta) * qa + math.sin(t * theta) * qb) / s)


def position_distance(a, b) -> float:
    """Euclidean distance between two 3-vectors."""
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def quat_to_matrix(q) -> np.ndarray:
    """3x3 rotation matrix of ``q``: ``quat_to_matrix(q) @ v == quat_apply(q, v)``."""
    w, x, y, z = _unit_quat(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(m) -> np.ndarray:
    """Unit quaternion ``(w, x, y, z)`` with ``w >= 0`` of the 3x3 rotation matrix ``m``.

    Shepperd's method: the largest of ``w``, ``x``, ``y``, ``z`` is computed from the trace or a
    diagonal entry, so the result stays accurate near half-turns. ``m`` must be a rotation matrix
    (orthonormal, determinant +1, checked to 1e-6).
    """
    r = np.asarray(m, dtype=np.float64)
    if r.shape != (3, 3) or not np.all(np.isfinite(r)):
        raise ValueError(f"rotation matrix must be 3x3 and finite, got shape {r.shape}")
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-6) or np.linalg.det(r) < 0.0:
        raise ValueError("matrix is not a rotation (orthonormal with determinant +1)")
    trace = float(np.trace(r))
    candidates = (trace, r[0, 0], r[1, 1], r[2, 2])
    k = int(np.argmax(candidates))
    if k == 0:
        s = 2.0 * math.sqrt(1.0 + trace)
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif k == 1:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif k == 2:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    out = _unit_quat(q)
    return -out if out[0] < 0.0 else out


def plate_vertical_quat(slot_yaw: float) -> np.ndarray:
    """Orientation of a plate standing vertically in a slot with heading ``slot_yaw``.

    A quarter turn about the plate's own ``y`` axis tips its normal from ``+z`` to ``+x``, then the
    slot yaw turns that normal onto the slot's ``x`` axis, which is horizontal: the plate is
    vertical and faces across the tines.
    """
    quarter_turn = quat_from_axis_angle((0.0, 1.0, 0.0), math.pi / 2)
    return quat_multiply(quat_from_yaw(slot_yaw), quarter_turn)


def tilt_from_vertical(q) -> float:
    """Angle in radians between a plate's plane and the vertical, from its orientation ``q``.

    The plate normal is the rotated ``z`` axis; the plate is vertical when that normal is
    horizontal, so the tilt is ``asin(|n_z|)``: zero for a vertical plate, ``pi / 2`` for a flat
    one.
    """
    n_z = float(quat_apply(q, (0.0, 0.0, 1.0))[2])
    return math.asin(min(abs(n_z), 1.0))


@dataclass(frozen=True, eq=False)
class Pose:
    """A position and an orientation in one frame. Arrays are copied and made read-only.

    ``position`` has shape ``(3,)``; ``quat_wxyz`` has shape ``(4,)``, ``(w, x, y, z)``, and is
    normalised on construction.
    """

    position: np.ndarray
    quat_wxyz: np.ndarray = _IDENTITY  # type: ignore[assignment]

    def __post_init__(self) -> None:
        pos = np.array(self.position, dtype=np.float64).reshape(-1)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            raise ValueError(f"position must be 3 finite numbers, got {self.position!r}")
        quat = _unit_quat(self.quat_wxyz).copy()
        pos.flags.writeable = False
        quat.flags.writeable = False
        object.__setattr__(self, "position", pos)
        object.__setattr__(self, "quat_wxyz", quat)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pose):
            return NotImplemented
        return bool(
            np.array_equal(self.position, other.position)
            and np.array_equal(self.quat_wxyz, other.quat_wxyz)
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        p = ", ".join(f"{v:.4f}" for v in self.position)
        q = ", ".join(f"{v:.4f}" for v in self.quat_wxyz)
        return f"Pose(position=({p}), quat_wxyz=({q}))"

    @property
    def yaw(self) -> float:
        """Heading of this pose's ``x`` axis, radians."""
        return yaw_from_quat(self.quat_wxyz)

    def raised(self, dz: float) -> Pose:
        """The same orientation, ``dz`` metres higher along the frame's ``z`` axis."""
        return Pose(self.position + np.array([0.0, 0.0, float(dz)]), self.quat_wxyz)

    def with_quat(self, quat_wxyz) -> Pose:
        """The same position with another orientation."""
        return Pose(self.position, quat_wxyz)


def pose_compose(a: Pose, b: Pose) -> Pose:
    """``a * b``: the pose ``b``, given in ``a``'s frame, expressed in the frame ``a`` is given in.

    Position ``a.p + R(a) b.p``, orientation ``a.q * b.q``. Read as transforms, ``b`` is applied
    first, as in :func:`quat_multiply`.
    """
    return Pose(
        a.position + quat_apply(a.quat_wxyz, b.position),
        quat_multiply(a.quat_wxyz, b.quat_wxyz),
    )


def pose_inverse(a: Pose) -> Pose:
    """The inverse transform: ``pose_compose(a, pose_inverse(a))`` is the identity pose."""
    q_inv = quat_conjugate(a.quat_wxyz)
    return Pose(-quat_apply(q_inv, a.position), q_inv)


def to_frame(world_pose: Pose, frame_pose: Pose) -> Pose:
    """``world_pose`` expressed in the frame whose world pose is ``frame_pose``.

    Used every control step with ``frame_pose`` the current pelvis pose, so a world-frame target
    becomes a pelvis-frame target and the arm, not the legs, compensates body sway
    (docs/architecture.md section 5). Inverse of ``pose_compose(frame_pose, local_pose)``.
    """
    return pose_compose(pose_inverse(frame_pose), world_pose)
