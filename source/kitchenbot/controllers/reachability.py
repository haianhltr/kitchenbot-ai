"""Reachability envelope of the right palm, sampled with URDF forward kinematics (#48).

docs/architecture.md section 7: targets outside a precomputed workspace envelope are rejected and
reported as ``ik_infeasible``, so an unreachable target is not mistaken for a controller failure.
This module builds that envelope without a simulator and answers ``contains`` queries; the gate
itself is :class:`kitchenbot.controllers.target_filter.ArmTargetPipeline`.

What is sampled:

- **Frame:** the pelvis frame, with the three waist joints at zero. The balance policy, not the arm
  controller, moves the waist (section 5), so the true envelope shifts when it counter-leans; #16
  decides whether to re-sample with the waist range (``right_arm_chain`` already takes the waist as
  held inputs) or to keep the waist-at-zero envelope.
- **Point:** the origin of ``right_hand_palm_link``, the Dex3 palm link fixed to the wrist. The IK's
  ``body_offset`` to the palm centre is #14's; it shifts the envelope by a few centimetres.
- **Configurations:** uniform in the URDF limits (``joint_limits.ARM_RIGHT_LIMITS``), with the
  5 degree margin of section 7 taken off both ends of the wrist roll, pitch and yaw ranges, drawn
  from ``numpy.random.default_rng(seed)``; seed and sample count are recorded in the file.
- **Voxels:** a grid of ``spacing`` (2 cm) cubes in the pelvis frame. A voxel is in the envelope if
  at least one sample's palm position falls in it; nothing is filled in or smoothed. The grid has
  one empty voxel of padding on every side.

Known limits: forward kinematics only, no collision check, so voxels inside the torso count as
reachable; near the boundary, where samples are sparse, the envelope is slightly conservative. #3
checks the chain against the Isaac Sim USD and #16 checks the envelope against the real IK.

File format (``.npz``, no pickles): ``packed`` (``np.packbits`` of the C-order boolean grid),
``shape``, ``origin`` (pelvis-frame corner of voxel ``(0, 0, 0)``, m), ``spacing`` (m), ``seed``,
``samples``, ``wrist_margin_deg``, ``frame``, ``tip_link``, ``joint_source``. The zip entries carry
a fixed timestamp, so regenerating with the same inputs on the same platform gives the same bytes.
``scripts/make_reach_envelope.py`` writes and checks it. numpy only, no simulator import.
"""

from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kitchenbot.robots.kinematics import UrdfChain

__all__ = [
    "DEFAULT_PATH",
    "DEFAULT_SAMPLES",
    "DEFAULT_SEED",
    "VOXEL_SIZE_M",
    "WRIST_MARGIN_DEG",
    "ReachabilityEnvelope",
    "build_envelope",
    "sample_tip_positions",
    "sampling_bounds",
]

#: Joint-limit margin on the wrist, docs/architecture.md section 7.
WRIST_MARGIN_DEG = 5.0
#: Edge length of one voxel, metres.
VOXEL_SIZE_M = 0.02
#: Seed and sample count of the committed envelope.
DEFAULT_SEED = 48
DEFAULT_SAMPLES = 2_000_000
#: The committed envelope of the right palm.
DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "right_arm_envelope.npz"

_BATCH = 250_000
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def sampling_bounds(
    chain: UrdfChain, wrist_margin_deg: float = WRIST_MARGIN_DEG
) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper sampling bounds of ``chain.active_joints``, wrist ranges shrunk."""
    margin = math.radians(wrist_margin_deg)
    lower, upper = chain.lower.copy(), chain.upper.copy()
    for i, name in enumerate(chain.active_joints):
        if "_wrist_" in name:
            lower[i] += margin
            upper[i] -= margin
    if np.any(lower >= upper):
        raise ValueError("the wrist margin leaves an empty joint range")
    return lower, upper


def sample_tip_positions(
    chain: UrdfChain,
    samples: int,
    seed: int,
    wrist_margin_deg: float = WRIST_MARGIN_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """``(q, positions)``: ``samples`` uniform configurations and their tip positions.

    Draws are made in fixed batches from one ``default_rng(seed)``, so the result depends only on
    ``samples``, ``seed`` and the bounds. Held joints (the waist) stay at zero.
    """
    if samples < 1:
        raise ValueError("samples must be at least 1")
    lower, upper = sampling_bounds(chain, wrist_margin_deg)
    rng = np.random.default_rng(seed)
    qs, positions = [], []
    for start in range(0, samples, _BATCH):
        q = rng.uniform(lower, upper, size=(min(_BATCH, samples - start), chain.dof))
        qs.append(q)
        positions.append(chain.fk_batch(q)[0])
    return np.concatenate(qs), np.concatenate(positions)


@dataclass(frozen=True, eq=False)
class ReachabilityEnvelope:
    """A boolean voxel grid of reachable tip positions in one frame (the pelvis)."""

    occupied: np.ndarray
    origin: np.ndarray
    spacing: float
    seed: int
    samples: int
    wrist_margin_deg: float = WRIST_MARGIN_DEG
    frame: str = "pelvis"
    tip_link: str = "right_hand_palm_link"
    joint_source: str = ""

    def __post_init__(self) -> None:
        occupied = np.array(self.occupied, dtype=bool)
        origin = np.array(self.origin, dtype=np.float64).reshape(3)
        if occupied.ndim != 3:
            raise ValueError(f"occupied must be a 3-D grid, got shape {occupied.shape}")
        if not self.spacing > 0.0:
            raise ValueError("spacing must be positive")
        occupied.flags.writeable = False
        origin.flags.writeable = False
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", float(self.spacing))

    # --- construction ------------------------------------------------------------------------

    @classmethod
    def from_points(cls, points: np.ndarray, spacing: float, **meta) -> ReachabilityEnvelope:
        """Mark the voxel of every point; the grid spans the points plus one voxel of padding."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        lo = np.floor(pts.min(axis=0) / spacing).astype(np.int64) - 1
        hi = np.floor(pts.max(axis=0) / spacing).astype(np.int64) + 1
        origin = lo * float(spacing)
        grid = np.zeros(tuple(hi - lo + 1), dtype=bool)
        empty = cls(grid, origin, spacing, **meta)
        idx = empty.voxel_index(pts)
        if np.any(idx < 1) or np.any(idx >= np.array(grid.shape) - 1):
            raise AssertionError("a point fell outside the padded grid")  # rounding guard
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return cls(grid, origin, spacing, **meta)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> ReachabilityEnvelope:
        """Read an envelope written by :meth:`save` (no pickles)."""
        with np.load(Path(path), allow_pickle=False) as data:
            shape = tuple(int(v) for v in data["shape"])
            count = int(np.prod(shape))
            occupied = np.unpackbits(data["packed"], count=count).astype(bool).reshape(shape)
            return cls(
                occupied=occupied,
                origin=data["origin"],
                spacing=float(data["spacing"]),
                seed=int(data["seed"]),
                samples=int(data["samples"]),
                wrist_margin_deg=float(data["wrist_margin_deg"]),
                frame=str(data["frame"]),
                tip_link=str(data["tip_link"]),
                joint_source=str(data["joint_source"]),
            )

    def save(self, path: str | Path) -> Path:
        """Write the packed grid and its metadata as a compressed ``.npz``."""
        arrays = {
            "packed": np.packbits(self.occupied.ravel(order="C")),
            "shape": np.array(self.occupied.shape, dtype=np.int64),
            "origin": self.origin,
            "spacing": np.float64(self.spacing),
            "seed": np.int64(self.seed),
            "samples": np.int64(self.samples),
            "wrist_margin_deg": np.float64(self.wrist_margin_deg),
            "frame": np.str_(self.frame),
            "tip_link": np.str_(self.tip_link),
            "joint_source": np.str_(self.joint_source),
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, value in arrays.items():
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 0
                zf.writestr(info, buffer.getvalue())
        return out

    # --- queries -----------------------------------------------------------------------------

    def voxel_index(self, points: np.ndarray) -> np.ndarray:
        """Integer voxel indices ``(N, 3)`` of ``points`` ``(N, 3)``; may lie outside the grid."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return np.floor((pts - self.origin) / self.spacing).astype(np.int64)

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        """Boolean ``(N,)``: which points lie in an occupied voxel."""
        idx = self.voxel_index(points)
        inside = np.all((idx >= 0) & (idx < np.array(self.occupied.shape)), axis=1)
        out = np.zeros(len(idx), dtype=bool)
        good = idx[inside]
        out[inside] = self.occupied[good[:, 0], good[:, 1], good[:, 2]]
        return out

    def contains(self, p) -> bool:
        """True if the point ``p`` (pelvis frame, metres) lies in an occupied voxel."""
        return bool(self.contains_points(np.asarray(p, dtype=np.float64).reshape(1, 3))[0])

    def margin_contains(self, p, m: float) -> bool:
        """True if every voxel within distance ``m`` of ``p`` is occupied: ``p`` is ``m`` inside.

        ``m = 0`` is :meth:`contains`. A positive margin keeps targets away from the sparse
        boundary of the sampled envelope.
        """
        if m < 0.0:
            raise ValueError("margin must be non-negative")
        point = np.asarray(p, dtype=np.float64).reshape(3)
        if m == 0.0:
            return self.contains(point)
        lo = self.voxel_index(point - m)[0]
        hi = self.voxel_index(point + m)[0]
        axes = [np.arange(a, b + 1) for a, b in zip(lo, hi, strict=True)]
        idx = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        box_lo = self.origin + idx * self.spacing
        nearest = np.clip(point, box_lo, box_lo + self.spacing)
        near = np.linalg.norm(nearest - point, axis=1) <= m
        idx = idx[near]
        if np.any(idx < 0) or np.any(idx >= np.array(self.occupied.shape)):
            return False
        return bool(np.all(self.occupied[idx[:, 0], idx[:, 1], idx[:, 2]]))

    # --- summaries ---------------------------------------------------------------------------

    @property
    def occupied_count(self) -> int:
        return int(self.occupied.sum())

    @property
    def fill_fraction(self) -> float:
        """Occupied voxels over all voxels of the grid."""
        return self.occupied_count / self.occupied.size

    @property
    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Min and max corners, metres, of the occupied voxels."""
        idx = np.argwhere(self.occupied)
        if len(idx) == 0:
            raise ValueError("the envelope is empty")
        return (
            self.origin + idx.min(axis=0) * self.spacing,
            self.origin + (idx.max(axis=0) + 1) * self.spacing,
        )

    def same_as(self, other: ReachabilityEnvelope) -> bool:
        """Equal grids, origins, spacing and sampling metadata."""
        return (
            self.occupied.shape == other.occupied.shape
            and bool(np.array_equal(self.occupied, other.occupied))
            and bool(np.array_equal(self.origin, other.origin))
            and self.spacing == other.spacing
            and (self.seed, self.samples, self.wrist_margin_deg)
            == (other.seed, other.samples, other.wrist_margin_deg)
            and (self.frame, self.tip_link, self.joint_source)
            == (other.frame, other.tip_link, other.joint_source)
        )


def build_envelope(
    chain: UrdfChain,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    spacing: float = VOXEL_SIZE_M,
    wrist_margin_deg: float = WRIST_MARGIN_DEG,
    joint_source: str = "",
) -> ReachabilityEnvelope:
    """Sample ``chain`` and return its envelope in the chain's base frame (waist at zero)."""
    _, positions = sample_tip_positions(chain, samples, seed, wrist_margin_deg)
    return ReachabilityEnvelope.from_points(
        positions,
        spacing,
        seed=seed,
        samples=samples,
        wrist_margin_deg=wrist_margin_deg,
        frame=chain.base_link,
        tip_link=chain.tip_link,
        joint_source=joint_source,
    )
