"""Reachability envelope of the right palm, sampled with URDF forward kinematics (#48, #56, #66).

docs/architecture.md section 7: targets outside a precomputed workspace envelope are rejected and
reported as ``ik_infeasible``, so an unreachable target is not mistaken for a controller failure.
This module builds that envelope without a simulator and answers ``contains`` queries; the gate
itself is :class:`kitchenbot.controllers.target_filter.ArmTargetPipeline`.

Two envelopes are committed, both of the finger-base point :data:`FINGER_BASE_OFFSET_M` (D012, the
IK's controlled point; regenerated in #66 with the seeds, sample counts and waist ranges of #48 and
#56, which sampled the palm link origin; the waist file again in #77 with only its pitch maximum
changed), each with a name for :meth:`ReachabilityEnvelope.load`:

- ``waist0``, ``data/right_arm_envelope.npz`` (#48, :data:`DEFAULT_PATH`): the waist held at zero.
  This is the one ``ArmTargetPipeline`` loads by default.
- ``waist``, ``data/right_arm_envelope_waist.npz`` (#56, :data:`WAIST_PATH`): the waist sampled
  too, waist pitch 0 to +26.81 deg (forward lean up to the soft limit
  ``joint_limits.WAIST_LEAN_MAX_RAD``, 0.468 rad, the lean the planner commands and the balance
  policy is trained on; regenerated in #77, #56 and #66 sampled up to the 0.52 rad URDF limit),
  yaw -20 to +20 deg, roll 0. It answers "what would a waist lean buy" for #10, #14 and #16
  (``scripts/reach_gap_analysis.py``) and is the dry run's gate with ``gate_envelope=waist``.
  Neither the arm controller nor the balance policy guarantees a posture in that range yet (#5
  trains the lean, #21 commands it), so it is an upper bound, not a gate the arm controller can
  rely on.

What is sampled:

- **Frame:** the pelvis frame. The balance policy, not the arm controller, moves the waist
  (section 5), so the true envelope shifts when it counter-leans. By default the three waist
  joints are held at zero. With ``waist_range_deg`` each sample also draws the waist uniformly in
  a range per joint (yaw, roll, pitch, in ``joint_maps.WAIST`` order), every range clipped to the
  URDF limits ``joint_limits.WAIST_LIMITS``: yaw +-150 deg (2.618 rad), roll and pitch +-29.79 deg
  (0.52 rad). Positive waist pitch leans the torso forward (the joint turns about the pelvis
  ``+y`` axis, which tips the torso's ``+z`` toward ``+x``). The envelope is then the union over the
  range: a point is in it if *some* waist posture in the range and some arm posture reach it.
- **Point:** the origin of ``right_hand_palm_link``, the Dex3 palm link fixed to the wrist, or,
  with ``tip_offset_m``, a point fixed in that link's frame (the IK's ``body_offset``, #14's).
  Finger bases sit 7.8 cm along the link's ``x`` (``right_hand_index_0_joint`` and
  ``right_hand_middle_0_joint`` origins in the URDF), so the offset moves the envelope by up to a
  few centimetres. The committed files use :data:`FINGER_BASE_OFFSET_M` and record it.
- **Configurations:** the seven arm joints uniform in the URDF limits
  (``joint_limits.ARM_RIGHT_LIMITS``), with the 5 degree margin of section 7 taken off both ends of
  the wrist roll, pitch and yaw ranges, drawn from ``numpy.random.default_rng(seed)``. The waist,
  when sampled, is drawn from a separate stream, ``default_rng([seed, WAIST_STREAM])``, so the arm
  draws of a seed are the same with and without the waist, and a zero range draws nothing at all:
  a zero-range envelope is the #48 envelope, bit for bit. Seed and sample count are recorded in
  the file.
- **Voxels:** a grid of ``spacing`` (2 cm) cubes in the pelvis frame. A voxel is in the envelope if
  at least one sample's palm position falls in it; nothing is filled in or smoothed. The grid has
  one empty voxel of padding on every side.

Known limits: forward kinematics only, no collision check, so voxels inside the torso count as
reachable; no orientation, so a voxel is reachable with *some* palm orientation, not the planner's;
near the boundary, where samples are sparse, the envelope is slightly conservative. #3 checks the
chain against the Isaac Sim USD and #16 checks the envelope against the real IK.

File format (``.npz``, no pickles): ``packed`` (``np.packbits`` of the C-order boolean grid),
``shape``, ``origin`` (pelvis-frame corner of voxel ``(0, 0, 0)``, m), ``spacing`` (m), ``seed``,
``samples``, ``wrist_margin_deg``, ``frame``, ``tip_link``, ``joint_source``; and, written only
when they are not zero, ``waist_range_deg`` (3 x 2: yaw, roll, pitch rows; lower, upper columns;
degrees, after clipping) and ``tip_offset_m`` (3). An absent key means zero, so the #48 file reads
as waist at zero and a zero-range generation writes the same bytes as #48. The zip entries carry a
fixed timestamp, so regenerating with the same inputs on the same platform gives the same bytes.
``scripts/make_reach_envelope.py`` writes and checks it. numpy only, no simulator import.
"""

from __future__ import annotations

import io
import math
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kitchenbot.robots import joint_maps
from kitchenbot.robots.joint_limits import WAIST_LEAN_MAX_RAD, WAIST_LIMITS
from kitchenbot.robots.kinematics import UrdfChain

__all__ = [
    "DEFAULT_PATH",
    "DEFAULT_SAMPLES",
    "DEFAULT_SEED",
    "ENVELOPE_PATHS",
    "FINGER_BASE_OFFSET_M",
    "VOXEL_SIZE_M",
    "WAIST_ENVELOPE_RANGE_DEG",
    "WAIST_ENVELOPE_SAMPLES",
    "WAIST_PATH",
    "WAIST_STREAM",
    "WRIST_MARGIN_DEG",
    "ZERO_WAIST_DEG",
    "ReachabilityEnvelope",
    "build_envelope",
    "clip_waist_range",
    "sample_configurations",
    "sample_tip_positions",
    "sampling_bounds",
    "waist_range_deg",
]

#: Joint-limit margin on the wrist, docs/architecture.md section 7.
WRIST_MARGIN_DEG = 5.0
#: Edge length of one voxel, metres.
VOXEL_SIZE_M = 0.02
#: Seed and sample count of the committed envelope.
DEFAULT_SEED = 48
DEFAULT_SAMPLES = 2_000_000
#: The committed envelope of the right palm, waist at zero (#48).
DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "right_arm_envelope.npz"
#: The committed envelope with the waist sampled (#56).
WAIST_PATH = Path(__file__).resolve().parent / "data" / "right_arm_envelope_waist.npz"
#: The committed envelopes by name, for :meth:`ReachabilityEnvelope.load` and the dry run's
#: ``gate_envelope`` extra: ``waist0`` the waist at zero, ``waist`` the waist sampled.
ENVELOPE_PATHS: dict[str, Path] = {"waist0": DEFAULT_PATH, "waist": WAIST_PATH}
#: The point both committed envelopes sample, in the ``right_hand_palm_link`` frame, metres: the
#: finger bases, the mean of the ``right_hand_index_0_joint`` and ``right_hand_middle_0_joint``
#: origins in the URDF fixture, which D012 makes the IK's controlled point (#14's ``body_offset``).
FINGER_BASE_OFFSET_M: tuple[float, float, float] = (0.0777, -0.0016, 0.0)

#: Waist yaw, roll and pitch ranges ``(lower, upper)`` in degrees of the #48 envelope: all zero.
ZERO_WAIST_DEG: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
#: Waist range of ``right_arm_envelope_waist.npz`` (yaw, roll, pitch; degrees, inside the URDF
#: limits, so clipping leaves it unchanged). Why: a forward lean is the only waist motion that
#: brings the palm toward the counter, and its maximum is the lean the planner commands and the
#: balance policy is trained on, ``joint_limits.WAIST_LEAN_MAX_RAD`` (the soft limit, 0.468 rad =
#: 26.81 deg; D012 as amended, #77; #56 requested 30 deg, clipped to the 0.52 rad URDF limit);
#: +-20 deg of yaw turns the right shoulder toward the plate box and the rack on the robot's right
#: without the 150 deg the URDF allows (which would also swing the arm behind the robot); roll
#: stays zero because a side lean moves the shoulder sideways by a few centimetres and costs
#: balance.
WAIST_ENVELOPE_RANGE_DEG: tuple[tuple[float, float], ...] = (
    (-20.0, 20.0),
    (0.0, 0.0),
    (0.0, math.degrees(WAIST_LEAN_MAX_RAD)),
)
#: Sample count of the waist envelope: three more dimensions than #48, so more samples per voxel.
WAIST_ENVELOPE_SAMPLES = 8_000_000
#: Second word of the waist generator's seed sequence: ``default_rng([seed, WAIST_STREAM])``.
WAIST_STREAM = 56

_BATCH = 250_000
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def clip_waist_range(
    ranges_deg: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    """``ranges_deg`` (yaw, roll, pitch; ``(lower, upper)`` in degrees) clipped to the URDF limits.

    Raises ``ValueError`` if a range has ``lower > upper`` or is not finite.
    """
    rows = [tuple(float(v) for v in row) for row in ranges_deg]
    if len(rows) != len(joint_maps.WAIST) or any(len(row) != 2 for row in rows):
        raise ValueError(f"waist ranges must be 3 (lower, upper) pairs {joint_maps.WAIST}")
    out = []
    for name, (lo, hi) in zip(joint_maps.WAIST, rows, strict=True):
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
            raise ValueError(f"{name}: range must be finite with lower <= upper, got {(lo, hi)}")
        limit = WAIST_LIMITS[name]
        lo_lim, hi_lim = math.degrees(limit.lower), math.degrees(limit.upper)
        # + 0.0 turns a -0.0 (from -abs(0)) into 0.0, so ranges print and compare cleanly
        out.append((min(max(lo, lo_lim), hi_lim) + 0.0, min(max(hi, lo_lim), hi_lim) + 0.0))
    return tuple(out)


def waist_range_deg(
    yaw_deg: float = 0.0,
    roll_deg: float = 0.0,
    pitch_min_deg: float = 0.0,
    pitch_max_deg: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    """The clipped waist range of ``make_reach_envelope.py``'s flags: yaw and roll are +-half
    ranges (``abs`` is taken), pitch is ``[pitch_min_deg, pitch_max_deg]``, positive leaning
    forward."""
    yaw, roll = abs(float(yaw_deg)), abs(float(roll_deg))
    return clip_waist_range(((-yaw, yaw), (-roll, roll), (pitch_min_deg, pitch_max_deg)))


def _is_zero(values) -> bool:
    return not np.any(np.asarray(values, dtype=np.float64))


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


def sample_configurations(
    chain: UrdfChain,
    samples: int,
    seed: int,
    wrist_margin_deg: float = WRIST_MARGIN_DEG,
    waist_range: Sequence[Sequence[float]] = ZERO_WAIST_DEG,
    tip_offset_m: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(q, waist, positions)``: arm configurations, waist postures and tip positions.

    ``q`` is ``(samples, chain.dof)`` uniform in :func:`sampling_bounds`, drawn in fixed batches
    from one ``default_rng(seed)``, exactly as in #48. ``waist`` is ``(samples, 3)`` in
    ``chain.held_joints`` order (the waist, radians): zeros when ``waist_range`` (degrees, see
    :func:`clip_waist_range`, applied here) is zero, otherwise uniform per joint from
    ``default_rng([seed, WAIST_STREAM])``. ``positions`` is ``(samples, 3)`` in the base frame, of
    the tip link origin plus ``tip_offset_m`` given in the tip link frame.
    """
    if samples < 1:
        raise ValueError("samples must be at least 1")
    lower, upper = sampling_bounds(chain, wrist_margin_deg)
    waist = np.radians(np.array(clip_waist_range(waist_range)))
    moving = not _is_zero(waist)
    if moving and chain.held_joints != joint_maps.WAIST:
        raise ValueError(f"waist sampling needs the waist as held joints, got {chain.held_joints}")
    offset = None if tip_offset_m is None or _is_zero(tip_offset_m) else np.asarray(tip_offset_m)
    if offset is not None and offset.shape != (3,):
        raise ValueError("tip_offset_m must have 3 values")
    full = None
    if moving:
        full = UrdfChain(
            chain.joints, chain.base_link, chain.tip_link, chain.held_joints + chain.active_joints
        )
    rng = np.random.default_rng(seed)
    waist_rng = np.random.default_rng([seed, WAIST_STREAM]) if moving else None
    qs, waists, positions = [], [], []
    for start in range(0, samples, _BATCH):
        n = min(_BATCH, samples - start)
        q = rng.uniform(lower, upper, size=(n, chain.dof))
        if full is None:
            w = np.zeros((n, len(chain.held_joints)))
            pos, rot = chain.fk_batch(q)
        else:
            w = waist_rng.uniform(waist[:, 0], waist[:, 1], size=(n, 3))
            pos, rot = full.fk_batch(np.hstack([w, q]))
        if offset is not None:
            pos = pos + rot @ offset
        qs.append(q)
        waists.append(w)
        positions.append(pos)
    return np.concatenate(qs), np.concatenate(waists), np.concatenate(positions)


def sample_tip_positions(
    chain: UrdfChain,
    samples: int,
    seed: int,
    wrist_margin_deg: float = WRIST_MARGIN_DEG,
    waist_range: Sequence[Sequence[float]] = ZERO_WAIST_DEG,
    tip_offset_m: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(q, positions)`` of :func:`sample_configurations`: arm configurations and tip positions.

    With the default zero waist range the result depends only on ``samples``, ``seed`` and the
    bounds, as in #48; held joints (the waist) stay at zero.
    """
    q, _, positions = sample_configurations(
        chain, samples, seed, wrist_margin_deg, waist_range, tip_offset_m
    )
    return q, positions


def _pairs(value) -> tuple[tuple[float, float], ...]:
    arr = np.asarray(value, dtype=np.float64).reshape(3, 2)
    return tuple((float(lo), float(hi)) for lo, hi in arr)


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
    #: Waist yaw, roll, pitch ``(lower, upper)`` in degrees the samples were drawn in.
    waist_range_deg: tuple[tuple[float, float], ...] = ZERO_WAIST_DEG
    #: Sampled point in the tip link frame, metres; zero is the link origin.
    tip_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

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
        object.__setattr__(self, "waist_range_deg", _pairs(self.waist_range_deg))
        offset = tuple(float(v) for v in np.asarray(self.tip_offset_m, dtype=np.float64).ravel())
        if len(offset) != 3:
            raise ValueError("tip_offset_m must have 3 values")
        object.__setattr__(self, "tip_offset_m", offset)

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

    @staticmethod
    def resolve(path_or_name: str | Path = DEFAULT_PATH) -> Path:
        """The file of a committed envelope's name (:data:`ENVELOPE_PATHS`: ``waist0``,
        ``waist``), or ``path_or_name`` itself as a path. A name wins over a file of that name in
        the working directory."""
        if isinstance(path_or_name, str) and path_or_name in ENVELOPE_PATHS:
            return ENVELOPE_PATHS[path_or_name]
        return Path(path_or_name)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> ReachabilityEnvelope:
        """Read an envelope written by :meth:`save` (no pickles); absent waist keys mean zero.

        ``path`` is a file, or the name of a committed envelope (``"waist0"``, ``"waist"``; see
        :meth:`resolve`).
        """
        with np.load(cls.resolve(path), allow_pickle=False) as data:
            shape = tuple(int(v) for v in data["shape"])
            count = int(np.prod(shape))
            occupied = np.unpackbits(data["packed"], count=count).astype(bool).reshape(shape)
            waist = data["waist_range_deg"] if "waist_range_deg" in data.files else ZERO_WAIST_DEG
            offset = data["tip_offset_m"] if "tip_offset_m" in data.files else (0.0, 0.0, 0.0)
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
                waist_range_deg=waist,
                tip_offset_m=offset,
            )

    def save(self, path: str | Path) -> Path:
        """Write the packed grid and its metadata as a compressed ``.npz``.

        ``waist_range_deg`` and ``tip_offset_m`` are written only when they are not zero, so a
        waist-at-zero envelope has exactly the #48 keys and bytes.
        """
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
        if not self.waist_at_zero:
            arrays["waist_range_deg"] = np.array(self.waist_range_deg, dtype=np.float64)
        if not _is_zero(self.tip_offset_m):
            arrays["tip_offset_m"] = np.array(self.tip_offset_m, dtype=np.float64)
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
    def waist_at_zero(self) -> bool:
        """True if the waist was held at zero (the #48 envelope)."""
        return _is_zero(self.waist_range_deg)

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
            and (self.waist_range_deg, self.tip_offset_m)
            == (other.waist_range_deg, other.tip_offset_m)
        )


def build_envelope(
    chain: UrdfChain,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    spacing: float = VOXEL_SIZE_M,
    wrist_margin_deg: float = WRIST_MARGIN_DEG,
    joint_source: str = "",
    waist_range: Sequence[Sequence[float]] = ZERO_WAIST_DEG,
    tip_offset_m: Sequence[float] | None = None,
) -> ReachabilityEnvelope:
    """Sample ``chain`` and return its envelope in the chain's base frame.

    The waist is at zero unless ``waist_range`` (degrees, clipped to the URDF limits) says
    otherwise; the clipped range is what the envelope records.
    """
    clipped = clip_waist_range(waist_range)
    _, _, positions = sample_configurations(
        chain, samples, seed, wrist_margin_deg, clipped, tip_offset_m
    )
    return ReachabilityEnvelope.from_points(
        positions,
        spacing,
        seed=seed,
        samples=samples,
        wrist_margin_deg=wrist_margin_deg,
        frame=chain.base_link,
        tip_link=chain.tip_link,
        joint_source=joint_source,
        waist_range_deg=clipped,
        tip_offset_m=(0.0, 0.0, 0.0) if tip_offset_m is None else tuple(tip_offset_m),
    )
