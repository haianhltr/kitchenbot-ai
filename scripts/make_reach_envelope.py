r"""Sample the right palm's reachability envelope from the URDF joint dump (#48, #56, #66; no
simulator).

Generate the committed envelope, waist at zero, of the finger-base point (D012; deterministic: the
same seed, sample count and offset give the same grid, and the same bytes on the same platform)::

    py -3.12 scripts/make_reach_envelope.py --seed 48 --samples 2000000 \
        --tip-offset 0.0777 -0.0016 0 \
        --out source/kitchenbot/controllers/data/right_arm_envelope.npz

Generate the committed waist envelope (#56): the waist sampled too, pitch 0 to +30 deg forward
lean (clipped to the 0.52 rad URDF limit, 29.79 deg), yaw -20 to +20 deg, roll 0; the arm draws
are the same as above, the waist comes from its own stream (``kitchenbot.controllers.reachability``
says why this range)::

    py -3.12 scripts/make_reach_envelope.py --seed 48 --samples 8000000 \
        --waist-yaw-deg 20 --waist-pitch-min-deg 0 --waist-pitch-max-deg 30 \
        --tip-offset 0.0777 -0.0016 0 \
        --out source/kitchenbot/controllers/data/right_arm_envelope_waist.npz

``--tip-offset X Y Z`` (metres, in the ``right_hand_palm_link`` frame) samples a point fixed in the
palm link instead of its origin; ``0.0777 -0.0016 0`` is the finger bases,
``reachability.FINGER_BASE_OFFSET_M`` (#66; #48 and #56 sampled the origin, the default). The file
records a non-zero offset as ``tip_offset_m`` and ``--check`` prints it.

The waist flags default to 0, which is the #48 behaviour bit for bit. ``--waist-yaw-deg`` and
``--waist-roll-deg`` are half ranges (+-); the pitch range is ``[min, max]``, positive leaning
forward. Every range is clipped to ``robots/joint_limits.WAIST_LIMITS`` (yaw +-2.618 rad, roll and
pitch +-0.52 rad) and the clipped range is what the file records. With a waist range and no
``--out`` the file goes to ``right_arm_envelope_waist.npz``, never over the #48 file.

Inspect a file: waist ranges, fill fraction, bounding box, occupied volume, seed, sample count and
file size (exit code 1 if the file is over 200 KB or the grid is empty)::

    py -3.12 scripts/make_reach_envelope.py \
        --check source/kitchenbot/controllers/data/right_arm_envelope.npz

Either form takes ``--plot`` to write ``outputs/figures/right_arm_envelope.png`` (git-ignored):
three orthographic projections of the occupied voxels in the pelvis frame.

The chain is built by ``kitchenbot.robots.kinematics.right_arm_chain`` from a joint dump written
by ``scripts/smoke/dump_urdf_joints.py``, by default the committed URDF fixture
``tests/fixtures/g1_dex3_joints_urdf.json``. What is sampled and how is documented in
``kitchenbot.controllers.reachability``. numpy and matplotlib only; nothing here imports Isaac Sim
or Isaac Lab.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from kitchenbot.controllers.reachability import (
    DEFAULT_PATH,
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    VOXEL_SIZE_M,
    WAIST_PATH,
    WRIST_MARGIN_DEG,
    ZERO_WAIST_DEG,
    ReachabilityEnvelope,
    build_envelope,
    waist_range_deg,
)
from kitchenbot.robots import joint_maps
from kitchenbot.robots.joint_limits import WAIST_LIMITS
from kitchenbot.robots.kinematics import right_arm_chain

REPO = Path(__file__).resolve().parents[1]
DEFAULT_JOINTS = REPO / "tests" / "fixtures" / "g1_dex3_joints_urdf.json"
DEFAULT_OUT = DEFAULT_PATH
DEFAULT_WAIST_OUT = WAIST_PATH
DEFAULT_PLOT = REPO / "outputs" / "figures" / "right_arm_envelope.png"
MAX_BYTES = 200_000  # issue #48 acceptance: the envelope file is under 200 KB


def load_dump(path: Path) -> tuple[list[dict], str]:
    """The dump's joint list and a short provenance string (URL at commit, or the file name)."""
    dump = json.loads(Path(path).read_text(encoding="utf-8"))
    source = dump.get("source") or {}
    if source.get("url") and source.get("commit"):
        provenance = f"{source['url']} (commit {source['commit']})"
    else:
        provenance = Path(path).name
    return dump["joints"], provenance


def generate(
    joints_path: Path = DEFAULT_JOINTS,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    spacing: float = VOXEL_SIZE_M,
    wrist_margin_deg: float = WRIST_MARGIN_DEG,
    waist_range: Sequence[Sequence[float]] = ZERO_WAIST_DEG,
    tip_offset_m: Sequence[float] | None = None,
) -> ReachabilityEnvelope:
    """Build the right-palm envelope from a joint dump file (waist at zero and the palm link
    origin by default; ``tip_offset_m`` in the palm link frame moves the sampled point)."""
    joints, provenance = load_dump(joints_path)
    return build_envelope(
        right_arm_chain(joints),
        samples=samples,
        seed=seed,
        spacing=spacing,
        wrist_margin_deg=wrist_margin_deg,
        joint_source=provenance,
        waist_range=waist_range,
        tip_offset_m=tip_offset_m,
    )


def _fmt_deg(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def waist_summary(envelope: ReachabilityEnvelope) -> str:
    """One line: the sampled waist ranges and the URDF limits they were clipped to."""
    ranges = ", ".join(
        f"{name.removeprefix('waist_').removesuffix('_joint')} [{_fmt_deg(lo)}, {_fmt_deg(hi)}]"
        for name, (lo, hi) in zip(joint_maps.WAIST, envelope.waist_range_deg, strict=True)
    )
    limits = ", ".join(
        f"{name.removeprefix('waist_').removesuffix('_joint')} "
        f"+-{_fmt_deg(math.degrees(WAIST_LIMITS[name].upper))}"
        for name in joint_maps.WAIST
    )
    state = "waist at zero" if envelope.waist_at_zero else "waist sampled"
    return f"waist ranges (deg): {ranges}; {state}; URDF limits {limits} deg"


def report(envelope: ReachabilityEnvelope, path: Path | None = None) -> str:
    """Plain-text summary for ``--check`` and after generation."""
    lo, hi = envelope.bounding_box
    fmt = ", ".join
    lines = [
        f"frame {envelope.frame}, tip {envelope.tip_link}",
        waist_summary(envelope),
        f"seed {envelope.seed}, samples {envelope.samples}, "
        f"wrist margin {envelope.wrist_margin_deg:g} deg",
        f"grid {'x'.join(str(n) for n in envelope.occupied.shape)} voxels of "
        f"{envelope.spacing * 100:g} cm, origin ({fmt(f'{v:.2f}' for v in envelope.origin)}) m",
        f"occupied {envelope.occupied_count} voxels = "
        f"{envelope.occupied_count * envelope.spacing**3:.3f} m^3, "
        f"fill fraction {envelope.fill_fraction:.4f}",
        f"bounding box min ({fmt(f'{v:.2f}' for v in lo)}) max ({fmt(f'{v:.2f}' for v in hi)}) m",
        f"joint source {envelope.joint_source}",
    ]
    if any(envelope.tip_offset_m):
        lines.insert(1, f"tip offset ({fmt(f'{v:.4f}' for v in envelope.tip_offset_m)}) m")
    if path is not None:
        lines.append(f"file {path} {Path(path).stat().st_size} bytes (limit {MAX_BYTES})")
    return "\n".join(lines)


def plot(envelope: ReachabilityEnvelope, out: Path = DEFAULT_PLOT) -> Path:
    """Three orthographic projections: how many occupied voxels lie behind each pixel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = envelope.occupied
    lo = envelope.origin
    hi = envelope.origin + np.array(grid.shape) * envelope.spacing
    views = [
        ("top view (looking down -z)", 2, (0, 1), "x forward (m)", "y left (m)"),
        ("side view (looking along +y)", 1, (0, 2), "x forward (m)", "z up (m)"),
        ("front view (looking along -x)", 0, (1, 2), "y left (m)", "z up (m)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for ax, (title, depth_axis, (a, b), xlabel, ylabel) in zip(axes, views, strict=True):
        depth = grid.sum(axis=depth_axis).astype(float)
        depth[depth == 0] = np.nan
        image = ax.imshow(
            depth.T,
            origin="lower",
            extent=(lo[a], hi[a], lo[b], hi[b]),
            cmap="Blues",
            interpolation="nearest",
        )
        ax.plot([0.0], [0.0], marker="+", color="#333333", markersize=12, mew=2)
        ax.annotate("pelvis", (0.0, 0.0), xytext=(6, 6), textcoords="offset points", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        ax.grid(color="#dddddd", linewidth=0.5)
        fig.colorbar(image, ax=ax, shrink=0.8, label="occupied voxels along the view")
    if envelope.waist_at_zero:
        waist = "waist at zero"
    else:
        (ylo, yhi), (rlo, rhi), (plo, phi) = envelope.waist_range_deg
        waist = (
            f"waist yaw {_fmt_deg(ylo)} to {_fmt_deg(yhi)}, roll {_fmt_deg(rlo)} to "
            f"{_fmt_deg(rhi)}, pitch {_fmt_deg(plo)} to {_fmt_deg(phi)} deg"
        )
    fig.suptitle(
        f"Right palm reachability envelope, pelvis frame, {waist} "
        f"({envelope.samples} samples, seed {envelope.seed}, "
        f"{envelope.spacing * 100:g} cm voxels, URDF forward kinematics)",
        fontsize=12,
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", type=Path, metavar="FILE", help="inspect FILE, write nothing")
    parser.add_argument("--joints", type=Path, default=DEFAULT_JOINTS, help="URDF joint dump")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="configurations")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="numpy default_rng seed")
    parser.add_argument("--spacing", type=float, default=VOXEL_SIZE_M, help="voxel size, m")
    parser.add_argument(
        "--waist-yaw-deg", type=float, default=0.0, help="waist yaw sampled in +-this, deg"
    )
    parser.add_argument(
        "--waist-roll-deg", type=float, default=0.0, help="waist roll sampled in +-this, deg"
    )
    parser.add_argument(
        "--waist-pitch-min-deg", type=float, default=0.0, help="waist pitch lower end, deg"
    )
    parser.add_argument(
        "--waist-pitch-max-deg",
        type=float,
        default=0.0,
        help="waist pitch upper end, deg (positive leans forward)",
    )
    parser.add_argument(
        "--tip-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="sampled point in the palm link frame, m (default the link origin; the committed "
        "files use the finger bases, 0.0777 -0.0016 0)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output .npz (default {DEFAULT_OUT.relative_to(REPO)}, or "
        f"{DEFAULT_WAIST_OUT.relative_to(REPO)} with a waist range)",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const=DEFAULT_PLOT,
        type=Path,
        metavar="PNG",
        help=f"also write the projections (default {DEFAULT_PLOT.relative_to(REPO)})",
    )
    args = parser.parse_args(argv)

    if args.check is not None:
        envelope = ReachabilityEnvelope.load(args.check)
        path = args.check
    else:
        requested = (
            (-abs(args.waist_yaw_deg), abs(args.waist_yaw_deg)),
            (-abs(args.waist_roll_deg), abs(args.waist_roll_deg)),
            (args.waist_pitch_min_deg, args.waist_pitch_max_deg),
        )
        waist = waist_range_deg(
            args.waist_yaw_deg,
            args.waist_roll_deg,
            args.waist_pitch_min_deg,
            args.waist_pitch_max_deg,
        )
        for name, want, got in zip(joint_maps.WAIST, requested, waist, strict=True):
            if tuple(want) != tuple(got):
                print(f"note: {name} range {want} deg clipped to the URDF limits: {got}")
        out = args.out
        if out is None:
            out = DEFAULT_OUT if waist == ZERO_WAIST_DEG else DEFAULT_WAIST_OUT
        start = time.perf_counter()
        envelope = generate(
            args.joints,
            args.samples,
            args.seed,
            args.spacing,
            waist_range=waist,
            tip_offset_m=tuple(args.tip_offset),
        )
        path = envelope.save(out)
        print(f"sampled and wrote in {time.perf_counter() - start:.1f} s")
    print(report(envelope, path))
    if args.plot is not None:
        print(f"plot {plot(envelope, args.plot)}")
    size = Path(path).stat().st_size
    if size > MAX_BYTES or envelope.occupied_count == 0:
        print(f"FAIL: {size} bytes (limit {MAX_BYTES}) or empty grid")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
