r"""Sample the right palm's reachability envelope from the URDF joint dump (#48, no simulator).

Generate the committed envelope (deterministic: the same seed and sample count give the same
grid)::

    py -3.12 scripts/make_reach_envelope.py --seed 48 --samples 2000000 \
        --out source/kitchenbot/controllers/data/right_arm_envelope.npz

Inspect a file: fill fraction, bounding box, occupied volume, seed, sample count and file size
(exit code 1 if the file is over 200 KB or the grid is empty)::

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
import sys
import time
from pathlib import Path

import numpy as np

from kitchenbot.controllers.reachability import (
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    VOXEL_SIZE_M,
    WRIST_MARGIN_DEG,
    ReachabilityEnvelope,
    build_envelope,
)
from kitchenbot.robots.kinematics import right_arm_chain

REPO = Path(__file__).resolve().parents[1]
DEFAULT_JOINTS = REPO / "tests" / "fixtures" / "g1_dex3_joints_urdf.json"
DEFAULT_OUT = REPO / "source" / "kitchenbot" / "controllers" / "data" / "right_arm_envelope.npz"
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
) -> ReachabilityEnvelope:
    """Build the right-palm envelope from a joint dump file."""
    joints, provenance = load_dump(joints_path)
    return build_envelope(
        right_arm_chain(joints),
        samples=samples,
        seed=seed,
        spacing=spacing,
        wrist_margin_deg=wrist_margin_deg,
        joint_source=provenance,
    )


def report(envelope: ReachabilityEnvelope, path: Path | None = None) -> str:
    """Plain-text summary for ``--check`` and after generation."""
    lo, hi = envelope.bounding_box
    fmt = ", ".join
    lines = [
        f"frame {envelope.frame}, tip {envelope.tip_link}, waist at zero",
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
    fig.suptitle(
        f"Right palm reachability envelope, pelvis frame, waist at zero "
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output .npz")
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
        start = time.perf_counter()
        envelope = generate(args.joints, args.samples, args.seed, args.spacing)
        path = envelope.save(args.out)
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
