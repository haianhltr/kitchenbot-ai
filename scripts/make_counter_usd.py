"""Procedural kitchen counter with a sink, and the floor, as static primitive-collider USD (#47).

Generate (the committed assets are counter.usda and floor.usda under
source/kitchenbot/scene/assets/kitchen/; every size comes from ``kitchenbot.scene.kitchen_dims``)::

    py -3.12 scripts/make_counter_usd.py
    py -3.12 scripts/make_counter_usd.py --out-dir some/other/dir

Validate (exit code 1 if the counter-top face is not at ``COUNTER_HEIGHT_M``, the floor top not at
z = 0, or anything else differs from ``kitchen_dims`` by more than 1 mm or a schema is missing)::

    py -3.12 scripts/make_counter_usd.py --check <counter.usda> <floor.usda>

Both files share the kitchen frame of ``kitchen_dims``: origin on the floor at the middle of the
counter's front edge, ``x`` along the counter, ``y`` from the front edge towards the wall, ``z`` up.

``counter.usda`` (default prim ``/Counter``) is nine boxes, all ``UsdGeom.Cube`` with
``UsdPhysics.CollisionAPI``:

* ``Top/Left``, ``Top/Right``, ``Top/Front``, ``Top/Back``: the top slab, split around the sink
  opening, ``COUNTER_TOP_THICKNESS_M`` thick with its face at ``COUNTER_HEIGHT_M``;
* ``Body/Left``, ``Body/Right``, ``Body/Front``, ``Body/Back``: the block under the slab, split the
  same way, so the four faces beside the opening are the basin walls;
* ``Body/SinkFloor``: the block under the opening, topped ``SINK_DEPTH_M`` below the counter-top
  face, which is the basin floor.

``floor.usda`` (default prim ``/Floor``) is one thin slab, ``FLOOR_SIZE_M`` across and
``FLOOR_THICKNESS_M`` thick, whose top face is z = 0. Isaac Lab's ``GroundPlaneCfg`` is the
alternative to this slab; #10 picks one.

Each file has one friction material (``kitchen_dims``) bound to every collider with the ``physics``
purpose. There is no ``RigidBodyAPI`` anywhere: counter and floor are static. Output is
deterministic. ``pxr`` (the ``usd-core`` wheel) is imported lazily, so ``--help`` works without it.
Nothing here imports Isaac Sim or Isaac Lab.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from kitchenbot.scene import kitchen_dims as kd
from kitchenbot.scene import usd_prims as up

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "source/kitchenbot/scene/assets/kitchen"

COUNTER_PATH = "/Counter"
FLOOR_PATH = "/Floor"
FLOOR_SLAB_PATH = f"{FLOOR_PATH}/Slab"
MATERIAL_NAME = "PhysicsMaterial"

BODY_COLOR = (0.62, 0.50, 0.38)
TOP_COLOR = (0.84, 0.84, 0.82)
SINK_COLOR = (0.70, 0.72, 0.75)
FLOOR_COLOR = (0.45, 0.45, 0.47)


def sink_opening() -> tuple[float, float, float, float]:
    """``(x0, x1, y0, y1)`` of the sink opening in the kitchen frame."""
    half_x, half_y = kd.SINK_SIZE_M[0] / 2.0, kd.SINK_SIZE_M[1] / 2.0
    return (
        kd.SINK_CENTRE_X_M - half_x,
        kd.SINK_CENTRE_X_M + half_x,
        kd.SINK_CENTRE_Y_M - half_y,
        kd.SINK_CENTRE_Y_M + half_y,
    )


def counter_boxes() -> list[tuple[str, tuple, tuple, tuple]]:
    """``(path, lo, hi, colour)`` of the nine counter boxes in the kitchen frame."""
    width, depth, height = kd.COUNTER_WIDTH_M, kd.COUNTER_DEPTH_M, kd.COUNTER_HEIGHT_M
    x_min, x_max = -width / 2.0, width / 2.0
    sx0, sx1, sy0, sy1 = sink_opening()
    z_slab = height - kd.COUNTER_TOP_THICKNESS_M
    z_sink = height - kd.SINK_DEPTH_M
    if not (x_min < sx0 < sx1 < x_max and 0.0 < sy0 < sy1 < depth):
        raise ValueError("the sink opening must lie strictly inside the counter top")
    if not 0.0 < z_sink < z_slab:
        raise ValueError("SINK_DEPTH_M must exceed the top thickness and stay above the floor")

    def layer(name: str, z0: float, z1: float, colour: tuple) -> list:
        return [
            (f"{COUNTER_PATH}/{name}/Left", (x_min, 0.0, z0), (sx0, depth, z1), colour),
            (f"{COUNTER_PATH}/{name}/Right", (sx1, 0.0, z0), (x_max, depth, z1), colour),
            (f"{COUNTER_PATH}/{name}/Front", (sx0, 0.0, z0), (sx1, sy0, z1), colour),
            (f"{COUNTER_PATH}/{name}/Back", (sx0, sy1, z0), (sx1, depth, z1), colour),
        ]

    return [
        *layer("Top", z_slab, height, TOP_COLOR),
        *layer("Body", 0.0, z_slab, BODY_COLOR),
        (f"{COUNTER_PATH}/Body/SinkFloor", (sx0, sy0, 0.0), (sx1, sy1, z_sink), SINK_COLOR),
    ]


def floor_box() -> tuple[tuple, tuple]:
    """``(lo, hi)`` of the floor slab: top face at z = 0, centred on the origin in x and y."""
    half_x, half_y = kd.FLOOR_SIZE_M[0] / 2.0, kd.FLOOR_SIZE_M[1] / 2.0
    return (-half_x, -half_y, -kd.FLOOR_THICKNESS_M), (half_x, half_y, 0.0)


# --------------------------------------------------------------------------------------------
# Writing (needs usd-core)
# --------------------------------------------------------------------------------------------


def write_counter_usd(out_path) -> Path:
    """Write the counter and sink to ``out_path`` (text USD when the suffix is ``.usda``)."""
    from pxr import UsdGeom

    boxes = counter_boxes()  # raises ValueError before touching the disk
    out_path = Path(out_path)
    stage = up.new_stage(
        out_path,
        COUNTER_PATH,
        f"Procedural kitchen counter: {kd.COUNTER_WIDTH_M:g} m wide, {kd.COUNTER_DEPTH_M:g} m "
        f"deep, top face at z = {kd.COUNTER_HEIGHT_M:g} m, sink "
        f"{kd.SINK_SIZE_M[0]:g} x {kd.SINK_SIZE_M[1]:g} x {kd.SINK_DEPTH_M:g} m. Kitchen frame: "
        f"origin on the floor at the middle of the front edge, x along the counter, y towards "
        f"the wall, z up. Generated by scripts/make_counter_usd.py; do not edit by hand.",
    )
    material = up.define_physics_material(
        stage, f"{COUNTER_PATH}/{MATERIAL_NAME}", kd.COUNTER_FRICTION, kd.SURFACE_RESTITUTION
    )
    UsdGeom.Scope.Define(stage, f"{COUNTER_PATH}/Top")
    UsdGeom.Scope.Define(stage, f"{COUNTER_PATH}/Body")
    for path, lo, hi, colour in boxes:
        up.define_box(stage, path, lo, hi, material, colour)
    stage.GetRootLayer().Save()
    return out_path


def write_floor_usd(out_path) -> Path:
    """Write the floor slab to ``out_path`` (text USD when the suffix is ``.usda``)."""
    out_path = Path(out_path)
    stage = up.new_stage(
        out_path,
        FLOOR_PATH,
        f"Procedural floor slab: {kd.FLOOR_SIZE_M[0]:g} x {kd.FLOOR_SIZE_M[1]:g} m, "
        f"{kd.FLOOR_THICKNESS_M * 1000:g} mm thick, top face at z = 0. Kitchen frame: centred on "
        f"the origin, z up. Generated by scripts/make_counter_usd.py; do not edit by hand.",
    )
    material = up.define_physics_material(
        stage, f"{FLOOR_PATH}/{MATERIAL_NAME}", kd.FLOOR_FRICTION, kd.SURFACE_RESTITUTION
    )
    lo, hi = floor_box()
    up.define_box(stage, FLOOR_SLAB_PATH, lo, hi, material, FLOOR_COLOR)
    stage.GetRootLayer().Save()
    return out_path


# --------------------------------------------------------------------------------------------
# Checking (needs usd-core)
# --------------------------------------------------------------------------------------------


def _check_boxes(stage, expected: dict, info: dict) -> None:
    """Compare every expected box with the prim of that path; record the union of the bounds."""
    from pxr import UsdGeom

    problems = info["problems"]
    cache = UsdGeom.XformCache()
    found = {}
    for path, (lo, hi) in expected.items():
        prim = stage.GetPrimAtPath(path)
        if not prim:
            problems.append(f"missing prim {path}")
            continue
        try:
            got_lo, got_hi = up.world_aabb(prim, cache)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        found[path] = (got_lo, got_hi)
        for name, got, want in (("min", got_lo, lo), ("max", got_hi, hi)):
            off = np.abs(got - np.asarray(want))
            if off.max() > up.TOLERANCE_M:
                problems.append(
                    f"{path} {name} is {np.round(got, 4).tolist()}, expected "
                    f"{np.round(want, 4).tolist()} (off by {off.max() * 1000:.1f} mm)"
                )
    info["found"] = found
    if found:
        info["bbox_min"] = np.min([b[0] for b in found.values()], axis=0)
        info["bbox_max"] = np.max([b[1] for b in found.values()], axis=0)


def check_counter_usd(path) -> dict:
    """Reopen ``path`` and compare it with ``kitchen_dims``; ``problems`` empty means it passes."""
    from pxr import Usd

    path = Path(path)
    info: dict = {"kind": "counter", "path": path.as_posix(), "bytes": path.stat().st_size}
    info["problems"] = problems = []
    stage = Usd.Stage.Open(str(path))
    problems.extend(up.stage_problems(stage))
    root = stage.GetDefaultPrim()
    if not root or str(root.GetPath()) != COUNTER_PATH:
        problems.append(f"default prim is {root.GetPath() if root else None}, not {COUNTER_PATH}")
        return info
    problems.extend(up.collider_problems(root, kd.COUNTER_FRICTION, kd.SURFACE_RESTITUTION))

    boxes = counter_boxes()
    expected = {p: (lo, hi) for p, lo, hi, _ in boxes}
    info["colliders"] = len(up.collider_prims(root))
    info["expected_colliders"] = len(boxes)
    if info["colliders"] != len(boxes):
        problems.append(f"{info['colliders']} colliders, expected {len(boxes)}")
    _check_boxes(stage, expected, info)

    found = info["found"]
    tops = [found[p][1][2] for p in found if p.startswith(f"{COUNTER_PATH}/Top/")]
    info["top_face_z"] = max(tops) if tops else None
    if tops and max(abs(z - kd.COUNTER_HEIGHT_M) for z in tops) > up.TOLERANCE_M:
        problems.append(
            f"counter-top face is at z = {min(tops):.4f} to {max(tops):.4f} m, expected "
            f"{kd.COUNTER_HEIGHT_M:.4f} m"
        )
    sink_floor = found.get(f"{COUNTER_PATH}/Body/SinkFloor")
    info["sink_floor_z"] = float(sink_floor[1][2]) if sink_floor else None

    # The basin must be open: no collider may enter the volume above the sink floor.
    sx0, sx1, sy0, sy1 = sink_opening()
    open_lo = np.array([sx0, sy0, kd.COUNTER_HEIGHT_M - kd.SINK_DEPTH_M])
    open_hi = np.array([sx1, sy1, kd.COUNTER_HEIGHT_M])
    tol = up.TOLERANCE_M
    for prim_path, (lo, hi) in found.items():
        inside = np.minimum(hi, open_hi) - np.maximum(lo, open_lo)
        if np.all(inside > tol):
            problems.append(f"{prim_path} reaches into the open basin of the sink")
    return info


def check_floor_usd(path) -> dict:
    """Reopen ``path`` and compare it with ``kitchen_dims``; ``problems`` empty means it passes."""
    from pxr import Usd

    path = Path(path)
    info: dict = {"kind": "floor", "path": path.as_posix(), "bytes": path.stat().st_size}
    info["problems"] = problems = []
    stage = Usd.Stage.Open(str(path))
    problems.extend(up.stage_problems(stage))
    root = stage.GetDefaultPrim()
    if not root or str(root.GetPath()) != FLOOR_PATH:
        problems.append(f"default prim is {root.GetPath() if root else None}, not {FLOOR_PATH}")
        return info
    problems.extend(up.collider_problems(root, kd.FLOOR_FRICTION, kd.SURFACE_RESTITUTION))
    info["colliders"] = len(up.collider_prims(root))
    info["expected_colliders"] = 1
    if info["colliders"] != 1:
        problems.append(f"{info['colliders']} colliders, expected 1")
    lo, hi = floor_box()
    _check_boxes(stage, {FLOOR_SLAB_PATH: (lo, hi)}, info)
    slab = info["found"].get(FLOOR_SLAB_PATH)
    info["top_face_z"] = float(slab[1][2]) if slab else None
    return info


def check_usd(path) -> dict:
    """Check a counter or a floor file, chosen by its default prim."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    name = str(root.GetPath()) if root else None
    if name == FLOOR_PATH:
        return check_floor_usd(path)
    if name == COUNTER_PATH:
        return check_counter_usd(path)
    path = Path(path)
    return {
        "kind": "unknown",
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "problems": [f"default prim is {name}, expected {COUNTER_PATH} or {FLOOR_PATH}"],
    }


def format_report(info: dict) -> str:
    """Compact ASCII report for ``--check``."""
    lines = [f"file: {info['path']} ({info['bytes']} bytes), {info['kind']}"]
    if "colliders" in info:
        lines.append(f"colliders: {info['colliders']} (expected {info['expected_colliders']})")
    if "bbox_min" in info:
        lo, hi = info["bbox_min"], info["bbox_max"]
        lines.append(
            f"bounds (m): x {lo[0]:+.3f} to {hi[0]:+.3f}, y {lo[1]:+.3f} to {hi[1]:+.3f}, "
            f"z {lo[2]:+.3f} to {hi[2]:+.3f}"
        )
    if info.get("top_face_z") is not None:
        want = kd.COUNTER_HEIGHT_M if info["kind"] == "counter" else 0.0
        lines.append(f"top face z = {info['top_face_z']:.4f} m (kitchen_dims: {want:.4f} m)")
    if info.get("sink_floor_z") is not None:
        want = kd.COUNTER_HEIGHT_M - kd.SINK_DEPTH_M
        lines.append(f"sink floor z = {info['sink_floor_z']:.4f} m (kitchen_dims: {want:.4f} m)")
    for problem in info["problems"]:
        lines.append(f"PROBLEM: {problem}")
    lines.append("RESULT: " + ("OK" if not info["problems"] else "FAIL"))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or check the procedural counter and floor USD (issue #47). Sizes "
        "come from kitchenbot.scene.kitchen_dims."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="where counter.usda and floor.usda are written (default: the committed assets)",
    )
    parser.add_argument(
        "--check",
        type=Path,
        nargs="+",
        metavar="FILE",
        help="validate these counter and floor files instead of writing",
    )
    args = parser.parse_args(argv)

    try:
        import pxr  # noqa: F401
    except ImportError:
        print(
            "make_counter_usd.py needs the OpenUSD Python package (module 'pxr').\n"
            "Install it with:  pip install usd-core   (or: pip install -e '.[assets]')",
            file=sys.stderr,
        )
        return 2

    if args.check:
        failed = False
        for file in args.check:
            if not file.is_file():
                parser.error(f"no such file: {file}")
            info = check_usd(file)
            print(format_report(info))
            failed = failed or bool(info["problems"])
        return 1 if failed else 0

    try:
        paths = [
            write_counter_usd(args.out_dir / "counter.usda"),
            write_floor_usd(args.out_dir / "floor.usda"),
        ]
    except ValueError as exc:
        parser.error(str(exc))
    failed = False
    for path in paths:
        info = check_usd(path)
        print(
            f"wrote {path.as_posix()} ({info['bytes']} bytes): {info['colliders']} colliders, "
            f"top face z = {info['top_face_z']:.4f} m"
        )
        for problem in info["problems"]:
            print(f"PROBLEM: {problem}", file=sys.stderr)
        failed = failed or bool(info["problems"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
