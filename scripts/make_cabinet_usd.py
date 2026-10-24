"""The dishwasher's plinth dressed as a low kitchen cabinet, as static primitive-collider USD (#70).

D013 keeps the compact dishwasher on a 0.55 m plinth beside the counter and asks for the plinth to
read as a low cabinet: a carcass with a door front in the counter's colour, static, nothing
articulated (D009's cabinets-as-dressing rule). Generate (the committed asset is
``dishwasher_cabinet.usda`` under source/kitchenbot/scene/assets/kitchen/; every size comes from
``kitchenbot.scene.kitchen_dims`` and ``kitchenbot.scene.dishwasher_frames``)::

    py -3.12 scripts/make_cabinet_usd.py
    py -3.12 scripts/make_cabinet_usd.py --out-dir some/other/dir

Validate (exit code 1 if the file differs from the module by more than 1 mm, a schema or colour is
wrong, the carcass footprint is not ``dishwasher_frames.plinth_box_kitchen``, or a clearance below
is too small)::

    py -3.12 scripts/make_cabinet_usd.py --check <dishwasher_cabinet.usda>

The file is in the kitchen frame of ``kitchen_dims`` (origin on the floor at the middle of the
counter's front edge, ``x`` along the counter, ``y`` towards the wall, ``z`` up), like
``counter.usda``, so it spawns at the environment origin unrotated. Default prim
``/DishwasherCabinet``, six boxes, all ``UsdGeom.Cube`` with ``UsdPhysics.CollisionAPI``:

* ``Carcass/Body``: the plinth's footprint from ``CABINET_KICK_HEIGHT_M`` up to
  ``PLINTH_GAP_M`` below the dishwasher body's underside;
* ``Carcass/Kick``: the toe-kick plate under it, set back ``CABINET_KICK_RECESS_M`` on the front;
* ``Door/Panel``: the door front, ``CABINET_DOOR_THICKNESS_M`` proud of the carcass on the face the
  dishwasher's own front looks at (its ``-x`` face; any other yaw of the dishwasher is refused),
  inset by ``CABINET_DOOR_GAP_M`` on the sides and bottom and ``CABINET_DOOR_TOP_REVEAL_M`` below
  the carcass top;
* ``Handle/Bar``, ``Handle/PostA``, ``Handle/PostB``: a bar handle on two posts, centred across the
  door, near its top.

Colours: the counter's ``BODY_COLOR`` for the carcass, kick plate and door, a steel grey for the
handle. One friction material is bound to every collider with the ``physics`` purpose. There is no
``RigidBodyAPI``: the cabinet is static. Output is deterministic.

Clearances (computed from the boxes, in the writer and in ``--check``): the door panel and the
handle against the open dishwasher door (``dishwasher_frames.footprints_kitchen()["door"]``, which
hangs in front of the unit and whose underside is the limit) must be at least
``CABINET_MIN_DOOR_CLEARANCE_M``, and against the robot's right leg (a slab at
``x <= ROBOT_RIGHT_LEG_SURFACE_X_M``) at least ``CABINET_MIN_LEG_CLEARANCE_M``. A clearance between
two boxes is the largest per-axis gap, a lower bound of their distance.

``pxr`` (the ``usd-core`` wheel) is imported lazily, so ``--help`` works without it. Nothing here
imports Isaac Sim or Isaac Lab.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from kitchenbot.scene import dishwasher_frames as df
from kitchenbot.scene import kitchen_dims as kd
from kitchenbot.scene import usd_prims as up

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "source/kitchenbot/scene/assets/kitchen"
FILE_NAME = "dishwasher_cabinet.usda"

CABINET_PATH = "/DishwasherCabinet"
MATERIAL_NAME = "PhysicsMaterial"
CARCASS_BODY = f"{CABINET_PATH}/Carcass/Body"
CARCASS_KICK = f"{CABINET_PATH}/Carcass/Kick"
DOOR_PANEL = f"{CABINET_PATH}/Door/Panel"
HANDLE_PARTS = (
    f"{CABINET_PATH}/Handle/Bar",
    f"{CABINET_PATH}/Handle/PostA",
    f"{CABINET_PATH}/Handle/PostB",
)

#: The counter's body colour (make_counter_usd.BODY_COLOR; the tests keep the two equal).
BODY_COLOR = (0.62, 0.50, 0.38)
#: Steel grey, the colour make_counter_usd gives the sink (SINK_COLOR).
HANDLE_COLOR = (0.70, 0.72, 0.75)


def _front_normal(yaw: float) -> tuple[float, float]:
    """Kitchen-frame direction the dishwasher's front face looks at (root-link ``-y``, yawed)."""
    return (math.sin(yaw), -math.cos(yaw))


def cabinet_boxes(
    scale: float = kd.RACK_SCALE, pose: tuple = kd.DISHWASHER_POSE_KITCHEN
) -> list[tuple[str, tuple, tuple, tuple]]:
    """``(path, lo, hi, colour)`` of the six cabinet boxes in the kitchen frame."""
    (cx, cy, _), (sx, sy, sz) = df.plinth_box_kitchen(scale, pose)  # raises if not axis-aligned
    nx, ny = _front_normal(pose[3])
    if abs(nx + 1.0) > 1e-9 or abs(ny) > 1e-9:
        raise ValueError(
            "the cabinet door is built on the -x face of the plinth: the dishwasher yaw must be "
            f"-90 deg, got {math.degrees(pose[3]):g} deg"
        )
    x0, x1 = cx - sx / 2.0, cx + sx / 2.0
    y0, y1 = cy - sy / 2.0, cy + sy / 2.0
    top = sz - kd.PLINTH_GAP_M
    kick_h, gap = kd.CABINET_KICK_HEIGHT_M, kd.CABINET_DOOR_GAP_M
    thick = kd.CABINET_DOOR_THICKNESS_M
    door_top = top - kd.CABINET_DOOR_TOP_REVEAL_M
    door_bottom = kick_h + gap
    length, section = kd.CABINET_HANDLE_LENGTH_M, kd.CABINET_HANDLE_SECTION_M
    projection = kd.CABINET_HANDLE_PROJECTION_M
    if not 0.0 < kick_h < top:
        raise ValueError("CABINET_KICK_HEIGHT_M must lie between the floor and the carcass top")
    if not 0.0 < kd.CABINET_KICK_RECESS_M < sx:
        raise ValueError("CABINET_KICK_RECESS_M must be positive and less than the carcass depth")
    if min(gap, thick, kd.CABINET_DOOR_TOP_REVEAL_M) <= 0.0 or door_top <= door_bottom:
        raise ValueError("the door panel needs positive thickness, gaps and height")
    if not 0.0 < section < projection or length <= 2.0 * section or length > (y1 - y0) - 2 * gap:
        raise ValueError("the handle must be a bar on two posts that fits across the door")
    y_mid = (y0 + y1) / 2.0
    bar_top = door_top - kd.CABINET_HANDLE_DROP_M
    bar_bottom = bar_top - section
    if bar_bottom <= door_bottom:
        raise ValueError("the handle must lie within the door panel's height")

    face = x0 - thick  # the door's outer face
    bar_x = face - projection
    return [
        (CARCASS_BODY, (x0, y0, kick_h), (x1, y1, top), BODY_COLOR),
        (CARCASS_KICK, (x0 + kd.CABINET_KICK_RECESS_M, y0, 0.0), (x1, y1, kick_h), BODY_COLOR),
        (DOOR_PANEL, (face, y0 + gap, door_bottom), (x0, y1 - gap, door_top), BODY_COLOR),
        (
            HANDLE_PARTS[0],
            (bar_x, y_mid - length / 2.0, bar_bottom),
            (bar_x + section, y_mid + length / 2.0, bar_top),
            HANDLE_COLOR,
        ),
        (
            HANDLE_PARTS[1],
            (bar_x + section, y_mid - length / 2.0, bar_bottom),
            (face, y_mid - length / 2.0 + section, bar_top),
            HANDLE_COLOR,
        ),
        (
            HANDLE_PARTS[2],
            (bar_x + section, y_mid + length / 2.0 - section, bar_bottom),
            (face, y_mid + length / 2.0, bar_top),
            HANDLE_COLOR,
        ),
    ]


def open_door_box(
    scale: float = kd.RACK_SCALE, pose: tuple = kd.DISHWASHER_POSE_KITCHEN
) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` of the open dishwasher door, from ``dishwasher_frames.footprints_kitchen``."""
    corners, (z0, z1) = df.footprints_kitchen(scale, pose=pose)["door"]
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    return np.array([min(xs), min(ys), z0]), np.array([max(xs), max(ys), z1])


def separation(a, b) -> float:
    """Largest per-axis gap between boxes ``(lo, hi)`` ``a`` and ``b``: negative if they overlap."""
    (a_lo, a_hi), (b_lo, b_hi) = a, b
    gaps = np.maximum(np.asarray(b_lo) - np.asarray(a_hi), np.asarray(a_lo) - np.asarray(b_hi))
    return float(gaps.max())


def clearances(
    boxes: dict, scale: float = kd.RACK_SCALE, pose: tuple = kd.DISHWASHER_POSE_KITCHEN
) -> dict[str, float]:
    """Metres between the door panel and handle and the open dishwasher door and the right leg.

    ``boxes`` maps a prim path to its ``(lo, hi)`` corners, from :func:`cabinet_boxes` or read
    back from a file. The leg is the slab ``x <= ROBOT_RIGHT_LEG_SURFACE_X_M``.
    """
    door = open_door_box(scale, pose)
    leg_x = kd.ROBOT_RIGHT_LEG_SURFACE_X_M
    handle = [boxes[p] for p in HANDLE_PARTS]
    return {
        "door_panel_to_open_door_m": separation(boxes[DOOR_PANEL], door),
        "handle_to_open_door_m": min(separation(box, door) for box in handle),
        "door_panel_to_right_leg_m": float(boxes[DOOR_PANEL][0][0] - leg_x),
        "handle_to_right_leg_m": min(float(box[0][0] - leg_x) for box in handle),
    }


def clearance_problems(found: dict[str, float]) -> list[str]:
    """One message per clearance of :func:`clearances` below its minimum."""
    limits = {
        "door_panel_to_open_door_m": kd.CABINET_MIN_DOOR_CLEARANCE_M,
        "handle_to_open_door_m": kd.CABINET_MIN_DOOR_CLEARANCE_M,
        "door_panel_to_right_leg_m": kd.CABINET_MIN_LEG_CLEARANCE_M,
        "handle_to_right_leg_m": kd.CABINET_MIN_LEG_CLEARANCE_M,
    }
    return [
        f"{name[: -len('_m')]} is {found[name] * 1000:.1f} mm, less than {limit * 1000:.0f} mm"
        for name, limit in limits.items()
        if found[name] < limit
    ]


# --------------------------------------------------------------------------------------------
# Writing (needs usd-core)
# --------------------------------------------------------------------------------------------


def write_cabinet_usd(
    out_path, scale: float = kd.RACK_SCALE, pose: tuple = kd.DISHWASHER_POSE_KITCHEN
) -> Path:
    """Write the cabinet to ``out_path`` (text USD when the suffix is ``.usda``)."""
    from pxr import UsdGeom

    boxes = cabinet_boxes(scale, pose)  # raises ValueError before touching the disk
    bad = clearance_problems(clearances({p: (lo, hi) for p, lo, hi, _ in boxes}, scale, pose))
    if bad:
        raise ValueError("clearance too small: " + "; ".join(bad))
    out_path = Path(out_path)
    (cx, cy, _), (sx, sy, sz) = df.plinth_box_kitchen(scale, pose)
    stage = up.new_stage(
        out_path,
        CABINET_PATH,
        f"Dishwasher plinth dressed as a low cabinet: carcass {sx:g} x {sy:g} x "
        f"{sz - kd.PLINTH_GAP_M:g} m centred at ({cx:g}, {cy:g}), door panel "
        f"{kd.CABINET_DOOR_THICKNESS_M * 1000:g} mm proud of the -x face, bar handle, "
        f"{kd.CABINET_KICK_HEIGHT_M * 1000:g} mm toe kick. Kitchen frame: origin on the floor at "
        f"the middle of the counter's front edge, x along the counter, y towards the wall, z up. "
        f"Static: no rigid body. Generated by scripts/make_cabinet_usd.py; do not edit by hand.",
    )
    material = up.define_physics_material(
        stage, f"{CABINET_PATH}/{MATERIAL_NAME}", kd.CABINET_FRICTION, kd.SURFACE_RESTITUTION
    )
    for scope in ("Carcass", "Door", "Handle"):
        UsdGeom.Scope.Define(stage, f"{CABINET_PATH}/{scope}")
    for path, lo, hi, colour in boxes:
        up.define_box(stage, path, lo, hi, material, colour)
    stage.GetRootLayer().Save()
    return out_path


# --------------------------------------------------------------------------------------------
# Checking (needs usd-core)
# --------------------------------------------------------------------------------------------


def _check_boxes(stage, expected: dict, info: dict) -> None:
    """Compare every expected box with the prim of that path; record what was found."""
    from pxr import UsdGeom

    problems = info["problems"]
    cache = UsdGeom.XformCache()
    found = {}
    for path, (lo, hi, colour) in expected.items():
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
        shown = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
        if not shown or len(shown) != 1 or np.abs(np.array(shown[0]) - colour).max() > 1e-6:
            problems.append(f"{path} display colour is {shown}, expected {list(colour)}")
    info["found"] = found


def check_cabinet_usd(
    path, scale: float = kd.RACK_SCALE, pose: tuple = kd.DISHWASHER_POSE_KITCHEN
) -> dict:
    """Reopen ``path`` and compare it with the modules; ``problems`` empty means it passes."""
    from pxr import Usd

    path = Path(path)
    info: dict = {"kind": "cabinet", "path": path.as_posix(), "bytes": path.stat().st_size}
    info["problems"] = problems = []
    stage = Usd.Stage.Open(str(path))
    problems.extend(up.stage_problems(stage))
    root = stage.GetDefaultPrim()
    if not root or str(root.GetPath()) != CABINET_PATH:
        problems.append(f"default prim is {root.GetPath() if root else None}, not {CABINET_PATH}")
        return info
    problems.extend(up.collider_problems(root, kd.CABINET_FRICTION, kd.SURFACE_RESTITUTION))

    try:
        boxes = cabinet_boxes(scale, pose)
    except ValueError as exc:
        problems.append(f"kitchen_dims cannot describe a cabinet: {exc}")
        return info
    info["colliders"] = len(up.collider_prims(root))
    info["expected_colliders"] = len(boxes)
    if info["colliders"] != len(boxes):
        problems.append(f"{info['colliders']} colliders, expected {len(boxes)}")
    _check_boxes(stage, {p: (lo, hi, colour) for p, lo, hi, colour in boxes}, info)
    found = info["found"]
    if found:
        info["bbox_min"] = np.min([b[0] for b in found.values()], axis=0)
        info["bbox_max"] = np.max([b[1] for b in found.values()], axis=0)
    if CARCASS_BODY not in found or CARCASS_KICK not in found:
        return info

    # The carcass is the plinth: its footprint is plinth_box_kitchen and its top stops PLINTH_GAP_M
    # short of the dishwasher body's underside.
    (cx, cy, _), (sx, sy, sz) = df.plinth_box_kitchen(scale, pose)
    carcass_lo = np.minimum(found[CARCASS_BODY][0], found[CARCASS_KICK][0])
    carcass_hi = np.maximum(found[CARCASS_BODY][1], found[CARCASS_KICK][1])
    want_lo = np.array([cx - sx / 2.0, cy - sy / 2.0, 0.0])
    want_hi = np.array([cx + sx / 2.0, cy + sy / 2.0, sz - kd.PLINTH_GAP_M])
    info["carcass_min"], info["carcass_max"] = carcass_lo, carcass_hi
    for name, got, want in (("min", carcass_lo, want_lo), ("max", carcass_hi, want_hi)):
        off = np.abs(got - want)
        if off.max() > up.TOLERANCE_M:
            problems.append(
                f"carcass {name} is {np.round(got, 4).tolist()}, plinth_box_kitchen gives "
                f"{np.round(want, 4).tolist()} (off by {off.max() * 1000:.1f} mm)"
            )
    body_underside = df.footprints_kitchen(scale, pose=pose)["body"][1][0]
    info["gap_under_body_m"] = float(body_underside - carcass_hi[2])
    if abs(info["gap_under_body_m"] - kd.PLINTH_GAP_M) > up.TOLERANCE_M:
        problems.append(
            f"carcass top is {info['gap_under_body_m'] * 1000:.1f} mm under the dishwasher body, "
            f"PLINTH_GAP_M is {kd.PLINTH_GAP_M * 1000:.1f} mm"
        )

    if all(p in found for p in (DOOR_PANEL, *HANDLE_PARTS)):
        info["clearances"] = clearances(found, scale, pose)
        problems.extend(clearance_problems(info["clearances"]))
    return info


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
    if "carcass_min" in info:
        lo, hi = info["carcass_min"], info["carcass_max"]
        lines.append(
            f"carcass (m): {hi[0] - lo[0]:.4f} x {hi[1] - lo[1]:.4f} x {hi[2] - lo[2]:.4f}, "
            f"x {lo[0]:+.4f} to {hi[0]:+.4f}, y {lo[1]:+.4f} to {hi[1]:+.4f} "
            f"(plinth_box_kitchen footprint)"
        )
        lines.append(
            f"gap under the dishwasher body = {info['gap_under_body_m'] * 1000:.1f} mm "
            f"(PLINTH_GAP_M: {kd.PLINTH_GAP_M * 1000:.1f} mm)"
        )
    if "clearances" in info:
        c = info["clearances"]
        door, leg = kd.CABINET_MIN_DOOR_CLEARANCE_M * 1000, kd.CABINET_MIN_LEG_CLEARANCE_M * 1000
        lines.append(
            f"clearance to the open dishwasher door's underside: door panel "
            f"{c['door_panel_to_open_door_m'] * 1000:.1f} mm, handle "
            f"{c['handle_to_open_door_m'] * 1000:.1f} mm (minimum {door:.0f} mm)"
        )
        lines.append(
            f"clearance to the right leg (x = {kd.ROBOT_RIGHT_LEG_SURFACE_X_M:.2f} m): door panel "
            f"{c['door_panel_to_right_leg_m'] * 1000:.1f} mm, handle "
            f"{c['handle_to_right_leg_m'] * 1000:.1f} mm (minimum {leg:.0f} mm)"
        )
    for problem in info["problems"]:
        lines.append(f"PROBLEM: {problem}")
    lines.append("RESULT: " + ("OK" if not info["problems"] else "FAIL"))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or check the dishwasher cabinet USD (issue #70). Sizes come from "
        "kitchenbot.scene.kitchen_dims and kitchenbot.scene.dishwasher_frames."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"where {FILE_NAME} is written (default: the committed asset)",
    )
    parser.add_argument(
        "--check",
        type=Path,
        nargs="+",
        metavar="FILE",
        help="validate these cabinet files instead of writing",
    )
    args = parser.parse_args(argv)

    try:
        import pxr  # noqa: F401
    except ImportError:
        print(
            "make_cabinet_usd.py needs the OpenUSD Python package (module 'pxr').\n"
            "Install it with:  pip install usd-core   (or: pip install -e '.[assets]')",
            file=sys.stderr,
        )
        return 2

    if args.check:
        failed = False
        for file in args.check:
            if not file.is_file():
                parser.error(f"no such file: {file}")
            info = check_cabinet_usd(file)
            print(format_report(info))
            failed = failed or bool(info["problems"])
        return 1 if failed else 0

    try:
        path = write_cabinet_usd(args.out_dir / FILE_NAME)
    except ValueError as exc:
        parser.error(str(exc))
    info = check_cabinet_usd(path)
    print(f"wrote {path.as_posix()} ({info['bytes']} bytes): {info['colliders']} colliders")
    for problem in info["problems"]:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    return 1 if info["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
