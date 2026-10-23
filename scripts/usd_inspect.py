#!/usr/bin/env python
"""Inspect a USD file without a simulator: prim tree, joints, meshes, collision approximations.

    python scripts/usd_inspect.py <file.usd|usda|usdc|usdz> [--depth 3] [--max-list 25]

Prints, using only the OpenUSD Python package (``pip install usd-core``):

* stage metadata and the number of composed layers
* the prim tree (type and path) down to ``--depth``
* every ``UsdPhysics.RevoluteJoint`` and ``UsdPhysics.PrismaticJoint`` with axis,
  ``physics:lowerLimit`` / ``physics:upperLimit``, the ``physics:body0`` / ``physics:body1``
  targets and any joint drive; other joint types are only counted
* the number of ``Mesh`` prims (and total points)
* which prims carry ``PhysicsCollisionAPI`` / ``PhysicsMeshCollisionAPI`` and their
  ``physics:approximation``

Nothing here imports Isaac Sim, Isaac Lab, omni or torch.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    from pxr import Usd, UsdGeom, UsdPhysics

    HAVE_PXR = True
except ImportError:  # reported by main() with install instructions
    HAVE_PXR = False

MISSING_PXR = (
    "usd_inspect.py needs the OpenUSD Python package (module 'pxr').\n"
    "Install it with:  pip install usd-core   (or: pip install -e '.[assets]')"
)
JOINT_UNITS = {"RevoluteJoint": "deg", "PrismaticJoint": "m"}


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def attr(prim, name):
    a = prim.GetAttribute(name)
    return a.Get() if a and a.HasAuthoredValue() else None


def targets(prim, rel_name) -> str:
    rel = prim.GetRelationship(rel_name)
    paths = [str(p) for p in rel.GetTargets()] if rel else []
    return ", ".join(paths) if paths else "-"


def print_tree(stage, depth: int, max_prims: int) -> None:
    """Depth-limited prim tree; collapsed subtrees show their descendant count."""
    printed = 0

    def walk(prim, level: int) -> None:
        nonlocal printed
        if printed >= max_prims:
            return
        n_desc = sum(1 for _ in Usd.PrimRange(prim)) - 1
        collapsed = level >= depth and n_desc
        suffix = f"  (+{n_desc} descendants)" if collapsed else ""
        print(f"{'  ' * level}{prim.GetPath()}  [{prim.GetTypeName() or 'untyped'}]{suffix}")
        printed += 1
        if collapsed:
            return
        for child in prim.GetChildren():
            walk(child, level + 1)

    for root in stage.GetPseudoRoot().GetChildren():
        walk(root, 0)
    if printed >= max_prims:
        print(f"... tree truncated at {max_prims} prims (raise --max-prims)")


def drive_summary(prim) -> str:
    out = []
    for token in ("angular", "linear"):
        if prim.HasAPI(UsdPhysics.DriveAPI, token):
            api = UsdPhysics.DriveAPI(prim, token)
            vals = {
                "stiffness": api.GetStiffnessAttr().Get(),
                "damping": api.GetDampingAttr().Get(),
                "target": api.GetTargetPositionAttr().Get(),
                "maxForce": api.GetMaxForceAttr().Get(),
            }
            body = " ".join(f"{k}={fmt(v)}" for k, v in vals.items())
            out.append(f"drive[{token}] {body}")
    return "; ".join(out) if out else "no DriveAPI"


def collect_joints(stage) -> tuple[list, Counter]:
    moving, other = [], Counter()
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            moving.append(prim)
        elif prim.IsA(UsdPhysics.Joint):
            other[prim.GetTypeName() or "PhysicsJoint"] += 1
    return moving, other


def print_joints(stage) -> list:
    moving, other = collect_joints(stage)
    print(f"Revolute/prismatic joints: {len(moving)}")
    for prim in moving:
        kind = prim.GetTypeName().replace("Physics", "")
        unit = JOINT_UNITS.get(kind, "")
        lo, hi = attr(prim, "physics:lowerLimit"), attr(prim, "physics:upperLimit")
        print(f"  {prim.GetPath()}  [{kind}]")
        print(
            f"    axis={fmt(attr(prim, 'physics:axis') or 'X (default)')}  "
            f"lower={fmt(lo)} upper={fmt(hi)} {unit}"
        )
        print(
            f"    body0={targets(prim, 'physics:body0')}  body1={targets(prim, 'physics:body1')}"
        )
        print(f"    {drive_summary(prim)}")
    if other:
        print("Other joints (counted only): " + ", ".join(f"{k} x{v}" for k, v in other.items()))
    print_link_extents(stage, moving)
    return moving


def print_link_extents(stage, joints: list) -> None:
    """Axis-aligned size in metres of every body a joint connects (default pose)."""
    paths: dict[str, None] = {}
    for prim in joints:
        for rel in ("physics:body0", "physics:body1"):
            for target in prim.GetRelationship(rel).GetTargets():
                paths[str(target)] = None
    if not paths:
        return
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    mpu = stage.GetMetadata("metersPerUnit") or 1.0
    print("Link extents (m, axis-aligned, default pose):")
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        size = box.GetSize()
        lo = box.GetMin()
        print(
            f"  {path}: size=({size[0] * mpu:.3f}, {size[1] * mpu:.3f}, {size[2] * mpu:.3f}) "
            f"min=({lo[0] * mpu:.3f}, {lo[1] * mpu:.3f}, {lo[2] * mpu:.3f})"
        )


def print_meshes(stage) -> list:
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    points = 0
    for prim in meshes:
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        points += len(pts) if pts is not None else 0
    print(f"Mesh prims: {len(meshes)}  (total points: {points})")
    return meshes


def print_collision(stage, meshes: list, max_list: int) -> None:
    rows = []
    for prim in stage.Traverse():
        has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
        has_mesh_col = prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        if has_col or has_mesh_col:
            approx = attr(prim, "physics:approximation")
            rows.append((prim, has_col, has_mesh_col, approx))
    print(f"Prims with collision schemas: {len(rows)}")
    summary = Counter(
        (
            "CollisionAPI" if c else "-",
            "MeshCollisionAPI" if m else "-",
            str(a) if a is not None else "unset (triangle mesh)",
        )
        for _, c, m, a in rows
    )
    for (c, m, a), n in sorted(summary.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d} x {c:12s} {m:16s} approximation={a}")
    for prim, c, m, a in rows[:max_list]:
        print(
            f"    {prim.GetPath()} [{prim.GetTypeName()}] "
            f"Collision={'Y' if c else 'n'} MeshCollision={'Y' if m else 'n'} "
            f"approximation={fmt(a)}"
        )
    if len(rows) > max_list:
        print(f"    ... {len(rows) - max_list} more (raise --max-list)")
    colliding = {p.GetPath() for p, _, _, _ in rows}

    def covered(prim) -> bool:
        path = prim.GetPath()
        while path.pathElementCount > 0:
            if path in colliding:
                return True
            path = path.GetParentPath()
        return False

    bare = sum(1 for m in meshes if not covered(m))
    print(f"Mesh prims with no collision schema on themselves or an ancestor: {bare}")
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print(f"RigidBodyAPI prims: {len(bodies)}   ArticulationRootAPI prims: {len(roots)}")
    for prim in roots[:max_list]:
        print(f"    articulation root: {prim.GetPath()}")


def inspect(path: Path, depth: int, max_prims: int, max_list: int, show_tree: bool) -> int:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        print(f"could not open {path} as a USD stage", file=sys.stderr)
        return 1
    up = stage.GetMetadata("upAxis")
    mpu = stage.GetMetadata("metersPerUnit")
    default = stage.GetDefaultPrim()
    n_prims = sum(1 for _ in stage.Traverse())
    layers = [layer.identifier for layer in stage.GetUsedLayers()]
    print(f"== {path}  ({path.stat().st_size / 1e6:.2f} MB)")
    print(
        f"upAxis={up} metersPerUnit={mpu} "
        f"defaultPrim={default.GetPath() if default else '-'} prims={n_prims} "
        f"composed layers={len(layers)}"
    )
    if show_tree:
        print(f"\n-- Prim tree (depth {depth})")
        print_tree(stage, depth, max_prims)
    print("\n-- Joints")
    print_joints(stage)
    print("\n-- Geometry")
    meshes = print_meshes(stage)
    print("\n-- Collision")
    print_collision(stage, meshes, max_list)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file", type=Path, help=".usd, .usda, .usdc or .usdz file")
    ap.add_argument("--depth", type=int, default=3, help="prim tree depth (default 3)")
    ap.add_argument("--max-prims", type=int, default=200, help="cap on printed tree lines")
    ap.add_argument("--max-list", type=int, default=25, help="cap on listed collision prims")
    ap.add_argument("--no-tree", action="store_true", help="skip the prim tree")
    args = ap.parse_args(argv)
    if not HAVE_PXR:
        print(MISSING_PXR, file=sys.stderr)
        return 2
    if not args.file.is_file():
        print(f"no such file: {args.file}", file=sys.stderr)
        return 1
    return inspect(args.file, args.depth, args.max_prims, args.max_list, not args.no_tree)


if __name__ == "__main__":
    sys.exit(main())
