"""Primitive-collider USD authoring and checking helpers for the kitchen generators (issue #47).

Shared by ``scripts/make_rack_tines_usd.py`` and ``scripts/make_counter_usd.py`` so the two write
and validate the same way as the plates (#30) did: text USD, metres, Z up, ``UsdGeom`` primitives
(``Cube``, ``Capsule``, ``Cylinder``) with ``UsdPhysics.CollisionAPI``, one ``UsdShade.Material``
with ``UsdPhysics.MaterialAPI`` bound with the ``physics`` purpose, and no rigid body: the file is
static or becomes part of a body that a scene config names.

``pxr`` (the ``usd-core`` wheel) is imported inside the functions, so importing this module and
``--help`` of the scripts work without it. Nothing here imports Isaac Sim, Isaac Lab, omni or torch.
Values are rounded to 1e-9 m before they are written so the text files are deterministic and free
of float noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PHYSICS_PURPOSE = "physics"  # material:binding:physics, the purpose PhysX reads
TOLERANCE_M = 1e-3  # what the --check modes accept between a file and kitchen_dims
_DECIMALS = 9
_PRIMITIVES = ("Cube", "Capsule", "Cylinder")


def clean(value: float) -> float:
    """Round to 1e-9 and drop the sign of zero, so equal inputs give identical text."""
    return round(float(value), _DECIMALS) + 0.0


def new_stage(out_path, root_path: str, doc: str):
    """Create a text stage at ``out_path`` (replacing a file there) with a default root Xform."""
    from pxr import Usd, UsdGeom, UsdPhysics

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.GetRootLayer().documentation = doc
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    return stage


def define_physics_material(stage, path: str, friction: float, restitution: float):
    """A ``UsdShade.Material`` with ``UsdPhysics.MaterialAPI`` (static = dynamic friction)."""
    from pxr import UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(clean(friction))
    api.CreateDynamicFrictionAttr(clean(friction))
    api.CreateRestitutionAttr(clean(restitution))
    return material


def _finish_collider(gprim, material, color) -> None:
    from pxr import Gf, UsdPhysics, UsdShade, Vt

    gprim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    prim = gprim.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, PHYSICS_PURPOSE
    )


def define_box(stage, path: str, lo, hi, material, color):
    """An axis-aligned box from corner ``lo`` to ``hi``, a unit ``Cube`` with a scale op."""
    from pxr import Gf, UsdGeom, Vt

    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    size, centre = hi - lo, (hi + lo) / 2.0
    if np.any(size <= 0.0):
        raise ValueError(f"box {path} has a non-positive size {size.tolist()}")
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)]))
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(clean, centre)))
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(clean, size)))
    _finish_collider(cube, material, color)
    return cube


def define_capsule_z(stage, path: str, centre, radius: float, length: float, material, color):
    """A vertical capsule, ``length`` long over both hemispherical caps, centred at ``centre``."""
    from pxr import Gf, UsdGeom, Vt

    if length <= 2.0 * radius:
        raise ValueError(f"capsule {path}: length {length} must exceed its diameter {2 * radius}")
    shaft = clean(length - 2.0 * radius)
    r = clean(radius)
    capsule = UsdGeom.Capsule.Define(stage, path)
    capsule.CreateAxisAttr(UsdGeom.Tokens.z)
    capsule.CreateRadiusAttr(r)
    capsule.CreateHeightAttr(shaft)
    half = shaft / 2.0 + r
    capsule.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-r, -r, -half), Gf.Vec3f(r, r, half)]))
    UsdGeom.Xformable(capsule).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*map(clean, centre))
    )
    _finish_collider(capsule, material, color)
    return capsule


# --------------------------------------------------------------------------------------------
# Reading and checking
# --------------------------------------------------------------------------------------------


def world_aabb(prim, xform_cache) -> tuple[np.ndarray, np.ndarray]:
    """World-space axis-aligned bounds of a ``Cube``, ``Capsule`` or ``Cylinder`` prim.

    Computed from the schema attributes (``size``, ``radius``, ``height``, ``axis``) and the
    local-to-world transform, not from the authored ``extent``, so a hand edit of either shows up.
    """
    from pxr import Gf, UsdGeom

    if prim.IsA(UsdGeom.Cube):
        half = np.full(3, abs(float(UsdGeom.Cube(prim).GetSizeAttr().Get())) / 2.0)
    elif prim.IsA(UsdGeom.Capsule) or prim.IsA(UsdGeom.Cylinder):
        shape = UsdGeom.Capsule(prim) if prim.IsA(UsdGeom.Capsule) else UsdGeom.Cylinder(prim)
        radius = float(shape.GetRadiusAttr().Get())
        half = np.full(3, radius)
        along = float(shape.GetHeightAttr().Get()) / 2.0
        if prim.IsA(UsdGeom.Capsule):
            along += radius
        half["XYZ".index(str(shape.GetAxisAttr().Get()))] = along
    else:
        raise ValueError(f"{prim.GetPath()} is a {prim.GetTypeName()}, not a primitive collider")
    matrix = xform_cache.GetLocalToWorldTransform(prim)
    corners = [
        np.array(matrix.Transform(Gf.Vec3d(sx * half[0], sy * half[1], sz * half[2])))
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ]
    return np.min(corners, axis=0), np.max(corners, axis=0)


def collider_prims(root) -> list:
    """Every prim at or under ``root`` that has ``UsdPhysics.CollisionAPI``, in path order."""
    from pxr import Usd, UsdPhysics

    return [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]


def stage_problems(stage) -> list[str]:
    """Metadata a physics-ready layer needs: Z up, metres, kilograms, a default prim."""
    from pxr import UsdGeom, UsdPhysics

    problems = []
    if str(UsdGeom.GetStageUpAxis(stage)) != "Z":
        problems.append(f"up axis is {UsdGeom.GetStageUpAxis(stage)}, expected Z")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        problems.append(f"metersPerUnit is {UsdGeom.GetStageMetersPerUnit(stage)}, expected 1.0")
    if UsdPhysics.GetStageKilogramsPerUnit(stage) != 1.0:
        problems.append("kilogramsPerUnit is not 1.0")
    if not stage.GetDefaultPrim():
        problems.append("no default prim")
    return problems


def collider_problems(root, friction: float, restitution: float) -> list[str]:
    """Check the primitive-collider contract under ``root``; an empty list means it holds.

    Every collider is a Cube, Capsule or Cylinder with CollisionAPI and without MeshCollisionAPI,
    has a physics-purpose material with the expected friction bound to it, and nothing at or under
    ``root`` is a mesh or has RigidBodyAPI (the file is static, or the rack link is the body).
    """
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    problems = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            problems.append(f"{prim.GetPath()} is a mesh; primitive colliders only")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            problems.append(f"{prim.GetPath()} has RigidBodyAPI; the geometry must stay static")
    for prim in collider_prims(root):
        path = prim.GetPath()
        if prim.GetTypeName() not in _PRIMITIVES:
            problems.append(f"{path} has CollisionAPI but is a {prim.GetTypeName()}")
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            problems.append(f"{path} has MeshCollisionAPI; primitive colliders only")
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            problems.append(f"{path} has collision disabled")
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(PHYSICS_PURPOSE)[0]
        if not material or not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
            problems.append(f"{path} has no physics-purpose material with MaterialAPI bound")
            continue
        api = UsdPhysics.MaterialAPI(material.GetPrim())
        for label, got, want in (
            ("static friction", api.GetStaticFrictionAttr().Get(), friction),
            ("dynamic friction", api.GetDynamicFrictionAttr().Get(), friction),
            ("restitution", api.GetRestitutionAttr().Get(), restitution),
        ):
            if got is None or abs(got - want) > 1e-6:
                problems.append(f"{path} material {label} is {got}, expected {want}")
    return problems
