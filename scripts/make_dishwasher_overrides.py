"""USD overrides layer for the ArtVIP dishwasher_2: colliders, limits, default targets (#59).

Generate (the committed layer is
source/kitchenbot/scene/assets/dishwasher/dishwasher_2_overrides.usda; the ArtVIP file itself is
git-ignored and fetched with ``scripts/fetch_assets.py --set dishwasher``)::

    py -3.12 scripts/make_dishwasher_overrides.py
    py -3.12 scripts/make_dishwasher_overrides.py --rack-travel 0.4 --out some.usda   # #11

Validate (exit code 1 on any problem)::

    py -3.12 scripts/make_dishwasher_overrides.py --check <FILE>.usda [--rack-travel 0.2]

The layer never copies the ArtVIP file: it references it by a relative path into
``scene/assets/third_party/`` (``MANIFEST.md``) and only adds opinions. Under the default prim
``/Dishwasher`` (where Isaac Lab's spawner writes the pose and the scale):

* ``ArtVIP``: references ``model_dishwasher_2.usda`` (its default prim ``/root``) and translates it
  by minus ``kitchen_dims.DISHWASHER_ROOT_LINK_IN_ASSET_M``, so the articulation root link
  ``E_body_5`` sits exactly at the layer origin. Isaac Lab writes ``init_state.pos`` both to the
  spawned prim and, at a reset, to the root link (``articulation.py:4056``), so the two must agree;
* ``over`` prims on the 13 collider meshes: ``physics:approximation`` = ``convexDecomposition`` on
  the two racks and ``convexHull`` on the body and the door (docs/architecture.md section 10; ArtVIP
  ships ``sdf`` everywhere), with ``PhysxSDFMeshCollisionAPI`` deleted;
* ``E_body_5``: ``OmniScriptingAPI`` deleted and ``omni:scripting:scripts`` emptied. ArtVIP attaches
  ``resource/dishwasher_control.py``, a Kit behaviour script that rewrites the door drive's
  stiffness and snaps the door to open or closed every frame, fighting our actuators;
* the door joint: limits 0..90 deg and drive target 90 deg (open) stated explicitly; the lower-rack
  joint: limits ``-rack_travel..0`` and drive target ``-rack_travel``. The default travel is #29's
  0.2 m, which this issue keeps; #11 raises it with ``--rack-travel 0.4``
  (``kitchen_dims.DISHWASHER_RACK_TRAVEL_M``) and regenerates. The upper rack is left as authored.

``--with-tines`` is the fallback for the tines: it adds ``RackTines`` (a reference to
``rack_tines.usda`` at ``kitchen_dims.RACK_TINES_POSE_IN_LINK`` under the compensating scale) under
the lower-rack link inside this layer, for the case where #11 finds that Isaac Lab's nested
``AssetBaseCfg`` (``scene/dishwasher_cfg.py``) does not attach it. Not used by default.

``--check`` checks the layer's own content (no ArtVIP file needed, so it runs on CI), then, when the
referenced file resolves, composes the stage with usd-core and asserts what the layer set on the
composed prims: every collider's approximation and that none is ``sdf``, the deleted schemas, the
joint limits and targets, one articulation root at the layer origin, and the ``kitchen_dims``
measurements (body size, door hinge, rack link, link scale) against the asset to 1 mm. It also
references ``rack_tines.usda`` under the lower-rack link the way ``dishwasher_cfg.py`` does and
checks that the slot markers land on ``dishwasher_frames.slot_poses_root`` to 1 mm with a composed
tine scale of exactly 1. A missing asset skips the composition (``--require-asset`` makes it fail).

``pxr`` (the ``usd-core`` wheel) is imported lazily, so ``--help`` works without it. Nothing here
imports Isaac Sim, Isaac Lab, omni or torch.
"""

import argparse
import math
import sys
from pathlib import Path

from kitchenbot.scene import dishwasher_frames as df
from kitchenbot.scene import kitchen_dims as kd
from kitchenbot.scene import usd_prims as up

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "source/kitchenbot/scene/assets"
DEFAULT_OUT = ASSET_DIR / "dishwasher/dishwasher_2_overrides.usda"
RACK_TINES_USD = ASSET_DIR / "dishwasher/rack_tines.usda"
#: The ArtVIP file as the layer references it, relative to the layer's directory.
ARTVIP_REFERENCE = f"../third_party/{df.ARTVIP_DISHWASHER_2}"
TINES_REFERENCE = "./rack_tines.usda"

ROOT_PATH = "/Dishwasher"
ASSET_PATH = f"{ROOT_PATH}/ArtVIP"
TINES_NAME = "RackTines"
TOLERANCE_M = 1e-3

HULL = "convexHull"
DECOMPOSITION = "convexDecomposition"
SDF_API = "PhysxSDFMeshCollisionAPI"
SCRIPTING_API = "OmniScriptingAPI"

#: The 13 collider meshes of dishwasher_2 (#29: all ``sdf``), relative to the asset root, with the
#: approximation this layer gives each.
COLLIDERS: dict[str, str] = {
    f"{df.BODY_LINK}/P_c4698ba0b0ef07e2": HULL,  # invisible in ArtVIP, still a collider
    f"{df.BODY_LINK}/P_63d8aef9c4af07e2": HULL,
    f"{df.BODY_LINK}/P_f7a07bce31ef07e2": HULL,
    f"{df.BODY_LINK}/P_7b637ac7626f07e2": HULL,
    f"{df.BODY_LINK}/P_75a52a7e554207e2": HULL,
    f"{df.BODY_LINK}/E_composite_1/P_a56573c2ba5b55d3": HULL,
    f"{df.BODY_LINK}/E_composite_1/P_13f95c09b3dfeed8": HULL,
    f"{df.BODY_LINK}/E_composite_1/P_c578ff39705a2ed8": HULL,
    f"{df.DOOR_LINK}/P_1eb2d8bb09b410d8": HULL,
    f"{df.DOOR_LINK}/P_51065ed96c8f07e2": HULL,
    f"{df.DOOR_LINK}/handle/P_4ab085662d6b07e2": HULL,
    f"{df.UPPER_RACK_LINK}/P_1440b71c964e2ed8": DECOMPOSITION,
    f"{df.LOWER_RACK_LINK}/P_23f8dca1790b2ed8": DECOMPOSITION,
}
DOOR_JOINT_PATH = f"{df.DOOR_LINK}/{df.DOOR_JOINT}"
LOWER_RACK_JOINT_PATH = f"{df.LOWER_RACK_LINK}/{df.LOWER_RACK_JOINT}"
UPPER_RACK_JOINT_PATH = f"{df.UPPER_RACK_LINK}/{df.UPPER_RACK_JOINT}"


def joint_values(rack_travel: float) -> dict[str, dict[str, float]]:
    """The joint attributes the layer authors, by joint path under the asset root."""
    return {
        DOOR_JOINT_PATH: {
            "physics:lowerLimit": 0.0,
            "physics:upperLimit": kd.DISHWASHER_DOOR_OPEN_DEG,
            "drive:angular:physics:targetPosition": kd.DISHWASHER_DOOR_OPEN_DEG,
        },
        LOWER_RACK_JOINT_PATH: {
            "physics:lowerLimit": -rack_travel,
            "physics:upperLimit": 0.0,
            "drive:linear:physics:targetPosition": -rack_travel,
        },
    }


def asset_offset() -> tuple[float, float, float]:
    """Translate of ``ArtVIP`` that puts the root link at the layer origin."""
    return tuple(-v + 0.0 for v in kd.DISHWASHER_ROOT_LINK_IN_ASSET_M)


def tines_xform(
    asset_scale: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(translate, scale) of the tines prim under the lower-rack link."""
    x, y, z, _yaw = kd.RACK_TINES_POSE_IN_LINK
    return (x, y, z), df.rack_tines_scale_in_link(asset_scale)


# --------------------------------------------------------------------------------------------
# Writing (needs usd-core)
# --------------------------------------------------------------------------------------------


def _over_path(layer, rel_path: str):
    """``over`` specs for every component of ``rel_path`` under the asset prim; the last one."""
    from pxr import Sdf

    spec = layer.GetPrimAtPath(ASSET_PATH)
    for name in rel_path.split("/"):
        child = spec.nameChildren.get(name)
        spec = child if child is not None else Sdf.PrimSpec(spec, name, Sdf.SpecifierOver)
    return spec


def _set(spec, name: str, type_name, value, uniform: bool = False):
    from pxr import Sdf

    variability = Sdf.VariabilityUniform if uniform else Sdf.VariabilityVarying
    attr = spec.attributes.get(name) or Sdf.AttributeSpec(spec, name, type_name, variability)
    attr.default = value
    return attr


def _add_xform_ops(spec, translate, scale=None) -> None:
    from pxr import Gf, Sdf, Vt

    ops = ["xformOp:translate"]
    _set(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*translate))
    if scale is not None:
        _set(spec, "xformOp:scale", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*scale))
        ops.append("xformOp:scale")
    _set(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, Vt.TokenArray(ops), uniform=True)


def write_overrides(
    out_path,
    rack_travel: float = -kd.DISHWASHER_RACK_LIMITS_M[0],
    with_tines: bool = False,
    asset_scale: float = kd.RACK_SCALE,
) -> Path:
    """Write the overrides layer to ``out_path`` (text USD)."""
    from pxr import Sdf

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    layer = Sdf.Layer.CreateNew(str(out_path))
    layer.pseudoRoot.SetInfo("upAxis", "Z")
    layer.pseudoRoot.SetInfo("metersPerUnit", 1.0)
    layer.pseudoRoot.SetInfo("kilogramsPerUnit", 1.0)
    layer.defaultPrim = ROOT_PATH.lstrip("/")
    flags = f"--rack-travel {rack_travel:g}" + (
        f" --with-tines --asset-scale {asset_scale:g}" if with_tines else ""
    )
    layer.documentation = (
        "Overrides for ArtVIP dishwasher_2 (#59): references the git-ignored ArtVIP file, puts the "
        "articulation root link E_body_5 at the origin, sets convexDecomposition on the racks and "
        "convexHull on the body and door, removes the ArtVIP behaviour script, door open 90 deg, "
        f"lower rack limits -{rack_travel:g}..0 m and target -{rack_travel:g} m. Generated by "
        f"scripts/make_dishwasher_overrides.py {flags}; do not edit by hand."
    )
    root = Sdf.PrimSpec(layer, ROOT_PATH.lstrip("/"), Sdf.SpecifierDef, "Xform")
    asset = Sdf.PrimSpec(root, "ArtVIP", Sdf.SpecifierDef, "Xform")
    asset.referenceList.Prepend(Sdf.Reference(ARTVIP_REFERENCE))
    _add_xform_ops(asset, asset_offset())

    body = _over_path(layer, df.BODY_LINK)
    body.SetInfo("apiSchemas", Sdf.TokenListOp.Create(deletedItems=[SCRIPTING_API]))
    _set(
        body, "omni:scripting:scripts", Sdf.ValueTypeNames.AssetArray, Sdf.AssetPathArray([]), True
    )

    for rel_path, approximation in COLLIDERS.items():
        mesh = _over_path(layer, rel_path)
        mesh.SetInfo("apiSchemas", Sdf.TokenListOp.Create(deletedItems=[SDF_API]))
        _set(mesh, "physics:approximation", Sdf.ValueTypeNames.Token, approximation, uniform=True)

    for rel_path, values in joint_values(rack_travel).items():
        joint = _over_path(layer, rel_path)
        for name, value in values.items():
            _set(joint, name, Sdf.ValueTypeNames.Float, float(value))

    if with_tines:
        link = _over_path(layer, df.LOWER_RACK_LINK)
        tines = Sdf.PrimSpec(link, TINES_NAME, Sdf.SpecifierDef, "Xform")
        tines.referenceList.Prepend(Sdf.Reference(TINES_REFERENCE))
        _add_xform_ops(tines, *tines_xform(asset_scale))

    layer.Save()
    return out_path


# --------------------------------------------------------------------------------------------
# Checking (needs usd-core)
# --------------------------------------------------------------------------------------------


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return a is not None and abs(float(a) - float(b)) <= tol


def check_layer_content(path, rack_travel: float) -> list[str]:
    """Checks on the layer's own opinions; needs no ArtVIP file."""
    from pxr import Sdf

    problems: list[str] = []
    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        return [f"cannot open {path}"]
    meta = layer.pseudoRoot
    for key, want in (("upAxis", "Z"), ("metersPerUnit", 1.0), ("kilogramsPerUnit", 1.0)):
        if meta.GetInfo(key) != want:
            problems.append(f"layer {key} is {meta.GetInfo(key)!r}, expected {want!r}")
    if layer.defaultPrim != ROOT_PATH.lstrip("/"):
        problems.append(
            f"default prim is {layer.defaultPrim!r}, expected {ROOT_PATH.lstrip('/')!r}"
        )
    asset = layer.GetPrimAtPath(ASSET_PATH)
    if asset is None:
        return problems + [f"missing {ASSET_PATH}"]
    refs = [r.assetPath for r in asset.referenceList.prependedItems]
    if refs != [ARTVIP_REFERENCE]:
        problems.append(f"{ASSET_PATH} references {refs}, expected [{ARTVIP_REFERENCE!r}]")
    translate = asset.attributes.get("xformOp:translate")
    if translate is None or any(
        not _close(a, b) for a, b in zip(translate.default, asset_offset(), strict=True)
    ):
        problems.append(f"{ASSET_PATH} translate is not {asset_offset()}")

    body = layer.GetPrimAtPath(f"{ASSET_PATH}/{df.BODY_LINK}")
    if body is None or SCRIPTING_API not in body.GetInfo("apiSchemas").deletedItems:
        problems.append(f"{df.BODY_LINK} does not delete {SCRIPTING_API}")
    scripts = body.attributes.get("omni:scripting:scripts") if body is not None else None
    if scripts is None or len(scripts.default) != 0:
        problems.append(f"{df.BODY_LINK} omni:scripting:scripts is not emptied")

    for rel_path, approximation in COLLIDERS.items():
        spec = layer.GetPrimAtPath(f"{ASSET_PATH}/{rel_path}")
        if spec is None:
            problems.append(f"no over for collider {rel_path}")
            continue
        attr = spec.attributes.get("physics:approximation")
        if attr is None or attr.default != approximation:
            got = None if attr is None else attr.default
            problems.append(f"{rel_path} approximation is {got!r}, expected {approximation!r}")
        if SDF_API not in spec.GetInfo("apiSchemas").deletedItems:
            problems.append(f"{rel_path} does not delete {SDF_API}")

    for rel_path, values in joint_values(rack_travel).items():
        spec = layer.GetPrimAtPath(f"{ASSET_PATH}/{rel_path}")
        for name, want in values.items():
            attr = None if spec is None else spec.attributes.get(name)
            if attr is None or not _close(attr.default, want):
                got = None if attr is None else attr.default
                problems.append(f"{rel_path} {name} is {got}, expected {want:g}")
    return problems


def _api_schemas(prim) -> list[str]:
    op = prim.GetMetadata("apiSchemas")
    return list(op.ApplyOperations([])) if op is not None else []


def _mesh_bounds(prim, cache):
    """World-space AABB of every Mesh at or under ``prim`` (points, not extents)."""
    import numpy as np
    from pxr import Usd, UsdGeom

    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Mesh):
            pts = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(), dtype=float)
            m = np.asarray(cache.GetLocalToWorldTransform(p), dtype=float)
            w = (np.c_[pts, np.ones(len(pts))] @ m)[:, :3]
            lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
    return lo, hi


def resolve_asset(path) -> Path:
    """Where the layer's ArtVIP reference points, as a file path."""
    return (Path(path).resolve().parent / ARTVIP_REFERENCE).resolve()


def check_composed(path, rack_travel: float, info: dict) -> list[str]:
    """Open the composed stage and assert the overrides took effect, plus the tine attachment."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    problems: list[str] = []
    tol = TOLERANCE_M
    stage = Usd.Stage.Open(str(path))
    problems.extend(up.stage_problems(stage))
    root = stage.GetPrimAtPath(ASSET_PATH)
    if not root or not root.GetPrimStack() or len(list(root.GetChildren())) == 0:
        return problems + [f"{ASSET_PATH} did not compose any ArtVIP prims"]
    cache = UsdGeom.XformCache()

    def expect(label: str, got: float, want: float, t: float = tol) -> None:
        if abs(got - want) > t:
            problems.append(f"{label} is {got:.5f}, expected {want:.5f}")

    # Colliders: the set, their approximations, the deleted schema.
    found = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROOT_PATH)):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Mesh):
            rel = str(prim.GetPath())[len(ASSET_PATH) + 1 :]
            found[rel] = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if SDF_API in _api_schemas(prim):
                problems.append(f"{rel} still applies {SDF_API}")
    info["colliders"] = len(found)
    if set(found) != set(COLLIDERS):
        problems.append(
            f"collider meshes differ from the layer: missing {sorted(set(COLLIDERS) - set(found))},"
            " "
            f"not covered {sorted(set(found) - set(COLLIDERS))}"
        )
    for rel, approximation in found.items():
        if approximation == "sdf" or approximation != COLLIDERS.get(rel, approximation):
            problems.append(f"{rel} composes to approximation {approximation!r}")
    counts: dict[str, int] = {}
    for approximation in found.values():
        counts[approximation] = counts.get(approximation, 0) + 1
    info["approximations"] = counts

    # Behaviour script gone, one articulation root, at the layer origin.
    body = stage.GetPrimAtPath(f"{ASSET_PATH}/{df.BODY_LINK}")
    if SCRIPTING_API in _api_schemas(body):
        problems.append(f"{df.BODY_LINK} still applies {SCRIPTING_API}")
    scripts = body.GetAttribute("omni:scripting:scripts").Get()
    if scripts:
        problems.append(f"{df.BODY_LINK} still lists scripts {list(scripts)}")
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if roots != [str(body.GetPath())]:
        problems.append(f"articulation roots are {roots}, expected [{body.GetPath()}]")
    body_world = cache.GetLocalToWorldTransform(body)
    if not Gf.IsClose(body_world, Gf.Matrix4d(1.0), 1e-6):
        problems.append(f"{df.BODY_LINK} is not at the layer origin: {body_world}")

    # Joints: what the layer set, composed.
    for rel_path, values in joint_values(rack_travel).items():
        joint = stage.GetPrimAtPath(f"{ASSET_PATH}/{rel_path}")
        for name, want in values.items():
            attr = joint.GetAttribute(name)
            got = attr.Get() if attr else None
            if got is None or abs(got - want) > 1e-6:
                problems.append(f"{rel_path} {name} composes to {got}, expected {want:g}")
    door = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"{ASSET_PATH}/{DOOR_JOINT_PATH}"))
    rack = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(f"{ASSET_PATH}/{LOWER_RACK_JOINT_PATH}"))
    upper = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(f"{ASSET_PATH}/{UPPER_RACK_JOINT_PATH}"))
    if door.GetAxisAttr().Get() != "X" or rack.GetAxisAttr().Get() != "Y":
        problems.append("door axis must be X and rack axis Y (#29)")
    info["joints"] = {
        "door": (door.GetLowerLimitAttr().Get(), door.GetUpperLimitAttr().Get()),
        "lower rack": (rack.GetLowerLimitAttr().Get(), rack.GetUpperLimitAttr().Get()),
        "upper rack": (upper.GetLowerLimitAttr().Get(), upper.GetUpperLimitAttr().Get()),
    }

    # kitchen_dims measurements against the asset (root-link frame = layer frame here).
    lo, hi = _mesh_bounds(body, cache)
    for axis, want in enumerate(kd.DISHWASHER_BODY_SIZE_M):
        expect(f"body size {'xyz'[axis]}", hi[axis] - lo[axis], want)
        expect(f"body min {'xyz'[axis]}", lo[axis], 0.0)
    hinge = door.GetLocalPos0Attr().Get()
    for axis, want in enumerate(kd.DISHWASHER_DOOR_HINGE_M):
        expect(f"door hinge {'xyz'[axis]}", hinge[axis], want)
    door_lo, door_hi = _mesh_bounds(stage.GetPrimAtPath(f"{ASSET_PATH}/{df.DOOR_LINK}"), cache)
    expect("door length", door_hi[2] - hinge[2], kd.DISHWASHER_DOOR_LENGTH_M)
    expect("door span x min", door_lo[0], kd.DISHWASHER_DOOR_SPAN_X_M[0])
    expect("door span x max", door_hi[0], kd.DISHWASHER_DOOR_SPAN_X_M[1])
    link = stage.GetPrimAtPath(f"{ASSET_PATH}/{df.LOWER_RACK_LINK}")
    link_world = cache.GetLocalToWorldTransform(link)
    for axis, want in enumerate(kd.DISHWASHER_LOWER_RACK_LINK_M):
        expect(f"lower rack link {'xyz'[axis]}", link_world.ExtractTranslation()[axis], want, 1e-6)
    link_scale = Gf.Vec3d(link_world.GetRow3(0)).GetLength()
    expect("lower rack link x scale", link_scale, kd.DISHWASHER_LOWER_RACK_LINK_SCALE_X, 1e-6)
    rack_lo, rack_hi = _mesh_bounds(link, cache)
    expect("rack width", rack_hi[0] - rack_lo[0], kd.RACK_WIDTH_M)
    expect("rack depth", rack_hi[1] - rack_lo[1], kd.RACK_DEPTH_M)
    expect("rack height", rack_hi[2] - rack_lo[2], kd.DISHWASHER_RACK_HEIGHT_M)
    up_lo, _ = _mesh_bounds(stage.GetPrimAtPath(f"{ASSET_PATH}/{df.UPPER_RACK_LINK}"), cache)
    expect("upper rack underside", up_lo[2], kd.DISHWASHER_UPPER_RACK_UNDERSIDE_Z_M)
    rack_mesh = next(c for c in link.GetChildren() if c.IsA(UsdGeom.Mesh))
    pts = np.asarray(UsdGeom.Mesh(rack_mesh).GetPointsAttr().Get(), dtype=float)
    floor_top = float(pts[pts[:, 2] < 0.02][:, 2].max())  # the floor wires, link-local
    expect("rack floor top in the link", floor_top, kd.RACK_TINES_POSE_IN_LINK[2])

    # The tines under the lower-rack link, as the layer has them or as dishwasher_cfg adds them.
    # This stage is the layer on its own, at asset scale 1, so the check composes at scale 1.
    tines_path = f"{link.GetPath()}/{TINES_NAME}"
    if not stage.GetPrimAtPath(tines_path):
        translate, scale = tines_xform(1.0)
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            tines = UsdGeom.Xform.Define(stage, tines_path)
            tines.GetPrim().GetReferences().AddReference(str(RACK_TINES_USD))
            tines.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translate))
            tines.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))
        info["tines"] = "composed here the way dishwasher_cfg.py attaches them"
    else:
        info["tines"] = "in the layer (--with-tines)"
    cache = UsdGeom.XformCache()
    worst = 0.0
    slots = df.slot_poses_root(1.0, 0.0)
    for i, (sx, sy, sz, _yaw) in enumerate(slots, start=1):
        marker = stage.GetPrimAtPath(f"{tines_path}/Slots/Slot_{i}")
        if not marker:
            problems.append(f"no slot marker Slot_{i} under {tines_path}")
            continue
        got = cache.GetLocalToWorldTransform(marker).ExtractTranslation()
        for axis, want in enumerate((sx, sy, sz)):
            worst = max(worst, abs(got[axis] - want))
            expect(f"Slot_{i} {'xyz'[axis]}", got[axis], want)
    info["max_slot_error_m"] = worst
    tine = stage.GetPrimAtPath(f"{tines_path}/Tines/Tine_neg_1")
    if tine:
        m = cache.GetLocalToWorldTransform(tine)
        lengths = [Gf.Vec3d(m.GetRow3(k)).GetLength() for k in range(3)]
        info["tine_scale"] = lengths
        if any(abs(v - 1.0) > 1e-6 for v in lengths):
            problems.append(f"composed tine scale is {lengths}, expected 1 on every axis")
    base = stage.GetPrimAtPath(f"{tines_path}/BaseFrame")
    if base:
        b_lo, b_hi = up.world_aabb(base, cache)
        for axis in range(2):
            expect(f"tine base min {'xy'[axis]} vs rack", b_lo[axis], rack_lo[axis])
            expect(f"tine base max {'xy'[axis]} vs rack", b_hi[axis], rack_hi[axis])
        floor_z = kd.DISHWASHER_LOWER_RACK_LINK_M[2] + kd.RACK_TINES_POSE_IN_LINK[2]
        expect("tine base underside vs rack floor", b_lo[2], floor_z)
    return problems


def check_overrides(path, rack_travel: float, require_asset: bool = False) -> dict:
    """Content checks, then composition when the asset is there; ``problems`` empty means OK."""
    path = Path(path)
    info: dict = {"path": path.as_posix(), "bytes": path.stat().st_size, "problems": []}
    info["problems"].extend(check_layer_content(path, rack_travel))
    asset = resolve_asset(path)
    info["asset"] = asset.as_posix()
    if asset.is_file():
        info["composed"] = True
        info["problems"].extend(check_composed(path, rack_travel, info))
    else:
        info["composed"] = False
        if require_asset:
            info["problems"].append(f"ArtVIP file not found at {asset} (--require-asset)")
    return info


def format_report(info: dict) -> str:
    lines = [f"file: {info['path']} ({info['bytes']} bytes)"]
    if info.get("composed"):
        lines.append(f"composed with: {info['asset']}")
        if "colliders" in info:
            lines.append(f"colliders: {info['colliders']} {info.get('approximations', {})}")
        for name, (lo, hi) in info.get("joints", {}).items():
            lines.append(f"joint limits {name}: {lo:g} .. {hi:g}")
        if "max_slot_error_m" in info:
            lines.append(
                f"tines: {info['tines']}; slot markers vs dishwasher_frames.slot_poses_root: "
                f"max {info['max_slot_error_m'] * 1000:.3f} mm"
            )
        if "tine_scale" in info:
            lines.append(
                "composed tine scale: " + ", ".join(f"{v:.6f}" for v in info["tine_scale"])
            )
    else:
        lines.append(f"composition: SKIPPED, ArtVIP file not found at {info['asset']}")
        lines.append("(fetch it with: py -3.12 scripts/fetch_assets.py --set dishwasher)")
    for problem in info["problems"]:
        lines.append(f"PROBLEM: {problem}")
    lines.append("RESULT: " + ("OK" if not info["problems"] else "FAIL"))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or check the overrides layer for ArtVIP dishwasher_2 (#59). "
        "Defaults: kitchenbot.scene.kitchen_dims.",
    )
    add = parser.add_argument
    add(
        "--rack-travel",
        type=float,
        default=-kd.DISHWASHER_RACK_LIMITS_M[0],
        help="lower-rack pull-out in metres: limits -TRAVEL..0 and target -TRAVEL (default "
        "%(default)s, #29; #11 uses 0.4)",
    )
    add("--with-tines", action="store_true", help="fallback: put rack_tines.usda into the layer")
    add("--asset-scale", type=float, default=kd.RACK_SCALE, help="for --with-tines (default 1.0)")
    add("--out", type=Path, default=DEFAULT_OUT, help="output file (default: the committed layer)")
    add("--check", type=Path, metavar="FILE", help="validate FILE instead of writing")
    add(
        "--require-asset",
        action="store_true",
        help="with --check: fail when the ArtVIP file is missing instead of skipping composition",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.rack_travel) or not 0.0 < args.rack_travel <= 1.0:
        parser.error("--rack-travel must be in (0, 1] metres")
    if not math.isfinite(args.asset_scale) or args.asset_scale <= 0.0:
        parser.error("--asset-scale must be positive")
    try:
        import pxr  # noqa: F401
    except ImportError:
        print(
            "make_dishwasher_overrides.py needs the OpenUSD Python package (module 'pxr').\n"
            "Install it with:  pip install usd-core   (or: pip install -e '.[assets]')",
            file=sys.stderr,
        )
        return 2

    if args.check is not None:
        if not args.check.is_file():
            parser.error(f"no such file: {args.check}")
        info = check_overrides(args.check, args.rack_travel, args.require_asset)
        print(format_report(info))
        return 1 if info["problems"] else 0

    path = write_overrides(args.out, args.rack_travel, args.with_tines, args.asset_scale)
    info = check_overrides(path, args.rack_travel)
    print(format_report(info).replace("file: ", "wrote: ", 1))
    return 1 if info["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
