"""Procedural rimmed dinner plate as a physics-ready text USD asset (issue #30, no simulator).

Generate (the committed assets are plate_22, plate_25 and plate_27 under
source/kitchenbot/scene/assets/plates/)::

    py -3.12 scripts/make_plate_usd.py --diameter 22 --out plate_22.usda

Inspect and validate (exit code 1 if the mesh is not watertight or a required schema is missing)::

    py -3.12 scripts/make_plate_usd.py --check source/kitchenbot/scene/assets/plates/plate_22.usda

The plate is a solid of revolution. Its radial cross-section (r, z) is revolved around Z with 64
segments, triangles only:

* flat centre: bottom face at z = 0, top face 10 mm above it, out to 25 mm short of the edge;
* sloped rim: over the outer 25 mm of radius both faces rise 10 mm, so the plate is 20 mm tall;
* the outer edge is a vertical 10 mm wall (the rim thickness of docs/architecture.md section 8).

The origin is the centre of the underside (the plate rests on z = 0), stage units are metres, Z up.
A thin shell is a known PhysX failure case, hence the solid body. Mass is authored (0.35 kg) and
does not scale with diameter, so the density implied by the mesh volume is only informational.

``pxr`` (the ``usd-core`` wheel) is imported lazily: the geometry and topology functions need only
numpy, so ``--help`` and the pure-numpy tests work without it. Nothing here imports Isaac Sim or
Isaac Lab.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SEGMENTS = 64  # facets around the Z axis
BODY_THICKNESS = 0.010  # m, vertical thickness of the body and of the outer rim wall
RIM_RISE = 0.010  # m, how far the rim climbs
RIM_WIDTH = 0.025  # m, radial extent of the sloped rim
MASS_KG = 0.35
FRICTION = 1.0  # static and dynamic
RESTITUTION = 0.0

ROOT_PATH = "/Plate"
MESH_PATH = "/Plate/mesh"
MATERIAL_PATH = "/Plate/PhysicsMaterial"
PHYSICS_PURPOSE = "physics"  # material:binding:physics, the purpose PhysX reads


# --------------------------------------------------------------------------------------------
# Geometry (numpy only)
# --------------------------------------------------------------------------------------------


def plate_profile(diameter_m: float) -> np.ndarray:
    """Half cross-section (r, z) from the bottom pole, round the rim, to the top pole.

    Six points: bottom centre, start of the rim on the underside, outer bottom corner, outer top
    corner, start of the rim on the top face, top centre. Both poles lie on the Z axis.
    """
    radius = diameter_m / 2.0
    if radius <= RIM_WIDTH:
        raise ValueError(
            f"diameter must exceed {2 * RIM_WIDTH * 100:.0f} cm, got {diameter_m * 100:g} cm"
        )
    flat = radius - RIM_WIDTH
    return np.array(
        [
            (0.0, 0.0),
            (flat, 0.0),
            (radius, RIM_RISE),
            (radius, RIM_RISE + BODY_THICKNESS),
            (flat, BODY_THICKNESS),
            (0.0, BODY_THICKNESS),
        ]
    )


def build_plate_mesh(diameter_m: float, segments: int = SEGMENTS) -> dict:
    """Revolve the profile into a closed triangle mesh with outward (right-handed) winding.

    Returns ``points`` (V, 3), ``face_vertex_counts`` (F,), ``face_vertex_indices`` (3F,) and
    ``normals`` (3F, 3): one normal per face corner, sharp across profile corners and smooth
    around the ring, to be authored as ``faceVarying`` so renderers do not smear the rim edges.
    """
    profile = plate_profile(diameter_m)
    n_path = len(profile)
    theta = 2.0 * np.pi * np.arange(segments) / segments

    # Vertices: bottom pole, (n_path - 2) rings of `segments`, top pole.
    rings = [
        np.stack([r * np.cos(theta), r * np.sin(theta), np.full(segments, z)], axis=1)
        for r, z in profile[1:-1]
    ]
    points = np.concatenate(
        [[[0.0, 0.0, profile[0, 1]]], *rings, [[0.0, 0.0, profile[-1, 1]]]], axis=0
    )

    def vid(j: int, k: int) -> int:
        if j == 0:
            return 0
        if j == n_path - 1:
            return len(points) - 1
        return 1 + (j - 1) * segments + (k % segments)

    indices: list[int] = []
    normals: list[np.ndarray] = []
    for j in range(n_path - 1):
        dr, dz = profile[j + 1] - profile[j]
        n_r, n_z = np.array([dz, -dr]) / np.hypot(dr, dz)  # outward normal of this path segment
        for k in range(segments):
            # Quad (a, b, c, d) = (j,k) (j,k+1) (j+1,k+1) (j+1,k) has outward winding; at a pole
            # two corners coincide and the quad collapses to the triangle that remains.
            a, b, c, d = (vid(j, k), vid(j, k + 1), vid(j + 1, k + 1), vid(j + 1, k))
            ang = (theta[k], theta[(k + 1) % segments], theta[(k + 1) % segments], theta[k])
            corner_angle = {a: ang[0], b: ang[1], c: ang[2], d: ang[3]}
            if j == 0:  # bottom pole: a == b
                tris = [(a, c, d)]
            elif j == n_path - 2:  # top pole: c == d
                tris = [(a, b, c)]
            else:
                tris = [(a, b, c), (a, c, d)]
            mid = theta[k] + np.pi / segments
            for tri in tris:
                for v in tri:
                    angle = mid if v in (0, len(points) - 1) else corner_angle[v]
                    indices.append(v)
                    normals.append(np.array([n_r * np.cos(angle), n_r * np.sin(angle), n_z]))

    return {
        "points": points,
        "face_vertex_counts": np.full(len(indices) // 3, 3, dtype=np.int32),
        "face_vertex_indices": np.array(indices, dtype=np.int32),
        "normals": np.array(normals),
    }


def analyze_topology(counts, indices, n_points: int | None = None) -> dict:
    """Edge and Euler bookkeeping for a polygon mesh given as USD face-vertex arrays.

    ``closed`` is true when every undirected edge has exactly two incident faces that traverse it
    in opposite directions (a closed, consistently oriented 2-manifold surface). ``euler`` is
    V - E + F over the vertices that are actually used; a single sphere-like solid gives 2.
    """
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    directed: Counter = Counter()
    start = 0
    for n in counts:
        face = indices[start : start + n]
        for i in range(n):
            directed[(int(face[i]), int(face[(i + 1) % n]))] += 1
        start += n
    undirected = {tuple(sorted(e)) for e in directed}
    bad_edges = sum(
        1
        for (a, b) in undirected
        if not (directed[(a, b)] == 1 and directed[(b, a)] == 1)
    )
    used = len(set(indices.tolist()))
    n_vertices, n_edges, n_faces = used, len(undirected), len(counts)
    euler = n_vertices - n_edges + n_faces
    unused = 0 if n_points is None else n_points - used
    return {
        "vertices": n_vertices,
        "edges": n_edges,
        "faces": n_faces,
        "euler": euler,
        "bad_edges": bad_edges,
        "unused_points": unused,
        "closed": bad_edges == 0 and unused == 0,
    }


def signed_volume(points, counts, indices) -> float:
    """Volume enclosed by the mesh (divergence theorem); positive when the winding is outward."""
    points = np.asarray(points, dtype=np.float64)
    total = 0.0
    start = 0
    for n in np.asarray(counts, dtype=np.int64):
        face = points[np.asarray(indices[start : start + n], dtype=np.int64)]
        for i in range(1, n - 1):
            total += float(np.dot(face[0], np.cross(face[i], face[i + 1]))) / 6.0
        start += n
    return total


# --------------------------------------------------------------------------------------------
# USD authoring and inspection (needs usd-core)
# --------------------------------------------------------------------------------------------


def write_plate_usd(diameter_m: float, out_path) -> Path:
    """Write the plate to ``out_path`` (text USD when the suffix is ``.usda``)."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    mesh_data = build_plate_mesh(diameter_m)  # raises ValueError before touching the disk
    points = mesh_data["points"].astype(np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.GetRootLayer().documentation = (
        f"Procedural rimmed plate, {diameter_m * 1000:g} mm diameter. Generated by "
        f"scripts/make_plate_usd.py --diameter {diameter_m * 100:g}; do not edit by hand."
    )

    root = UsdGeom.Xform.Define(stage, ROOT_PATH)
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(MASS_KG)

    mesh = UsdGeom.Mesh.Define(stage, MESH_PATH)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(mesh_data["face_vertex_counts"]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(mesh_data["face_vertex_indices"]))
    lo, hi = points.min(axis=0), points.max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(mesh_data["normals"].astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.93, 0.92, 0.89)]))

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr(UsdPhysics.Tokens.convexDecomposition)

    material = UsdShade.Material.Define(stage, MATERIAL_PATH)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(FRICTION)
    physics_material.CreateDynamicFrictionAttr(FRICTION)
    physics_material.CreateRestitutionAttr(RESTITUTION)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, PHYSICS_PURPOSE
    )

    stage.GetRootLayer().Save()
    return out_path


REQUIRED_APIS = {
    "root": ["PhysicsRigidBodyAPI", "PhysicsMassAPI"],
    "mesh": ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"],
    "material": ["PhysicsMaterialAPI"],
}


def check_plate_usd(path) -> dict:
    """Reopen ``path`` and gather everything the report prints. ``problems`` empty means pass."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    path = Path(path)
    info: dict = {"path": path.as_posix(), "bytes": path.stat().st_size, "problems": []}
    problems: list[str] = info["problems"]

    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    info["up_axis"] = str(UsdGeom.GetStageUpAxis(stage))
    info["meters_per_unit"] = UsdGeom.GetStageMetersPerUnit(stage)
    info["kg_per_unit"] = UsdPhysics.GetStageKilogramsPerUnit(stage)
    if info["up_axis"] != "Z":
        problems.append(f"up axis is {info['up_axis']}, expected Z")
    if info["meters_per_unit"] != 1.0:
        problems.append(f"metersPerUnit is {info['meters_per_unit']}, expected 1.0")
    if not root:
        problems.append("no default prim")
        return info
    info["default_prim"] = str(root.GetPath())

    meshes = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        problems.append(f"expected exactly 1 mesh under the default prim, found {len(meshes)}")
        return info
    mesh_prim = meshes[0]
    mesh = UsdGeom.Mesh(mesh_prim)
    info["mesh_path"] = str(mesh_prim.GetPath())

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(root).ComputeAlignedRange()
    lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
    info.update(bbox_min=lo, bbox_max=hi, bbox_size=hi - lo)

    points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64).reshape(-1, 3)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    topo = analyze_topology(counts, indices, len(points))
    volume = signed_volume(points, counts, indices)
    info.update(topo)
    info["volume_m3"] = volume
    info["watertight"] = topo["closed"] and topo["euler"] == 2 and volume > 0.0
    if not info["watertight"]:
        problems.append(
            f"mesh is not watertight (bad edges {topo['bad_edges']}, unused points "
            f"{topo['unused_points']}, V-E+F {topo['euler']}, volume {volume:.3e} m^3)"
        )

    material_binding = UsdShade.MaterialBindingAPI(mesh_prim)
    material = material_binding.ComputeBoundMaterial(PHYSICS_PURPOSE)[0]
    material_prim = material.GetPrim() if material else None

    apis = {
        "root": [str(a) for a in root.GetAppliedSchemas()],
        "mesh": [str(a) for a in mesh_prim.GetAppliedSchemas()],
        "material": [str(a) for a in material_prim.GetAppliedSchemas()] if material_prim else [],
    }
    info["apis"] = apis
    for slot, needed in REQUIRED_APIS.items():
        for api in needed:
            if api not in apis[slot]:
                problems.append(f"{slot} prim is missing {api}")

    mass_attr = UsdPhysics.MassAPI(root).GetMassAttr()
    info["mass"] = mass_attr.Get() if mass_attr and mass_attr.HasAuthoredValue() else None
    if not info["mass"] or info["mass"] <= 0.0:
        problems.append("no positive physics:mass authored on the default prim")

    approx = UsdPhysics.MeshCollisionAPI(mesh_prim).GetApproximationAttr().Get()
    info["approximation"] = str(approx) if approx is not None else None
    if info["approximation"] != "convexDecomposition":
        problems.append(f"collision approximation is {info['approximation']}")

    if material_prim is None:
        problems.append("no physics-purpose material bound to the mesh")
    else:
        pm = UsdPhysics.MaterialAPI(material_prim)
        info["material"] = {
            "path": str(material_prim.GetPath()),
            "static_friction": pm.GetStaticFrictionAttr().Get(),
            "dynamic_friction": pm.GetDynamicFrictionAttr().Get(),
            "restitution": pm.GetRestitutionAttr().Get(),
        }
    return info


def format_report(info: dict) -> str:
    """Compact ASCII report for ``--check``."""
    lines = [f"file: {info['path']} ({info['bytes']} bytes)"]
    lines.append(
        f"stage: up={info.get('up_axis')} metersPerUnit={info.get('meters_per_unit')} "
        f"kgPerUnit={info.get('kg_per_unit')} defaultPrim={info.get('default_prim')}"
    )
    if "bbox_size" in info:
        lo, hi, size = info["bbox_min"], info["bbox_max"], info["bbox_size"]
        lines.append(
            f"bbox (m): min ({lo[0]:.4f}, {lo[1]:.4f}, {lo[2]:.4f}) "
            f"max ({hi[0]:.4f}, {hi[1]:.4f}, {hi[2]:.4f}) "
            f"size {size[0]:.4f} x {size[1]:.4f} x {size[2]:.4f}"
        )
    if "euler" in info:
        lines.append(
            f"mesh {info['mesh_path']}: V={info['vertices']} E={info['edges']} F={info['faces']} "
            f"V-E+F={info['euler']} volume={info['volume_m3'] * 1e6:.1f} cm^3"
        )
        lines.append(
            f"watertight: {'yes' if info['watertight'] else 'NO'} "
            f"(bad edges {info['bad_edges']}, unused points {info['unused_points']})"
        )
    for slot, names in info.get("apis", {}).items():
        lines.append(f"APIs on {slot}: {', '.join(names) or '(none)'}")
    if "approximation" in info:
        lines.append(f"physics:approximation: {info['approximation']}")
    if "mass" in info:
        lines.append(f"physics:mass: {info['mass']:.6g} kg")
    if "material" in info:
        m = info["material"]
        lines.append(
            f"physics material {m['path']}: static {m['static_friction']} "
            f"dynamic {m['dynamic_friction']} restitution {m['restitution']}"
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
        description="Generate or check a procedural rimmed plate USD (issue #30)."
    )
    parser.add_argument("--diameter", type=float, help="plate diameter in centimetres, e.g. 22")
    parser.add_argument("--out", type=Path, help="output file, e.g. plate_22.usda")
    parser.add_argument(
        "--check", type=Path, metavar="FILE", help="inspect and validate FILE instead of writing"
    )
    args = parser.parse_args(argv)

    if args.check is not None:
        if args.diameter is not None or args.out is not None:
            parser.error("--check takes only a file; do not combine it with --diameter or --out")
        if not args.check.is_file():
            parser.error(f"no such file: {args.check}")
        info = check_plate_usd(args.check)
        print(format_report(info))
        return 1 if info["problems"] else 0

    if args.diameter is None or args.out is None:
        parser.error("--diameter and --out are required unless --check is given")
    try:
        path = write_plate_usd(args.diameter / 100.0, args.out)
    except ValueError as exc:
        parser.error(str(exc))
    info = check_plate_usd(path)
    print(
        f"wrote {path.as_posix()} ({info['bytes']} bytes): "
        f"V={info['vertices']} F={info['faces']}"
    )
    return 1 if info["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
