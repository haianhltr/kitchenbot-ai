"""Procedural plate USD (issue #30). No simulator.

The geometry and topology tests need only numpy. The tests that write or read USD skip cleanly
when ``pxr`` (the ``usd-core`` wheel) is not installed, so a CI without it stays green.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "make_plate_usd.py"
PLATE_DIR = REPO / "source" / "kitchenbot" / "scene" / "assets" / "plates"
DIAMETERS_CM = [22, 25, 27]


def _load_script():
    spec = importlib.util.spec_from_file_location("make_plate_usd", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mp = _load_script()


@pytest.mark.parametrize("diameter_cm", DIAMETERS_CM)
def test_mesh_is_a_closed_outward_sphere_like_solid(diameter_cm):
    mesh = mp.build_plate_mesh(diameter_cm / 100.0)
    topo = mp.analyze_topology(
        mesh["face_vertex_counts"], mesh["face_vertex_indices"], len(mesh["points"])
    )
    assert len(mesh["points"]) == 2 + 4 * 64  # two poles and four rings of 64
    assert topo["bad_edges"] == 0
    assert topo["unused_points"] == 0
    assert topo["euler"] == 2
    volume = mp.signed_volume(
        mesh["points"], mesh["face_vertex_counts"], mesh["face_vertex_indices"]
    )
    assert volume > 0.0  # winding is outward
    assert mesh["normals"].shape == (len(mesh["face_vertex_indices"]), 3)
    assert np.allclose(np.linalg.norm(mesh["normals"], axis=1), 1.0)


def test_open_or_misoriented_mesh_is_not_closed():
    mesh = mp.build_plate_mesh(0.22)
    counts, idx = mesh["face_vertex_counts"], mesh["face_vertex_indices"]
    n_points = len(mesh["points"])
    # Drop one triangle: a hole.
    assert not mp.analyze_topology(counts[:-1], idx[:-3], n_points)["closed"]
    # Flip one triangle: its edges are now traversed twice in the same direction.
    flipped = idx.copy()
    flipped[3:6] = flipped[[3, 5, 4]]
    assert not mp.analyze_topology(counts, flipped, n_points)["closed"]


def test_profile_dimensions():
    profile = mp.plate_profile(0.25)
    assert profile[:, 0].max() == pytest.approx(0.125)
    assert profile[:, 1].min() == 0.0
    assert profile[:, 1].max() == pytest.approx(0.020)
    with pytest.raises(ValueError):
        mp.plate_profile(0.04)


def test_generated_plate_22_matches_the_issue(tmp_path):
    pytest.importorskip("pxr", reason="usd-core not installed")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    out = mp.write_plate_usd(0.22, tmp_path / "plate_22.usda")
    stage = Usd.Stage.Open(str(out))
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0

    root = stage.GetDefaultPrim()
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    size = box.ComputeWorldBound(root).ComputeAlignedRange().GetSize()
    assert size[0] == pytest.approx(0.22, abs=1e-3)
    assert size[1] == pytest.approx(0.22, abs=1e-3)
    assert size[2] == pytest.approx(0.02, abs=1e-3)

    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert root.HasAPI(UsdPhysics.MassAPI)
    assert UsdPhysics.MassAPI(root).GetMassAttr().Get() == pytest.approx(0.35)

    mesh_prim = next(p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh))
    assert mesh_prim.HasAPI(UsdPhysics.CollisionAPI)
    assert mesh_prim.HasAPI(UsdPhysics.MeshCollisionAPI)
    approximation = UsdPhysics.MeshCollisionAPI(mesh_prim).GetApproximationAttr().Get()
    assert approximation == "convexDecomposition"
    assert UsdGeom.Mesh(mesh_prim).GetSubdivisionSchemeAttr().Get() == "none"

    material = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial("physics")[0]
    physics_material = UsdPhysics.MaterialAPI(material.GetPrim())
    assert physics_material.GetStaticFrictionAttr().Get() == 1.0
    assert physics_material.GetDynamicFrictionAttr().Get() == 1.0
    assert physics_material.GetRestitutionAttr().Get() == 0.0

    mesh = UsdGeom.Mesh(mesh_prim)
    topo = mp.analyze_topology(
        mesh.GetFaceVertexCountsAttr().Get(),
        mesh.GetFaceVertexIndicesAttr().Get(),
        len(mesh.GetPointsAttr().Get()),
    )
    assert topo["closed"]
    assert topo["euler"] == 2

    info = mp.check_plate_usd(out)
    assert info["problems"] == []
    assert info["watertight"]


def test_regenerating_over_an_existing_file_works(tmp_path):
    pytest.importorskip("pxr", reason="usd-core not installed")
    out = tmp_path / "plate.usda"
    mp.write_plate_usd(0.22, out)
    mp.write_plate_usd(0.25, out)
    assert mp.check_plate_usd(out)["bbox_size"][0] == pytest.approx(0.25, abs=1e-3)


def test_cli_generate_then_check_then_fail_on_a_hole(tmp_path):
    pytest.importorskip("pxr", reason="usd-core not installed")
    from pxr import Usd, UsdGeom, Vt

    out = tmp_path / "plate_22.usda"
    run = [sys.executable, str(SCRIPT)]
    made = subprocess.run(
        [*run, "--diameter", "22", "--out", str(out)], capture_output=True, text=True
    )
    assert made.returncode == 0, made.stdout + made.stderr
    ok = subprocess.run([*run, "--check", str(out)], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "watertight: yes" in ok.stdout
    assert "RESULT: OK" in ok.stdout

    stage = Usd.Stage.Open(str(out))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mp.MESH_PATH))
    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts[:-1]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices[:-3]))
    stage.GetRootLayer().Save()
    del stage

    bad = subprocess.run([*run, "--check", str(out)], capture_output=True, text=True)
    assert bad.returncode == 1
    assert "watertight: NO" in bad.stdout
    assert "RESULT: FAIL" in bad.stdout


@pytest.mark.parametrize("diameter_cm", DIAMETERS_CM)
def test_committed_plate_is_valid_and_current(diameter_cm, tmp_path):
    pytest.importorskip("pxr", reason="usd-core not installed")
    from pxr import Usd, UsdGeom

    committed = PLATE_DIR / f"plate_{diameter_cm}.usda"
    assert committed.is_file(), committed
    assert committed.stat().st_size < 1_000_000

    info = mp.check_plate_usd(committed)
    assert info["problems"] == []
    assert info["bbox_size"][0] == pytest.approx(diameter_cm / 100.0, abs=1e-3)
    assert info["bbox_size"][2] == pytest.approx(0.02, abs=1e-3)
    assert info["mass"] == pytest.approx(0.35)

    # The committed file must equal what the generator writes today (no hand edits, no drift).
    fresh = mp.write_plate_usd(diameter_cm / 100.0, tmp_path / committed.name)

    def mesh_arrays(path):
        stage = Usd.Stage.Open(str(path))  # keep the stage alive while reading the prim
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mp.MESH_PATH))
        return (
            np.array(mesh.GetPointsAttr().Get()),
            list(mesh.GetFaceVertexCountsAttr().Get()),
            list(mesh.GetFaceVertexIndicesAttr().Get()),
        )

    points_a, counts_a, indices_a = mesh_arrays(committed)
    points_b, counts_b, indices_b = mesh_arrays(fresh)
    assert np.allclose(points_a, points_b, atol=1e-6)
    assert counts_a == counts_b
    assert indices_a == indices_b
