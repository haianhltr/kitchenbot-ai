"""Tests for scripts/fetch_assets.py and scripts/usd_inspect.py (no network, no simulator)."""

import hashlib
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_assets = _load("fetch_assets")
usd_inspect = _load("usd_inspect")


def _manifest(tmp_path: Path, payload: bytes, sha256: str | None) -> dict:
    return {
        "schema": 1,
        "dest_root": str(tmp_path / "third_party"),
        "sources": {"web": {"name": "Web", "url": "https://example.invalid", "licence": "MIT"}},
        "files": [
            {
                "set": "demo",
                "asset": "thing",
                "source": "web",
                "kind": "url",
                "url": "https://example.invalid/thing.bin",
                "dest": "web/thing.bin",
                "sha256": sha256,
                "size": len(payload),
                "licence": "MIT",
                "purpose": "test",
            }
        ],
    }


def test_committed_manifest_is_complete():
    manifest = fetch_assets.load_manifest(fetch_assets.DEFAULT_MANIFEST)
    assert {"dishwasher", "cabinets"} <= set(fetch_assets.set_names(manifest))
    total = 0
    for f in manifest["files"]:
        assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]), f["dest"]
        assert f["licence"] and f["purpose"] and f["url"].startswith("https://")
        total += f["size"]
    assert total < 2 * 1024**3, "download cap is 2 GB"


def test_committed_manifest_md_lists_every_file():
    manifest = fetch_assets.load_manifest(fetch_assets.DEFAULT_MANIFEST)
    md = (Path(fetch_assets.REPO_ROOT) / manifest["dest_root"] / "MANIFEST.md").read_text(
        encoding="utf-8"
    )
    assert md == fetch_assets.render_manifest_md(manifest), "rerun --write-manifest-md"
    for f in manifest["files"]:
        assert f["dest"] in md


def test_validate_rejects_escaping_dest(tmp_path):
    manifest = _manifest(tmp_path, b"x", None)
    manifest["files"][0]["dest"] = "../evil.bin"
    with pytest.raises(ValueError, match="dest"):
        fetch_assets.validate_manifest(manifest)


def test_file_state(tmp_path):
    payload = b"hello"
    good = hashlib.sha256(payload).hexdigest()
    manifest = _manifest(tmp_path, payload, good)
    entry = manifest["files"][0]
    root = Path(manifest["dest_root"])
    assert fetch_assets.file_state(entry, root) == "missing"
    (root / "web").mkdir(parents=True)
    (root / entry["dest"]).write_bytes(payload)
    assert fetch_assets.file_state(entry, root) == "ok"
    entry["sha256"] = "0" * 64
    assert fetch_assets.file_state(entry, root) == "mismatch"
    entry["sha256"] = None
    assert fetch_assets.file_state(entry, root) == "unrecorded"


def test_second_run_downloads_nothing(tmp_path, capsys):
    payload = b"hello"
    manifest = _manifest(tmp_path, payload, hashlib.sha256(payload).hexdigest())
    root = Path(manifest["dest_root"])
    (root / "web").mkdir(parents=True)
    (root / "web/thing.bin").write_bytes(payload)
    path = tmp_path / "m.yaml"
    fetch_assets.dump_manifest(manifest, path)
    assert fetch_assets.main(["--set", "demo", "--manifest", str(path)]) == 0
    assert "downloaded 0 files" in capsys.readouterr().out


def test_dry_run_reports_missing_without_network(tmp_path, capsys):
    manifest = _manifest(tmp_path, b"hello", "0" * 64)
    path = tmp_path / "m.yaml"
    fetch_assets.dump_manifest(manifest, path)
    assert fetch_assets.main(["--set", "demo", "--dry-run", "--manifest", str(path)]) == 0
    out = capsys.readouterr().out
    assert "would download 1 files" in out
    assert not (Path(manifest["dest_root"]) / "web").exists()


def test_dump_manifest_round_trips(tmp_path):
    manifest = _manifest(tmp_path, b"abc", None)
    path = tmp_path / "m.yaml"
    fetch_assets.dump_manifest(manifest, path)
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == manifest


USDA = """#usda 1.0
(
    defaultPrim = "root"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "root"
{
    def Xform "body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        def Cube "box" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"])
        {
            uniform token physics:approximation = "convexHull"
        }
    }
    def Xform "door" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
    }
    def Xform "rack" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
    }
    def PhysicsRevoluteJoint "hinge"
    {
        uniform token physics:axis = "X"
        rel physics:body0 = </root/body>
        rel physics:body1 = </root/door>
        float physics:lowerLimit = 0
        float physics:upperLimit = 90
    }
    def PhysicsPrismaticJoint "slide"
    {
        uniform token physics:axis = "Y"
        rel physics:body0 = </root/body>
        rel physics:body1 = </root/rack>
        float physics:lowerLimit = -0.2
        float physics:upperLimit = 0
    }
}
"""


def test_usd_inspect_reports_joints_and_collision(tmp_path, capsys):
    pytest.importorskip("pxr")
    path = tmp_path / "demo.usda"
    path.write_text(USDA, encoding="utf-8")
    assert usd_inspect.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "Revolute/prismatic joints: 2" in out
    assert "[RevoluteJoint]" in out and "lower=0 upper=90 deg" in out
    assert "[PrismaticJoint]" in out and "lower=-0.2 upper=0 m" in out
    assert "body0=/root/body  body1=/root/door" in out
    assert "approximation=convexHull" in out


def test_usd_inspect_without_pxr_points_at_usd_core(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(usd_inspect, "HAVE_PXR", False)
    path = tmp_path / "demo.usda"
    path.write_text(USDA, encoding="utf-8")
    assert usd_inspect.main([str(path)]) == 2
    assert "pip install usd-core" in capsys.readouterr().err
