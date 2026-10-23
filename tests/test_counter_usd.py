"""Procedural counter, sink and floor USD (issue #47). No simulator.

The tests that write or read USD skip cleanly when ``pxr`` (the ``usd-core`` wheel) is not
installed, so a CI without it stays green. The pure-Python ones always run.
"""

import sys

import pytest
from usd_helpers import (
    FORBIDDEN_IMPORTS,
    REPO,
    edit_stage,
    imported_top_level_modules,
    lf_bytes,
    load_script,
    nudge,
    requires_usd,
    run_script,
)

from kitchenbot.scene import kitchen_dims as kd

ASSETS = REPO / "source" / "kitchenbot" / "scene" / "assets" / "kitchen"
COUNTER = ASSETS / "counter.usda"
FLOOR = ASSETS / "floor.usda"
SCRIPT = "make_counter_usd"

mc = load_script(SCRIPT)


# --------------------------------------------------------------------------------------------
# No pxr needed
# --------------------------------------------------------------------------------------------


def test_the_script_imports_no_simulator():
    assert imported_top_level_modules(REPO / "scripts" / f"{SCRIPT}.py").isdisjoint(
        FORBIDDEN_IMPORTS
    )


def test_help_and_the_missing_pxr_message_work_without_pxr(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "pxr", None)  # `import pxr` now raises ImportError
    module = load_script(SCRIPT)
    with pytest.raises(SystemExit) as stop:
        module.main(["--help"])
    assert stop.value.code == 0
    assert "--check" in capsys.readouterr().out
    assert module.main(["--out-dir", str(tmp_path)]) == 2
    assert "usd-core" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_boxes_tile_the_counter_around_the_sink():
    boxes = mc.counter_boxes()
    assert len(boxes) == 9
    sides = ("Left", "Right", "Front", "Back")
    assert {path for path, *_ in boxes} == {
        *(f"/Counter/{layer}/{side}" for layer in ("Top", "Body") for side in sides),
        "/Counter/Body/SinkFloor",
    }
    top_faces = {hi[2] for path, _, hi, _ in boxes if "/Top/" in path}
    assert top_faces == {kd.COUNTER_HEIGHT_M}
    sink_floor = next(hi[2] for path, _, hi, _ in boxes if path.endswith("SinkFloor"))
    assert sink_floor == pytest.approx(kd.COUNTER_HEIGHT_M - kd.SINK_DEPTH_M)
    # Volumes add up: the whole footprint to the counter-top face, minus the open basin.
    volume = sum((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]) for _, lo, hi, _ in boxes)
    whole = kd.COUNTER_WIDTH_M * kd.COUNTER_DEPTH_M * kd.COUNTER_HEIGHT_M
    basin = kd.SINK_SIZE_M[0] * kd.SINK_SIZE_M[1] * kd.SINK_DEPTH_M
    assert volume == pytest.approx(whole - basin)


def test_floor_top_is_zero():
    lo, hi = mc.floor_box()
    assert hi[2] == 0.0 and lo[2] == -kd.FLOOR_THICKNESS_M
    assert (hi[0] - lo[0], hi[1] - lo[1]) == pytest.approx(kd.FLOOR_SIZE_M)


# --------------------------------------------------------------------------------------------
# Generation (needs usd-core)
# --------------------------------------------------------------------------------------------


@requires_usd
def test_generated_counter_matches_the_issue(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    out = mc.write_counter_usd(tmp_path / "counter.usda")
    stage = Usd.Stage.Open(str(out))
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    root = stage.GetDefaultPrim()
    prims = list(Usd.PrimRange(root))
    assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims)  # static
    assert not any(p.IsA(UsdGeom.Mesh) for p in prims)
    colliders = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 9 and all(p.IsA(UsdGeom.Cube) for p in colliders)

    materials = [p for p in prims if p.IsA(UsdShade.Material)]
    assert len(materials) == 1
    assert UsdPhysics.MaterialAPI(materials[0]).GetStaticFrictionAttr().Get() == kd.COUNTER_FRICTION
    for prim in colliders:
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]
        assert bound.GetPath() == materials[0].GetPath()

    # Counter-top face, from the stage's own bounds: z = COUNTER_HEIGHT_M within 1 mm.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    top = stage.GetPrimAtPath("/Counter/Top")
    top_box = cache.ComputeWorldBound(top).ComputeAlignedRange()
    assert top_box.GetMax()[2] == pytest.approx(kd.COUNTER_HEIGHT_M, abs=1e-3)
    assert top_box.GetMin()[2] == pytest.approx(
        kd.COUNTER_HEIGHT_M - kd.COUNTER_TOP_THICKNESS_M, abs=1e-3
    )
    whole = cache.ComputeWorldBound(root).ComputeAlignedRange()
    assert tuple(whole.GetMin()) == pytest.approx((-kd.COUNTER_WIDTH_M / 2, 0.0, 0.0), abs=1e-3)
    assert tuple(whole.GetMax()) == pytest.approx(
        (kd.COUNTER_WIDTH_M / 2, kd.COUNTER_DEPTH_M, kd.COUNTER_HEIGHT_M), abs=1e-3
    )
    assert mc.check_counter_usd(out)["problems"] == []


@requires_usd
def test_generated_floor_matches_the_issue(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    out = mc.write_floor_usd(tmp_path / "floor.usda")
    stage = Usd.Stage.Open(str(out))
    root = stage.GetDefaultPrim()
    prims = list(Usd.PrimRange(root))
    assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims)
    slab = stage.GetPrimAtPath("/Floor/Slab")
    assert slab.HasAPI(UsdPhysics.CollisionAPI) and slab.IsA(UsdGeom.Cube)
    material = UsdShade.MaterialBindingAPI(slab).ComputeBoundMaterial("physics")[0]
    assert UsdPhysics.MaterialAPI(material.GetPrim()).GetDynamicFrictionAttr().Get() == 1.0

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(root).ComputeAlignedRange()
    assert box.GetMax()[2] == pytest.approx(0.0, abs=1e-3)  # the floor top is z = 0
    assert box.GetSize()[0] == pytest.approx(kd.FLOOR_SIZE_M[0], abs=1e-3)
    assert box.GetSize()[2] == pytest.approx(kd.FLOOR_THICKNESS_M, abs=1e-3)
    assert mc.check_floor_usd(out)["problems"] == []


@requires_usd
def test_same_input_gives_byte_identical_output(tmp_path):
    for write, name in ((mc.write_counter_usd, "counter.usda"), (mc.write_floor_usd, "floor.usda")):
        first, second = write(tmp_path / "a" / name), write(tmp_path / "b" / name)
        assert first.read_bytes() == second.read_bytes()


@requires_usd
@pytest.mark.parametrize(
    ("committed", "write"),
    [(COUNTER, "write_counter_usd"), (FLOOR, "write_floor_usd")],
    ids=["counter", "floor"],
)
def test_committed_file_is_small_valid_and_equals_a_fresh_generation(committed, write, tmp_path):
    assert committed.is_file(), committed
    assert committed.stat().st_size < 100_000
    info = mc.check_usd(committed)
    assert info["problems"] == []
    fresh = getattr(mc, write)(tmp_path / committed.name)
    assert lf_bytes(committed) == lf_bytes(fresh), (
        f"{committed.name} is stale or hand edited: regenerate with scripts/make_counter_usd.py"
    )


@requires_usd
def test_committed_numbers():
    counter, floor = mc.check_usd(COUNTER), mc.check_usd(FLOOR)
    assert counter["top_face_z"] == pytest.approx(0.9, abs=1e-3)  # COUNTER_HEIGHT_M, from #10
    assert counter["sink_floor_z"] == pytest.approx(kd.COUNTER_HEIGHT_M - kd.SINK_DEPTH_M, abs=1e-3)
    assert floor["top_face_z"] == pytest.approx(0.0, abs=1e-3)
    assert (counter["colliders"], floor["colliders"]) == (9, 1)


# --------------------------------------------------------------------------------------------
# The command line and --check on tampered copies (needs usd-core)
# --------------------------------------------------------------------------------------------


@requires_usd
def test_cli_writes_the_committed_files_and_checks_them(tmp_path):
    made = run_script(SCRIPT, "--out-dir", tmp_path)
    assert made.returncode == 0, made.stdout + made.stderr
    assert lf_bytes(tmp_path / "counter.usda") == lf_bytes(COUNTER)
    assert lf_bytes(tmp_path / "floor.usda") == lf_bytes(FLOOR)
    ok = run_script(SCRIPT, "--check", COUNTER, FLOOR)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert ok.stdout.count("RESULT: OK") == 2
    assert "top face z = 0.9000 m" in ok.stdout and "top face z = 0.0000 m" in ok.stdout


def _copy(tmp_path, source, name):
    copy = tmp_path / name
    copy.write_bytes(source.read_bytes())
    return copy


@requires_usd
def test_cli_check_accepts_a_sub_millimetre_nudge(tmp_path):
    copy = _copy(tmp_path, COUNTER, "counter.usda")
    with edit_stage(copy) as stage:
        nudge(stage, "/Counter/Top/Left", dz=0.0004)
    assert run_script(SCRIPT, "--check", copy).returncode == 0


def _raise_top(stage):
    for side in ("Left", "Right", "Front", "Back"):
        nudge(stage, f"/Counter/Top/{side}", dz=0.005)


def _lower_one_top_piece(stage):
    nudge(stage, "/Counter/Top/Back", dz=-0.002)


def _shift_the_counter(stage):
    for layer in ("Top", "Body"):
        for side in ("Left", "Right", "Front", "Back"):
            nudge(stage, f"/Counter/{layer}/{side}", dx=0.01)


def _raise_the_sink_floor(stage):
    nudge(stage, "/Counter/Body/SinkFloor", dz=0.05)


def _fill_the_basin(stage):
    from pxr import UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, "/Counter/Body/Plug")
    cube.CreateSizeAttr(0.2)
    UsdGeom.Xformable(cube).AddTranslateOp().Set((-0.55, 0.3, 0.85))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _drop_a_collider(stage):
    from pxr import UsdPhysics

    stage.GetPrimAtPath("/Counter/Body/Left").RemoveAPI(UsdPhysics.CollisionAPI)


def _make_rigid(stage):
    from pxr import UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/Counter/Body/Left"))


def _slippery(stage):
    stage.GetPrimAtPath("/Counter/PhysicsMaterial").GetAttribute("physics:dynamicFriction").Set(0.1)


def _unbind(stage):
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/Counter/Top/Front")).UnbindAllBindings()


COUNTER_TAMPERS = [
    _raise_top,
    _lower_one_top_piece,
    _shift_the_counter,
    _raise_the_sink_floor,
    _fill_the_basin,
    _drop_a_collider,
    _make_rigid,
    _slippery,
    _unbind,
]


@requires_usd
@pytest.mark.parametrize("tamper", COUNTER_TAMPERS, ids=lambda f: f.__name__.strip("_"))
def test_cli_check_fails_on_a_tampered_counter(tamper, tmp_path):
    copy = _copy(tmp_path, COUNTER, "counter.usda")
    with edit_stage(copy) as stage:
        tamper(stage)
    bad = run_script(SCRIPT, "--check", copy)
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "PROBLEM:" in bad.stdout and "RESULT: FAIL" in bad.stdout


def _sink_floor(stage):
    nudge(stage, "/Floor/Slab", dz=-0.005)  # top face now at z = -5 mm


def _raise_floor(stage):
    nudge(stage, "/Floor/Slab", dz=0.005)


def _shrink_floor(stage):
    stage.GetPrimAtPath("/Floor/Slab").GetAttribute("xformOp:scale").Set((3.0, 6.0, 0.1))


def _floor_rigid(stage):
    from pxr import UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/Floor/Slab"))


FLOOR_TAMPERS = [_sink_floor, _raise_floor, _shrink_floor, _floor_rigid]


@requires_usd
@pytest.mark.parametrize("tamper", FLOOR_TAMPERS, ids=lambda f: f.__name__.strip("_"))
def test_cli_check_fails_on_a_tampered_floor(tamper, tmp_path):
    copy = _copy(tmp_path, FLOOR, "floor.usda")
    with edit_stage(copy) as stage:
        tamper(stage)
    bad = run_script(SCRIPT, "--check", copy)
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "PROBLEM:" in bad.stdout and "RESULT: FAIL" in bad.stdout


@requires_usd
def test_cli_check_reports_every_file_and_fails_if_any_fails(tmp_path):
    bad = _copy(tmp_path, FLOOR, "floor.usda")
    with edit_stage(bad) as stage:
        _raise_floor(stage)
    result = run_script(SCRIPT, "--check", COUNTER, bad)
    assert result.returncode == 1
    assert result.stdout.count("RESULT: OK") == 1 and result.stdout.count("RESULT: FAIL") == 1


@requires_usd
def test_cli_check_rejects_a_file_that_is_neither_counter_nor_floor():
    plate = REPO / "source/kitchenbot/scene/assets/plates/plate_22.usda"
    result = run_script(SCRIPT, "--check", plate)
    assert result.returncode == 1
    assert "default prim is /Plate" in result.stdout


@requires_usd
def test_cli_check_of_a_missing_file_is_a_usage_error(tmp_path):
    assert run_script(SCRIPT, "--check", tmp_path / "nope.usda").returncode == 2
