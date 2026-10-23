"""Procedural rack tines USD (issue #47). No simulator.

The tests that write or read USD skip cleanly when ``pxr`` (the ``usd-core`` wheel) is not
installed, so a CI without it stays green. The pure-Python ones (imports, ``--help``, the message
without ``pxr``) always run.
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

from kitchenbot.scene.kitchen_dims import RackSlotLayout

COMMITTED = REPO / "source" / "kitchenbot" / "scene" / "assets" / "dishwasher" / "rack_tines.usda"
SCRIPT = "make_rack_tines_usd"

mr = load_script(SCRIPT)


# --------------------------------------------------------------------------------------------
# No pxr needed
# --------------------------------------------------------------------------------------------


def test_the_script_and_its_helpers_import_no_simulator():
    for path in (REPO / "scripts" / f"{SCRIPT}.py", REPO / "source/kitchenbot/scene/usd_prims.py"):
        assert imported_top_level_modules(path).isdisjoint(FORBIDDEN_IMPORTS)


def test_help_and_the_missing_pxr_message_work_without_pxr(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "pxr", None)  # `import pxr` now raises ImportError
    module = load_script(SCRIPT)  # a fresh copy: nothing may import pxr at import time
    with pytest.raises(SystemExit) as stop:
        module.main(["--help"])
    assert stop.value.code == 0
    assert "--check" in capsys.readouterr().out

    out = tmp_path / "rack_tines.usda"
    assert module.main(["--out", str(out)]) == 2
    assert "usd-core" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    "flags",
    [["--slots", "0"], ["--scale", "0"], ["--pitch", "0.09"], ["--row-spacing", "0.3"]],
)
def test_impossible_layout_flags_are_a_usage_error(flags, tmp_path):
    result = run_script(SCRIPT, *flags, "--out", tmp_path / "x.usda")
    assert result.returncode == 2
    assert not (tmp_path / "x.usda").exists()


# --------------------------------------------------------------------------------------------
# Generation (needs usd-core)
# --------------------------------------------------------------------------------------------


@requires_usd
def test_generated_file_matches_the_issue(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    layout = RackSlotLayout()
    out = mr.write_rack_tines_usd(layout, tmp_path / "rack_tines.usda")
    stage = Usd.Stage.Open(str(out))
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    root = stage.GetDefaultPrim()
    assert str(root.GetPath()) == "/RackTines"

    prims = list(Usd.PrimRange(root))
    assert not any(p.IsA(UsdGeom.Mesh) for p in prims)  # primitives only
    assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims)  # the rack link is the body
    colliders = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 1 + 2 * 6  # base frame plus six tines in each of two rows
    assert sum(p.IsA(UsdGeom.Capsule) for p in colliders) == 12
    assert sum(p.IsA(UsdGeom.Cube) for p in colliders) == 1

    materials = [p for p in prims if p.IsA(UsdShade.Material)]
    assert len(materials) == 1
    api = UsdPhysics.MaterialAPI(materials[0])
    assert api.GetStaticFrictionAttr().Get() == 1.0
    assert api.GetDynamicFrictionAttr().Get() == 1.0
    for prim in colliders:
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]
        assert bound.GetPath() == materials[0].GetPath()

    # The gap centres, read from the tines themselves, are the slot poses to within 1 mm.
    cache = UsdGeom.XformCache()
    tines = [p for p in prims if p.GetName().startswith("Tine_")]
    assert len(tines) == 12
    centres = [tuple(cache.GetLocalToWorldTransform(p).ExtractTranslation()) for p in tines]
    for i in range(1, 6):
        x, y, z, _ = layout.slot_pose(i)
        neighbours = [c for c in centres if abs(c[0] - x) < layout.pitch / 2 + 1e-6]
        assert len(neighbours) == 4  # the tine on each side of the gap, in both rows
        assert sum(c[0] for c in neighbours) / 4 == pytest.approx(x, abs=1e-3)
        assert sum(c[1] for c in neighbours) / 4 == pytest.approx(y, abs=1e-3)
        top = max(c[2] for c in neighbours) + layout.tine_height / 2
        assert top == pytest.approx(z, abs=1e-3)  # tine top is the slot entry
        marker = stage.GetPrimAtPath(f"/RackTines/Slots/Slot_{i}")
        pos = cache.GetLocalToWorldTransform(marker).ExtractTranslation()
        assert tuple(pos) == pytest.approx((x, y, z), abs=1e-3)

    # The two rows are symmetric and inside the base frame.
    ys = sorted({round(c[1], 6) for c in centres})
    assert ys == pytest.approx([-0.09, 0.09])
    xs = [c[0] for c in centres]
    assert min(xs) - layout.tine_radius > -layout.width / 2
    assert max(xs) + layout.tine_radius < layout.width / 2

    info = mr.check_rack_tines_usd(out, layout)
    assert info["problems"] == []
    assert info["max_gap_error_m"] < 1e-6


@requires_usd
def test_same_input_gives_byte_identical_output(tmp_path):
    first = mr.write_rack_tines_usd(RackSlotLayout(), tmp_path / "a" / "rack_tines.usda")
    second = mr.write_rack_tines_usd(RackSlotLayout(), tmp_path / "b" / "rack_tines.usda")
    assert first.read_bytes() == second.read_bytes()
    other = mr.write_rack_tines_usd(RackSlotLayout(scale=1.4), tmp_path / "c.usda")
    assert other.read_bytes() != first.read_bytes()


@requires_usd
def test_regenerating_over_an_existing_file_works(tmp_path):
    out = tmp_path / "rack_tines.usda"
    mr.write_rack_tines_usd(RackSlotLayout(slots=3), out)
    mr.write_rack_tines_usd(RackSlotLayout(), out)
    assert mr.check_rack_tines_usd(out, RackSlotLayout())["problems"] == []


@requires_usd
def test_committed_file_is_small_valid_and_equals_a_fresh_generation(tmp_path):
    assert COMMITTED.is_file(), COMMITTED
    assert COMMITTED.stat().st_size < 100_000
    layout = RackSlotLayout()
    info = mr.check_rack_tines_usd(COMMITTED, layout)
    assert info["problems"] == []
    assert info["colliders"] == 13
    fresh = mr.write_rack_tines_usd(layout, tmp_path / COMMITTED.name)
    assert lf_bytes(COMMITTED) == lf_bytes(fresh), (
        "rack_tines.usda is stale or hand edited: regenerate with scripts/make_rack_tines_usd.py"
    )


@requires_usd
@pytest.mark.parametrize("scale", [1.3, 1.4, 1.5])
def test_scaled_file_passes_only_against_its_own_scale(scale, tmp_path):
    layout = RackSlotLayout(scale=scale)
    out = mr.write_rack_tines_usd(layout, tmp_path / "scaled.usda")
    assert mr.check_rack_tines_usd(out, layout)["problems"] == []
    assert mr.check_rack_tines_usd(out, RackSlotLayout())["problems"] != []
    info = mr.check_rack_tines_usd(out, layout)
    assert info["gap_centres"][4][0] == pytest.approx(scale * 0.10, abs=1e-6)


# --------------------------------------------------------------------------------------------
# The command line and --check on tampered copies (needs usd-core)
# --------------------------------------------------------------------------------------------


@requires_usd
def test_cli_writes_then_checks_the_committed_defaults(tmp_path):
    out = tmp_path / "rack_tines.usda"
    made = run_script(SCRIPT, "--out", out)
    assert made.returncode == 0, made.stdout + made.stderr
    assert lf_bytes(out) == lf_bytes(COMMITTED)
    ok = run_script(SCRIPT, "--check", COMMITTED)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "RESULT: OK" in ok.stdout
    assert "colliders: 13 (expected 13)" in ok.stdout
    assert "max deviation from slot_pose: 0.000 mm" in ok.stdout


@requires_usd
def test_cli_check_with_other_layout_flags_fails_on_the_committed_file():
    other_layouts = (
        ["--pitch", "0.06"],
        ["--scale", "1.4"],
        ["--slots", "4"],
        ["--row-spacing", "0.2"],
    )
    for flags in other_layouts:
        bad = run_script(SCRIPT, "--check", COMMITTED, *flags)
        assert bad.returncode == 1, flags
        assert "RESULT: FAIL" in bad.stdout


@requires_usd
def test_cli_check_accepts_a_sub_millimetre_nudge(tmp_path):
    copy = tmp_path / "copy.usda"
    copy.write_bytes(COMMITTED.read_bytes())
    with edit_stage(copy) as stage:
        nudge(stage, "/RackTines/Tines/Tine_pos_4", dx=0.0004)
    ok = run_script(SCRIPT, "--check", copy)
    assert ok.returncode == 0, ok.stdout


def _move_tine(stage):
    nudge(stage, "/RackTines/Tines/Tine_neg_3", dx=0.005)


def _move_marker(stage):
    nudge(stage, "/RackTines/Slots/Slot_3", dz=0.005)


def _remove_tine(stage):
    stage.RemovePrim("/RackTines/Tines/Tine_pos_6")


def _taller_tine(stage):
    stage.GetPrimAtPath("/RackTines/Tines/Tine_pos_2").GetAttribute("height").Set(0.084)


def _thicker_base(stage):
    base = stage.GetPrimAtPath("/RackTines/BaseFrame")
    base.GetAttribute("xformOp:scale").Set((0.366, 0.287, 0.02))


def _drop_collision(stage):
    from pxr import UsdPhysics

    stage.GetPrimAtPath("/RackTines/Tines/Tine_neg_1").RemoveAPI(UsdPhysics.CollisionAPI)


def _make_rigid(stage):
    from pxr import UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(stage.GetDefaultPrim())


def _soften_material(stage):
    stage.GetPrimAtPath("/RackTines/PhysicsMaterial").GetAttribute("physics:staticFriction").Set(0.5)


def _unbind_material(stage):
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/RackTines/Tines/Tine_pos_1")).UnbindAllBindings()


def _swap_in_a_mesh(stage):
    from pxr import UsdGeom

    stage.RemovePrim("/RackTines/Tines/Tine_neg_2")
    UsdGeom.Mesh.Define(stage, "/RackTines/Tines/Tine_neg_2")


def _wrong_up_axis(stage):
    from pxr import UsdGeom

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)


TAMPERS = [
    _move_tine,
    _move_marker,
    _remove_tine,
    _taller_tine,
    _thicker_base,
    _drop_collision,
    _make_rigid,
    _soften_material,
    _unbind_material,
    _swap_in_a_mesh,
    _wrong_up_axis,
]


@requires_usd
@pytest.mark.parametrize("tamper", TAMPERS, ids=lambda f: f.__name__.strip("_"))
def test_cli_check_fails_on_a_tampered_copy(tamper, tmp_path):
    copy = tmp_path / "copy.usda"
    copy.write_bytes(COMMITTED.read_bytes())
    with edit_stage(copy) as stage:
        tamper(stage)
    bad = run_script(SCRIPT, "--check", copy)
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "PROBLEM:" in bad.stdout
    assert "RESULT: FAIL" in bad.stdout


@requires_usd
def test_cli_check_of_a_missing_file_is_a_usage_error(tmp_path):
    result = run_script(SCRIPT, "--check", tmp_path / "nope.usda")
    assert result.returncode == 2
