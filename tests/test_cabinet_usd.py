"""Dishwasher cabinet USD (issue #70): the plinth dressed as a low cabinet. No simulator.

The tests that write or read USD skip cleanly when ``pxr`` (the ``usd-core`` wheel) is not
installed, so a CI without it stays green. The pure-Python ones always run.
"""

import math
import sys

import numpy as np
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

from kitchenbot.scene import dishwasher_frames as df
from kitchenbot.scene import kitchen_dims as kd

ASSETS = REPO / "source" / "kitchenbot" / "scene" / "assets" / "kitchen"
CABINET = ASSETS / "dishwasher_cabinet.usda"
SCRIPT = "make_cabinet_usd"

mcab = load_script(SCRIPT)

ROOT = "/DishwasherCabinet"
PATHS = {
    "/DishwasherCabinet/Carcass/Body",
    "/DishwasherCabinet/Carcass/Kick",
    "/DishwasherCabinet/Door/Panel",
    "/DishwasherCabinet/Handle/Bar",
    "/DishwasherCabinet/Handle/PostA",
    "/DishwasherCabinet/Handle/PostB",
}


def _boxes():
    return {p: (np.array(lo), np.array(hi)) for p, lo, hi, _ in mcab.cabinet_boxes()}


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


def test_six_boxes_with_the_documented_names():
    boxes = mcab.cabinet_boxes()
    assert {path for path, *_ in boxes} == PATHS
    assert len(boxes) == 6


def test_carcass_is_the_plinth_footprint_up_to_the_gap_under_the_body():
    (cx, cy, _), (sx, sy, sz) = df.plinth_box_kitchen()
    boxes = _boxes()
    body, kick = boxes["/DishwasherCabinet/Carcass/Body"], boxes["/DishwasherCabinet/Carcass/Kick"]
    lo, hi = np.minimum(body[0], kick[0]), np.maximum(body[1], kick[1])
    assert lo == pytest.approx((cx - sx / 2, cy - sy / 2, 0.0))
    assert hi == pytest.approx((cx + sx / 2, cy + sy / 2, sz - kd.PLINTH_GAP_M))
    assert kd.PLINTH_GAP_M == 0.002
    # Under the dishwasher body, with the 2 mm between them.
    body_underside = df.footprints_kitchen()["body"][1][0]
    assert body_underside - hi[2] == pytest.approx(kd.PLINTH_GAP_M)
    # The body of the carcass stands on the kick plate, which is set back on the -x face only.
    assert body[0][2] == pytest.approx(kd.CABINET_KICK_HEIGHT_M)
    assert kick[1][2] == pytest.approx(kd.CABINET_KICK_HEIGHT_M)
    assert kick[0][0] - body[0][0] == pytest.approx(kd.CABINET_KICK_RECESS_M)
    assert (kick[0][1], kick[1][0], kick[1][1]) == pytest.approx(
        (body[0][1], body[1][0], body[1][1])
    )


def test_door_panel_is_on_the_face_the_dishwasher_looks_at():
    boxes = _boxes()
    body = boxes["/DishwasherCabinet/Carcass/Body"]
    lo, hi = boxes["/DishwasherCabinet/Door/Panel"]
    # The dishwasher's front is its root-link -y face; yaw -90 deg turns that to kitchen -x.
    assert mcab._front_normal(kd.DISHWASHER_POSE_KITCHEN[3]) == pytest.approx((-1.0, 0.0))
    assert hi[0] == pytest.approx(body[0][0])  # touches the carcass face
    assert hi[0] - lo[0] == pytest.approx(kd.CABINET_DOOR_THICKNESS_M) == pytest.approx(0.018)
    gap = kd.CABINET_DOOR_GAP_M
    assert lo[1] - body[0][1] == pytest.approx(gap) and body[1][1] - hi[1] == pytest.approx(gap)
    assert lo[2] - body[0][2] == pytest.approx(gap)  # above the kick plate
    assert body[1][2] - hi[2] == pytest.approx(kd.CABINET_DOOR_TOP_REVEAL_M)


def test_handle_is_a_bar_on_two_posts_centred_on_the_door():
    boxes = _boxes()
    door_lo, door_hi = boxes["/DishwasherCabinet/Door/Panel"]
    bar_lo, bar_hi = boxes["/DishwasherCabinet/Handle/Bar"]
    face = door_lo[0]
    assert face - bar_lo[0] == pytest.approx(kd.CABINET_HANDLE_PROJECTION_M)
    assert bar_hi[1] - bar_lo[1] == pytest.approx(kd.CABINET_HANDLE_LENGTH_M)
    assert bar_hi[0] - bar_lo[0] == pytest.approx(kd.CABINET_HANDLE_SECTION_M)
    assert bar_hi[2] - bar_lo[2] == pytest.approx(kd.CABINET_HANDLE_SECTION_M)
    assert (bar_lo[1] + bar_hi[1]) / 2 == pytest.approx((door_lo[1] + door_hi[1]) / 2)
    assert door_hi[2] - bar_hi[2] == pytest.approx(kd.CABINET_HANDLE_DROP_M)
    for post in ("PostA", "PostB"):
        lo, hi = boxes[f"/DishwasherCabinet/Handle/{post}"]
        assert hi[0] == pytest.approx(face)  # stands on the door face
        assert lo[0] == pytest.approx(bar_hi[0])  # and carries the bar
        assert (lo[2], hi[2]) == pytest.approx((bar_lo[2], bar_hi[2]))
        assert hi[1] - lo[1] == pytest.approx(kd.CABINET_HANDLE_SECTION_M)
        assert bar_lo[1] - 1e-9 <= lo[1] and hi[1] <= bar_hi[1] + 1e-9


def test_colours_are_the_counters_body_colour_and_steel():
    counter = load_script("make_counter_usd")
    assert mcab.BODY_COLOR == counter.BODY_COLOR
    assert mcab.HANDLE_COLOR == counter.SINK_COLOR
    colours = {path: colour for path, _, _, colour in mcab.cabinet_boxes()}
    for path in PATHS:
        want = mcab.HANDLE_COLOR if "/Handle/" in path else counter.BODY_COLOR
        assert colours[path] == want, path


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, math.pi, 0.3])
def test_only_the_minus_90_degree_yaw_is_built(yaw):
    pose = (*kd.DISHWASHER_POSE_KITCHEN[:3], yaw)
    with pytest.raises(ValueError):
        mcab.cabinet_boxes(pose=pose)


def test_the_unit_has_to_be_raised():
    with pytest.raises(ValueError):
        mcab.cabinet_boxes(pose=(0.59, -0.015, 0.0, -math.pi / 2))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CABINET_KICK_HEIGHT_M", 0.6),
        ("CABINET_KICK_RECESS_M", 0.5),
        ("CABINET_DOOR_TOP_REVEAL_M", 0.5),
        ("CABINET_DOOR_THICKNESS_M", 0.0),
        ("CABINET_HANDLE_LENGTH_M", 0.5),
        ("CABINET_HANDLE_PROJECTION_M", 0.01),
        ("CABINET_HANDLE_DROP_M", 0.5),
    ],
)
def test_inconsistent_constants_are_refused(monkeypatch, name, value):
    monkeypatch.setattr(kd, name, value)
    with pytest.raises(ValueError):
        mcab.cabinet_boxes()


def test_separation_is_the_largest_axis_gap():
    unit = (np.zeros(3), np.ones(3))
    assert mcab.separation(unit, (np.array([3.0, 0.0, 0.0]), np.array([4.0, 1.0, 1.0]))) == 2.0
    assert mcab.separation(unit, (np.array([0.5, 0.5, 1.25]), np.array([2.0, 2.0, 2.0]))) == 0.25
    assert mcab.separation(unit, (np.full(3, 0.5), np.full(3, 2.0))) == pytest.approx(-0.5)


def test_the_open_door_hangs_in_front_of_the_panel_so_only_z_separates_them():
    lo, hi = mcab.open_door_box()
    panel_lo, panel_hi = _boxes()["/DishwasherCabinet/Door/Panel"]
    assert (lo[2], hi[2]) == pytest.approx(df.footprints_kitchen()["door"][1])
    assert lo[2] == pytest.approx(0.5445, abs=1e-4)  # the underside the issue quotes: 0.54 to 0.57
    for axis in (0, 1):  # the plan views overlap, so the vertical gap is the clearance
        assert min(hi[axis], panel_hi[axis]) > max(lo[axis], panel_lo[axis])


def test_clearances_are_stated_in_numbers():
    c = mcab.clearances(_boxes())
    underside = df.footprints_kitchen()["door"][1][0]
    panel_top = df.plinth_box_kitchen()[1][2] - kd.PLINTH_GAP_M - kd.CABINET_DOOR_TOP_REVEAL_M
    assert c["door_panel_to_open_door_m"] == pytest.approx(underside - panel_top)
    assert c["handle_to_open_door_m"] == pytest.approx(
        underside - (panel_top - kd.CABINET_HANDLE_DROP_M)
    )
    assert c["door_panel_to_right_leg_m"] == pytest.approx(0.59 - 0.018 - 0.17)
    assert c["handle_to_right_leg_m"] == pytest.approx(0.59 - 0.018 - 0.032 - 0.17)
    # The numbers quoted in the REPORT (millimetres).
    assert round(c["door_panel_to_open_door_m"] * 1000, 1) == 16.5
    assert round(c["handle_to_open_door_m"] * 1000, 1) == 76.5
    assert round(c["door_panel_to_right_leg_m"] * 1000, 1) == 402.0
    assert round(c["handle_to_right_leg_m"] * 1000, 1) == 370.0
    assert mcab.clearance_problems(c) == []


def test_clearance_minima_are_the_issues_and_the_frames_tests():
    import test_dishwasher_frames as frames_tests

    assert kd.CABINET_MIN_DOOR_CLEARANCE_M == 0.01
    assert kd.CABINET_MIN_LEG_CLEARANCE_M == 0.03  # `x0 - LEG_SURFACE_M >= 0.03` there
    assert kd.ROBOT_RIGHT_LEG_SURFACE_X_M == frames_tests.LEG_SURFACE_M


def test_a_door_as_tall_as_the_carcass_would_meet_the_open_dishwasher_door(monkeypatch):
    """Why there is a top reveal: with a 3 mm one the panel reaches into the open door."""
    monkeypatch.setattr(kd, "CABINET_DOOR_TOP_REVEAL_M", kd.CABINET_DOOR_GAP_M)
    c = mcab.clearances(_boxes())
    assert c["door_panel_to_open_door_m"] < 0.0
    problems = mcab.clearance_problems(c)
    assert len(problems) == 1 and "door_panel_to_open_door" in problems[0]


def test_a_cabinet_too_close_to_the_leg_is_reported(monkeypatch):
    monkeypatch.setattr(kd, "ROBOT_RIGHT_LEG_SURFACE_X_M", 0.55)
    problems = mcab.clearance_problems(mcab.clearances(_boxes()))
    assert [p.split(" is ")[0] for p in problems] == [
        "door_panel_to_right_leg",
        "handle_to_right_leg",
    ]


# --------------------------------------------------------------------------------------------
# Generation (needs usd-core)
# --------------------------------------------------------------------------------------------


@requires_usd
def test_generated_cabinet_matches_the_issue(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    out = mcab.write_cabinet_usd(tmp_path / "dishwasher_cabinet.usda")
    stage = Usd.Stage.Open(str(out))
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    root = stage.GetDefaultPrim()
    assert str(root.GetPath()) == ROOT
    prims = list(Usd.PrimRange(root))
    assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims)  # static
    assert not any(p.IsA(UsdGeom.Mesh) for p in prims)
    colliders = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert {str(p.GetPath()) for p in colliders} == PATHS
    assert all(p.IsA(UsdGeom.Cube) for p in colliders)

    materials = [p for p in prims if p.IsA(UsdShade.Material)]
    assert len(materials) == 1
    api = UsdPhysics.MaterialAPI(materials[0])
    assert api.GetStaticFrictionAttr().Get() == kd.CABINET_FRICTION
    assert api.GetDynamicFrictionAttr().Get() == kd.CABINET_FRICTION
    assert api.GetRestitutionAttr().Get() == kd.SURFACE_RESTITUTION
    for prim in colliders:
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]
        assert bound.GetPath() == materials[0].GetPath()
        colour = tuple(UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()[0])
        want = mcab.HANDLE_COLOR if "/Handle/" in str(prim.GetPath()) else mcab.BODY_COLOR
        assert colour == pytest.approx(want, abs=1e-6)
    assert mcab.check_cabinet_usd(out)["problems"] == []


@requires_usd
def test_stage_bounds_and_clearances_from_usd_itself(tmp_path):
    """Footprint and clearances from ``UsdGeom.BBoxCache``, not from the script's own reader."""
    from pxr import Usd, UsdGeom

    out = mcab.write_cabinet_usd(tmp_path / "dishwasher_cabinet.usda")
    stage = Usd.Stage.Open(str(out))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    def bounds(path):
        box = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        return np.array(box.GetMin()), np.array(box.GetMax())

    (cx, cy, _), (sx, sy, sz) = df.plinth_box_kitchen()
    carcass_lo, carcass_hi = bounds(f"{ROOT}/Carcass")
    assert carcass_lo == pytest.approx((cx - sx / 2, cy - sy / 2, 0.0), abs=1e-3)
    assert carcass_hi == pytest.approx(
        (cx + sx / 2, cy + sy / 2, sz - kd.PLINTH_GAP_M), abs=1e-3
    )
    assert carcass_hi[2] == pytest.approx(0.548, abs=1e-3)
    assert df.footprints_kitchen()["body"][1][0] - carcass_hi[2] == pytest.approx(0.002, abs=1e-3)

    # Door panel and handle against the open dishwasher door's underside (they overlap in plan).
    corners, (door_z0, _) = df.footprints_kitchen()["door"]
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    for scope in ("Door", "Handle"):
        lo, hi = bounds(f"{ROOT}/{scope}")
        assert min(max(xs), hi[0]) > max(min(xs), lo[0])
        assert min(max(ys), hi[1]) > max(min(ys), lo[1])
        assert door_z0 - hi[2] >= 0.01, f"{scope} top {hi[2]:.4f}, door underside {door_z0:.4f}"
        assert lo[0] - 0.17 >= 0.03, f"{scope} is {lo[0] - 0.17:.3f} m from the right leg"
    assert door_z0 - bounds(f"{ROOT}/Door")[1][2] == pytest.approx(0.0165, abs=1e-3)
    assert door_z0 - bounds(f"{ROOT}/Handle")[1][2] == pytest.approx(0.0765, abs=1e-3)


@requires_usd
def test_an_unsafe_cabinet_is_refused_before_anything_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(kd, "CABINET_DOOR_TOP_REVEAL_M", kd.CABINET_DOOR_GAP_M)
    out = tmp_path / "dishwasher_cabinet.usda"
    with pytest.raises(ValueError, match="clearance too small"):
        mcab.write_cabinet_usd(out)
    assert not out.exists()


@requires_usd
def test_same_input_gives_byte_identical_output(tmp_path):
    first = mcab.write_cabinet_usd(tmp_path / "a" / "dishwasher_cabinet.usda")
    second = mcab.write_cabinet_usd(tmp_path / "b" / "dishwasher_cabinet.usda")
    assert first.read_bytes() == second.read_bytes()


@requires_usd
def test_committed_file_is_small_valid_and_equals_a_fresh_generation(tmp_path):
    assert CABINET.is_file(), CABINET
    assert CABINET.stat().st_size < 100_000
    info = mcab.check_cabinet_usd(CABINET)
    assert info["problems"] == []
    fresh = mcab.write_cabinet_usd(tmp_path / CABINET.name)
    assert lf_bytes(CABINET) == lf_bytes(fresh), (
        "dishwasher_cabinet.usda is stale or hand edited: regenerate with "
        "scripts/make_cabinet_usd.py"
    )


@requires_usd
def test_committed_numbers():
    info = mcab.check_cabinet_usd(CABINET)
    assert info["colliders"] == info["expected_colliders"] == 6
    assert info["gap_under_body_m"] == pytest.approx(kd.PLINTH_GAP_M, abs=1e-6)
    c = info["clearances"]
    assert c["door_panel_to_open_door_m"] == pytest.approx(0.0165, abs=1e-6)
    assert c["handle_to_open_door_m"] == pytest.approx(0.0765, abs=1e-6)
    assert c["door_panel_to_right_leg_m"] == pytest.approx(0.402, abs=1e-6)
    assert c["handle_to_right_leg_m"] == pytest.approx(0.370, abs=1e-6)


# --------------------------------------------------------------------------------------------
# The command line and --check on tampered copies (needs usd-core)
# --------------------------------------------------------------------------------------------


@requires_usd
def test_cli_writes_the_committed_file_and_checks_it(tmp_path):
    made = run_script(SCRIPT, "--out-dir", tmp_path)
    assert made.returncode == 0, made.stdout + made.stderr
    assert lf_bytes(tmp_path / "dishwasher_cabinet.usda") == lf_bytes(CABINET)
    ok = run_script(SCRIPT, "--check", CABINET)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "RESULT: OK" in ok.stdout
    assert "colliders: 6 (expected 6)" in ok.stdout
    assert "door panel 16.5 mm, handle 76.5 mm (minimum 10 mm)" in ok.stdout
    assert "door panel 402.0 mm, handle 370.0 mm (minimum 30 mm)" in ok.stdout


def _copy(tmp_path, name="dishwasher_cabinet.usda"):
    copy = tmp_path / name
    copy.write_bytes(CABINET.read_bytes())
    return copy


@requires_usd
def test_cli_check_accepts_a_sub_millimetre_nudge(tmp_path):
    copy = _copy(tmp_path)
    with edit_stage(copy) as stage:
        nudge(stage, "/DishwasherCabinet/Carcass/Body", dz=0.0004)
    assert run_script(SCRIPT, "--check", copy).returncode == 0


def _raise_the_door_into_the_open_dishwasher_door(stage):
    nudge(stage, "/DishwasherCabinet/Door/Panel", dz=0.0075)  # leaves 9 mm of the 16.5


def _raise_the_handle_to_the_open_door(stage):
    for part in ("Bar", "PostA", "PostB"):
        nudge(stage, f"/DishwasherCabinet/Handle/{part}", dz=0.0675)  # leaves 9 mm of the 76.5


def _move_the_door_to_the_leg(stage):
    nudge(stage, "/DishwasherCabinet/Door/Panel", dx=-0.38)  # leaves 22 mm of the 402


def _move_the_handle_to_the_leg(stage):
    for part in ("Bar", "PostA", "PostB"):
        nudge(stage, f"/DishwasherCabinet/Handle/{part}", dx=-0.35)  # leaves 20 mm of the 370


def _shift_the_carcass(stage):
    for part in ("Body", "Kick"):
        nudge(stage, f"/DishwasherCabinet/Carcass/{part}", dx=0.01)


def _lift_the_carcass_onto_the_dishwasher(stage):
    nudge(stage, "/DishwasherCabinet/Carcass/Body", dz=0.002)


def _remove_the_gap_under_the_body(stage):
    stage.GetPrimAtPath("/DishwasherCabinet/Carcass/Body").GetAttribute("xformOp:scale").Set(
        (0.3949, 0.4819, 0.452)  # 2 mm taller about the same centre: the gap is gone
    )


def _widen_the_carcass(stage):
    stage.GetPrimAtPath("/DishwasherCabinet/Carcass/Kick").GetAttribute("xformOp:scale").Set(
        (0.3349, 0.4919, 0.1)
    )


def _recolour_the_door(stage):
    from pxr import Gf, Vt

    stage.GetPrimAtPath("/DishwasherCabinet/Door/Panel").GetAttribute("primvars:displayColor").Set(
        Vt.Vec3fArray([Gf.Vec3f(0.2, 0.2, 0.8)])
    )


def _drop_a_collider(stage):
    from pxr import UsdPhysics

    stage.GetPrimAtPath("/DishwasherCabinet/Handle/Bar").RemoveAPI(UsdPhysics.CollisionAPI)


def _remove_a_post(stage):
    stage.RemovePrim("/DishwasherCabinet/Handle/PostB")


def _add_a_collider(stage):
    from pxr import UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, "/DishwasherCabinet/Carcass/Extra")
    cube.CreateSizeAttr(0.1)
    UsdGeom.Xformable(cube).AddTranslateOp().Set((0.8, -0.25, 0.3))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _make_rigid(stage):
    from pxr import UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/DishwasherCabinet/Carcass/Body"))


def _slippery(stage):
    stage.GetPrimAtPath("/DishwasherCabinet/PhysicsMaterial").GetAttribute(
        "physics:dynamicFriction"
    ).Set(0.1)


def _unbind(stage):
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(
        stage.GetPrimAtPath("/DishwasherCabinet/Door/Panel")
    ).UnbindAllBindings()


TAMPERS = [
    _raise_the_door_into_the_open_dishwasher_door,
    _raise_the_handle_to_the_open_door,
    _move_the_door_to_the_leg,
    _move_the_handle_to_the_leg,
    _shift_the_carcass,
    _lift_the_carcass_onto_the_dishwasher,
    _remove_the_gap_under_the_body,
    _widen_the_carcass,
    _recolour_the_door,
    _drop_a_collider,
    _remove_a_post,
    _add_a_collider,
    _make_rigid,
    _slippery,
    _unbind,
]


@requires_usd
@pytest.mark.parametrize("tamper", TAMPERS, ids=lambda f: f.__name__.strip("_"))
def test_cli_check_fails_on_a_tampered_cabinet(tamper, tmp_path):
    copy = _copy(tmp_path)
    with edit_stage(copy) as stage:
        tamper(stage)
    bad = run_script(SCRIPT, "--check", copy)
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "PROBLEM:" in bad.stdout and "RESULT: FAIL" in bad.stdout


@requires_usd
def test_the_clearance_failures_name_the_clearance(tmp_path):
    copy = _copy(tmp_path)
    with edit_stage(copy) as stage:
        _raise_the_door_into_the_open_dishwasher_door(stage)
    out = run_script(SCRIPT, "--check", copy).stdout
    assert "door_panel_to_open_door is 9.0 mm, less than 10 mm" in out
    copy = _copy(tmp_path, "leg.usda")
    with edit_stage(copy) as stage:
        _move_the_door_to_the_leg(stage)
    out = run_script(SCRIPT, "--check", copy).stdout
    assert "door_panel_to_right_leg is 22.0 mm, less than 30 mm" in out


@requires_usd
def test_cli_check_rejects_a_file_that_is_not_the_cabinet():
    result = run_script(SCRIPT, "--check", ASSETS / "counter.usda")
    assert result.returncode == 1
    assert "default prim is /Counter" in result.stdout


@requires_usd
def test_cli_check_of_a_missing_file_is_a_usage_error(tmp_path):
    assert run_script(SCRIPT, "--check", tmp_path / "nope.usda").returncode == 2
