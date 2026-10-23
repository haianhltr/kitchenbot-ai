"""The ArtVIP dishwasher_2 overrides layer (#59). No simulator.

CI has usd-core but not the ArtVIP file (it is git-ignored), so the layer's own content is checked
everywhere and the composition only where ``scripts/fetch_assets.py --set dishwasher`` has run;
those tests skip otherwise. The pure-Python ones always run.
"""

import shutil
import sys

import pytest
from usd_helpers import (
    FORBIDDEN_IMPORTS,
    REPO,
    imported_top_level_modules,
    lf_bytes,
    load_script,
    requires_usd,
    run_script,
)

from kitchenbot.scene import dishwasher_frames as df
from kitchenbot.scene import kitchen_dims as kd

SCRIPT = "make_dishwasher_overrides"
DISHWASHER_DIR = REPO / "source" / "kitchenbot" / "scene" / "assets" / "dishwasher"
COMMITTED = DISHWASHER_DIR / "dishwasher_2_overrides.usda"
THIRD_PARTY = REPO / "source" / "kitchenbot" / "scene" / "assets" / "third_party"
ARTVIP_FILE = THIRD_PARTY / df.ARTVIP_DISHWASHER_2
AUTHORED_TRAVEL = -kd.DISHWASHER_RACK_LIMITS_M[0]

mo = load_script(SCRIPT)

requires_asset = pytest.mark.skipif(
    not ARTVIP_FILE.is_file(), reason="ArtVIP dishwasher_2 not fetched (git-ignored; CI has none)"
)


# --------------------------------------------------------------------------------------------
# No pxr needed
# --------------------------------------------------------------------------------------------


def test_script_and_frames_import_no_simulator():
    # The script imports pxr only inside functions, behind the check in main(); the modules never.
    script = imported_top_level_modules(REPO / "scripts" / f"{SCRIPT}.py")
    assert script.isdisjoint(FORBIDDEN_IMPORTS)
    for path in (
        REPO / "source/kitchenbot/scene/dishwasher_frames.py",
        REPO / "source/kitchenbot/scene/kitchen_dims.py",
    ):
        assert imported_top_level_modules(path).isdisjoint(FORBIDDEN_IMPORTS | {"pxr"}), path


def test_help_and_the_missing_pxr_message_work_without_pxr(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "pxr", None)
    module = load_script(SCRIPT)
    with pytest.raises(SystemExit) as stop:
        module.main(["--help"])
    assert stop.value.code == 0
    assert "--rack-travel" in capsys.readouterr().out
    out = tmp_path / "o.usda"
    assert module.main(["--out", str(out)]) == 2
    assert "usd-core" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    "flags", [["--rack-travel", "0"], ["--rack-travel", "-0.2"], ["--asset-scale", "0"]]
)
def test_impossible_flags_are_a_usage_error(flags, tmp_path):
    result = run_script(SCRIPT, *flags, "--out", tmp_path / "x.usda")
    assert result.returncode == 2
    assert not (tmp_path / "x.usda").exists()


def test_the_layer_covers_every_inspected_collider():
    assert len(mo.COLLIDERS) == 13  # #29: 13 sdf colliders in dishwasher_2
    racks = {k for k, v in mo.COLLIDERS.items() if v == mo.DECOMPOSITION}
    assert racks == {
        f"{df.UPPER_RACK_LINK}/P_1440b71c964e2ed8",
        f"{df.LOWER_RACK_LINK}/P_23f8dca1790b2ed8",
    }
    assert {k.split("/")[0] for k, v in mo.COLLIDERS.items() if v == mo.HULL} == {
        df.BODY_LINK,
        df.DOOR_LINK,
    }
    assert mo.ARTVIP_REFERENCE == "../third_party/" + df.ARTVIP_DISHWASHER_2
    assert mo.asset_offset() == pytest.approx([-v for v in kd.DISHWASHER_ROOT_LINK_IN_ASSET_M])


def test_joint_values_keep_the_authored_limits():
    values = mo.joint_values(AUTHORED_TRAVEL)
    rack = values[mo.LOWER_RACK_JOINT_PATH]
    assert rack == {
        "physics:lowerLimit": -0.2,
        "physics:upperLimit": 0.0,
        "drive:linear:physics:targetPosition": -0.2,
    }
    door = values[mo.DOOR_JOINT_PATH]
    assert door["drive:angular:physics:targetPosition"] == 90.0
    assert (door["physics:lowerLimit"], door["physics:upperLimit"]) == (0.0, 90.0)


# --------------------------------------------------------------------------------------------
# The layer's own content (usd-core, no ArtVIP file)
# --------------------------------------------------------------------------------------------


@requires_usd
def test_committed_layer_content():
    assert COMMITTED.is_file() and COMMITTED.stat().st_size < 50_000
    assert mo.check_layer_content(COMMITTED, AUTHORED_TRAVEL) == []
    text = COMMITTED.read_text(encoding="utf-8")
    assert "sdf" not in text.replace("PhysxSDFMeshCollisionAPI", "")
    assert text.count('physics:approximation = "convexDecomposition"') == 2
    assert text.count('physics:approximation = "convexHull"') == 11


@requires_usd
def test_committed_layer_is_what_the_generator_writes(tmp_path):
    fresh = mo.write_overrides(tmp_path / COMMITTED.name)
    assert lf_bytes(fresh) == lf_bytes(COMMITTED), "regenerate: make_dishwasher_overrides.py"


@requires_usd
def test_check_without_the_asset_skips_composition(tmp_path):
    out = mo.write_overrides(tmp_path / "o.usda")  # the relative reference does not resolve here
    result = run_script(SCRIPT, "--check", out)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "composition: SKIPPED" in result.stdout and "RESULT: OK" in result.stdout
    strict = run_script(SCRIPT, "--check", out, "--require-asset")
    assert strict.returncode == 1 and "--require-asset" in strict.stdout


@requires_usd
@pytest.mark.parametrize(
    ("rel_path", "name", "value", "needle"),
    [
        (
            f"{df.LOWER_RACK_LINK}/P_23f8dca1790b2ed8",
            "physics:approximation",
            "sdf",
            "approximation",
        ),
        (mo.LOWER_RACK_JOINT_PATH, "physics:lowerLimit", -0.3, "lowerLimit"),
        (mo.DOOR_JOINT_PATH, "drive:angular:physics:targetPosition", 45.0, "targetPosition"),
    ],
)
def test_content_check_catches_a_hand_edit(tmp_path, rel_path, name, value, needle):
    from pxr import Sdf

    path = mo.write_overrides(tmp_path / "o.usda")
    layer = Sdf.Layer.FindOrOpen(str(path))
    layer.GetPrimAtPath(f"{mo.ASSET_PATH}/{rel_path}").attributes[name].default = value
    layer.Save()
    problems = mo.check_layer_content(path, AUTHORED_TRAVEL)
    assert any(needle in p for p in problems), problems


@requires_usd
def test_content_check_catches_a_kept_behaviour_script(tmp_path):
    from pxr import Sdf

    path = mo.write_overrides(tmp_path / "o.usda")
    layer = Sdf.Layer.FindOrOpen(str(path))
    layer.GetPrimAtPath(f"{mo.ASSET_PATH}/{df.BODY_LINK}").ClearInfo("apiSchemas")
    layer.Save()
    assert any("OmniScriptingAPI" in p for p in mo.check_layer_content(path, AUTHORED_TRAVEL))


@requires_usd
def test_raised_travel_for_11(tmp_path):
    path = mo.write_overrides(tmp_path / "o.usda", rack_travel=kd.DISHWASHER_RACK_TRAVEL_M)
    assert mo.check_layer_content(path, kd.DISHWASHER_RACK_TRAVEL_M) == []
    assert mo.check_layer_content(path, AUTHORED_TRAVEL) != []


@requires_usd
def test_authored_tines_and_counter_still_check():
    tines = DISHWASHER_DIR / "rack_tines.usda"
    kitchen = REPO / "source/kitchenbot/scene/assets/kitchen"
    ok = run_script("make_rack_tines_usd", "--check", tines)
    assert ok.returncode == 0 and "RESULT: OK" in ok.stdout, ok.stdout
    ok = run_script("make_counter_usd", "--check", kitchen / "counter.usda", kitchen / "floor.usda")
    assert ok.returncode == 0 and "FAIL" not in ok.stdout, ok.stdout


# --------------------------------------------------------------------------------------------
# Composition with the ArtVIP file (skipped where it is not fetched)
# --------------------------------------------------------------------------------------------


def _absolute_copy(tmp_path, source, **edits):
    """Copy a generated layer into ``tmp_path`` with its references made absolute."""
    text = source.read_text(encoding="utf-8")
    text = text.replace(f"@{mo.ARTVIP_REFERENCE}@", f"@{ARTVIP_FILE.as_posix()}@")
    text = text.replace(
        f"@{mo.TINES_REFERENCE}@", f"@{(DISHWASHER_DIR / 'rack_tines.usda').as_posix()}@"
    )
    for old, new in edits.items():
        text = text.replace(old, new)
    out = tmp_path / source.name
    out.write_text(text, encoding="utf-8")
    return out


@requires_usd
@requires_asset
def test_committed_layer_composes_and_holds_its_overrides():
    info = mo.check_overrides(COMMITTED, AUTHORED_TRAVEL, require_asset=True)
    assert info["composed"] and info["problems"] == [], info["problems"]
    assert info["colliders"] == 13
    assert info["approximations"] == {"convexHull": 11, "convexDecomposition": 2}
    assert info["max_slot_error_m"] < 1e-3
    assert info["tine_scale"] == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)
    result = run_script(SCRIPT, "--check", COMMITTED, "--require-asset")
    assert result.returncode == 0 and "RESULT: OK" in result.stdout, result.stdout


@requires_usd
@requires_asset
def test_composition_catches_an_sdf_collider(tmp_path):
    token = 'uniform token physics:approximation = "{}"'
    edit = {token.format("convexDecomposition"): token.format("sdf")}
    bad = _absolute_copy(tmp_path, COMMITTED, **edit)
    problems = mo.check_composed(bad, AUTHORED_TRAVEL, {})
    assert any("'sdf'" in p for p in problems), problems


@requires_usd
@requires_asset
def test_with_tines_fallback_composes(tmp_path):
    generated = mo.write_overrides(tmp_path / "gen" / "o.usda", with_tines=True)
    layer = _absolute_copy(tmp_path, generated)
    info: dict = {}
    assert mo.check_composed(layer, AUTHORED_TRAVEL, info) == []
    assert info["tines"].startswith("in the layer")
    assert info["max_slot_error_m"] < 1e-3


@requires_usd
@requires_asset
def test_the_artvip_file_is_not_copied():
    assert ARTVIP_FILE.is_file()
    tracked = shutil.which("git")
    if tracked is None:
        pytest.skip("git not on PATH")
    import subprocess

    out = subprocess.run(
        ["git", "ls-files", "--", str(THIRD_PARTY)], capture_output=True, text=True, cwd=REPO
    )
    assert out.stdout.split() == ["source/kitchenbot/scene/assets/third_party/MANIFEST.md"]
