"""The #59 simulator drafts (dishwasher, kitchen scene, smoke script): parsed, never imported."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from kitchenbot.scene import dishwasher_frames as df

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "source" / "kitchenbot"
HEADER = (
    "# UNTESTED DRAFT (#59): written against Isaac Lab v3.0.0 source without a simulator; "
    "#10 and #11 run and fix it."
)
DRAFTS = [
    "source/kitchenbot/scene/dishwasher_cfg.py",
    "source/kitchenbot/scene/kitchen_scene_cfg.py",
    "scripts/smoke/load_kitchen.py",
]
SIM_PREFIXES = ("isaaclab", "isaacsim", "omni", "torch", "pxr", "gymnasium")


def tree(relative: str) -> ast.Module:
    return ast.parse((REPO / relative).read_text(encoding="utf-8"), filename=relative)


def module_value(relative: str, name: str):
    for node in tree(relative).body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return node.value
    raise AssertionError(f"{name} not assigned in {relative}")


def calls(relative: str, func: str) -> list[ast.Call]:
    out = []
    for node in ast.walk(tree(relative)):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name == func:
                out.append(node)
    return out


def keyword(call: ast.Call, name: str) -> ast.expr:
    return next(k.value for k in call.keywords if k.arg == name)


@pytest.mark.parametrize("relative", DRAFTS)
def test_draft_parses_and_starts_with_the_header(relative):
    text = (REPO / relative).read_text(encoding="utf-8")
    first = text.splitlines()[0]
    assert first.split("  # noqa")[0] == HEADER
    ast.parse(text, filename=relative)
    compile(text, relative, "exec")


@pytest.mark.parametrize("relative", DRAFTS)
def test_draft_imports_a_simulator_package(relative):
    roots = set()
    for node in ast.walk(tree(relative)):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert any(root.startswith(SIM_PREFIXES) for root in roots), roots


def test_tested_scene_modules_load_no_simulator_module():
    code = (
        "import sys\n"
        "import kitchenbot.scene, kitchenbot.scene.kitchen_dims\n"
        "import kitchenbot.scene.dishwasher_frames\n"
        "import kitchenbot.scene.randomization\n"
        f"bad = {SIM_PREFIXES!r}\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0].startswith(bad)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert out.stdout.strip() == "[]"


def test_dishwasher_actuators_drive_the_joint_prims():
    """The groups are keyed by link but match joints by the joint prims' names (#29 used links)."""
    actuators = calls("source/kitchenbot/scene/dishwasher_cfg.py", "ImplicitActuatorCfg")
    assert len(actuators) == 3
    joints, gains = [], []
    for call in actuators:
        (expr,) = keyword(call, "joint_names_expr").elts
        joints.append(expr.attr)
        gains.append((keyword(call, "stiffness").attr, keyword(call, "damping").attr))
    assert joints == ["DOOR_JOINT", "UPPER_RACK_JOINT", "LOWER_RACK_JOINT"]
    assert gains[0] == ("DOOR_STIFFNESS_PER_RAD", "DOOR_DAMPING_PER_RAD")
    assert gains[1] == gains[2] == ("RACK_STIFFNESS", "RACK_DAMPING")
    assert {df.DOOR_JOINT, df.UPPER_RACK_JOINT, df.LOWER_RACK_JOINT} == {
        "RevoluteJoint_dishwasher_2_middle",
        "PrismaticJoint_dishwasher_2_up",
        "PrismaticJoint_dishwasher_2_down",
    }


def test_dishwasher_spawns_the_overrides_layer_with_contact_sensors():
    (spawn, tines_spawn) = calls("source/kitchenbot/scene/dishwasher_cfg.py", "UsdFileCfg")
    assert keyword(spawn, "activate_contact_sensors").value is True
    assert ast.unparse(keyword(spawn, "usd_path")) == "str(DISHWASHER_USD)"
    assert ast.unparse(keyword(tines_spawn, "usd_path")) == "str(RACK_TINES_USD)"
    assert "rack_tines_scale_in_link" in ast.unparse(keyword(tines_spawn, "scale"))
    for name, path in (
        ("DISHWASHER_USD", "dishwasher_2_overrides.usda"),
        ("RACK_TINES_USD", "rack_tines.usda"),
    ):
        assert path in ast.unparse(module_value("source/kitchenbot/scene/dishwasher_cfg.py", name))
        assert (PKG / "scene" / "assets" / "dishwasher" / path).is_file()


def test_tines_are_a_child_of_the_lower_rack_link():
    value = module_value("source/kitchenbot/scene/dishwasher_cfg.py", "ASSET_CHILD")
    assert ast.literal_eval(value) == "ArtVIP"  # the referencing prim in the overrides layer
    text = (REPO / "source/kitchenbot/scene/dishwasher_cfg.py").read_text(encoding="utf-8")
    assert 'f"{lower_rack_link_path(dishwasher_prim_path)}/{TINES_CHILD}"' in text


def test_scene_layers_and_their_fields():
    relative = "source/kitchenbot/scene/kitchen_scene_cfg.py"
    layers = ast.literal_eval(module_value(relative, "LAYERS"))
    fields = ast.literal_eval(module_value(relative, "LAYER_FIELDS"))
    assert layers == ("counter", "dishwasher", "plates")
    assert tuple(fields) == layers
    scene = next(
        n
        for n in ast.walk(tree(relative))
        if isinstance(n, ast.ClassDef) and n.name == "KitchenSceneCfg"
    )
    declared = [n.target.id for n in scene.body if isinstance(n, ast.AnnAssign)]
    assert {"floor", "robot", "light"} <= set(declared)
    for names in fields.values():
        assert set(names) <= set(declared)
    files = ast.literal_eval(module_value(relative, "PLATE_FILES"))
    for name in files.values():
        assert (PKG / "scene" / "assets" / "plates" / name).is_file()
    for name in ("counter.usda", "floor.usda"):
        assert (PKG / "scene" / "assets" / "kitchen" / name).is_file()


def test_smoke_script_defaults():
    relative = "scripts/smoke/load_kitchen.py"
    arguments = {ast.literal_eval(c.args[0]): c for c in calls(relative, "add_argument") if c.args}
    assert {"--layers", "--num_envs", "--steps", "--fixed-base", "--headless"} <= set(arguments)
    assert ast.literal_eval(keyword(arguments["--steps"], "default")) == 500
    layers = ast.literal_eval(keyword(arguments["--layers"], "default")).split(",")
    scene_layers = ast.literal_eval(
        module_value("source/kitchenbot/scene/kitchen_scene_cfg.py", "LAYERS")
    )
    assert set(layers) <= set(scene_layers)
    text = (REPO / relative).read_text(encoding="utf-8")
    assert "set_position_index" in text and "PhysxCfg()" in text
