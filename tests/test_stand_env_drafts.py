"""The #46 simulator drafts: parsed, never imported. Plus the import guard and the training YAML."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import kitchenbot
from kitchenbot.locomotion import obs_layout
from kitchenbot.locomotion import upper_body_targets as ubt
from kitchenbot.locomotion import waist_lean_targets as wlt

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "source" / "kitchenbot"
HEADER = (
    "# UNTESTED DRAFT (#46): written against Isaac Lab v3.0.0 source without a simulator; "
    "#5 runs and fixes it."
)
#: The drafts #68 edited carry both issue numbers.
HEADER_68 = HEADER.replace("(#46)", "(#46 and #68)")
EDITED_BY_68 = {
    "locomotion/commands.py",
    "locomotion/rewards.py",
    "locomotion/stand_env_cfg.py",
}
DRAFTS = [
    "robots/g1_dex3.py",
    "locomotion/commands.py",
    "locomotion/upper_body_action.py",
    "locomotion/rewards.py",
    "locomotion/curriculums.py",
    "locomotion/stand_env_cfg.py",
    "locomotion/agents/rsl_rl_ppo_cfg.py",
    "tasks/registry.py",
]
SIM_PREFIXES = ("isaaclab", "isaacsim", "omni", "torch", "pxr", "gymnasium")


def tree(relative: str) -> ast.Module:
    return ast.parse((PKG / relative).read_text(encoding="utf-8"), filename=relative)


def class_body(module: ast.Module, name: str) -> list[ast.stmt]:
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node.body
    raise AssertionError(f"class {name} not found")


def assigned_value(body: list[ast.stmt], name: str):
    """Literal value of ``name = ...`` or ``name: T = ...`` in ``body``."""
    for node in body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not assigned")


@pytest.mark.parametrize("relative", DRAFTS)
def test_draft_parses_and_starts_with_the_header(relative):
    text = (PKG / relative).read_text(encoding="utf-8")
    first = text.splitlines()[0]
    # The header is 106 characters; a trailing ruff suppression of E501 keeps the 100-column rule.
    assert first.split("  # noqa")[0] == (HEADER_68 if relative in EDITED_BY_68 else HEADER)
    ast.parse(text, filename=relative)
    compile(text, relative, "exec")


@pytest.mark.parametrize("relative", DRAFTS)
def test_draft_imports_a_simulator_package(relative):
    """Each draft really is simulator code, which is why nothing imports it on CI."""
    roots = set()
    for node in ast.walk(tree(relative)):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert any(root.startswith(SIM_PREFIXES) for root in roots), roots


@pytest.mark.parametrize(
    "relative", ["locomotion/__init__.py", "locomotion/agents/__init__.py", "tasks/__init__.py"]
)
def test_package_inits_import_nothing(relative):
    module = tree(relative)
    assert not [n for n in ast.walk(module) if isinstance(n, ast.Import | ast.ImportFrom)]
    assert ast.get_docstring(module)


def test_packages_and_cores_load_no_simulator_module():
    code = (
        "import sys\n"
        "import kitchenbot.locomotion, kitchenbot.locomotion.agents, kitchenbot.tasks\n"
        "import kitchenbot.locomotion.upper_body_targets, kitchenbot.locomotion.obs_layout\n"
        "import kitchenbot.locomotion.waist_lean_targets\n"
        f"bad = {SIM_PREFIXES!r}\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0].startswith(bad)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert out.stdout.strip() == "[]"


def test_policy_group_follows_the_layout_table():
    body = class_body(tree("locomotion/stand_env_cfg.py"), "PolicyCfg")
    names = [t.id for n in body if isinstance(n, ast.Assign) for t in n.targets]
    assert names == [term.name for term in obs_layout.FRAME_TERMS]


def test_registry_points_at_classes_in_the_drafts():
    tasks = assigned_value(tree("tasks/registry.py").body, "TASKS")
    assert list(tasks) == ["KitchenBot-Stand-G1-Dex3"]
    kwargs = tasks["KitchenBot-Stand-G1-Dex3"]
    for key in ("env_cfg_entry_point", "rsl_rl_cfg_entry_point"):
        module, cls = kwargs[key].split(":")
        relative = module.removeprefix("kitchenbot.").replace(".", "/") + ".py"
        assert relative in DRAFTS
        class_body(tree(relative), cls)


# -- the waist lean command (#68, D012) -------------------------------------------------------


def assigned_node(body: list[ast.stmt], name: str) -> ast.expr:
    """The right-hand side of ``name = ...`` or ``name: T = ...`` in ``body``, unevaluated."""
    for node in body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return node.value
    raise AssertionError(f"{name} not assigned")


def call_target(node: ast.expr) -> str:
    """Dotted name of the callee of a call node, e.g. ``mdp.generated_commands``."""
    assert isinstance(node, ast.Call), ast.dump(node)
    return ast.unparse(node.func)


def keyword_node(call: ast.expr, name: str) -> ast.expr:
    """The unevaluated keyword argument ``name`` of a call node."""
    assert isinstance(call, ast.Call)
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    raise AssertionError(f"no keyword {name}")


def keyword_value(call: ast.expr, name: str):
    """Literal value of the keyword argument ``name`` of a call node."""
    return ast.literal_eval(keyword_node(call, name))


def dict_entry(node: ast.expr, key: str) -> ast.expr:
    """The unevaluated value stored under the string ``key`` of a dict display."""
    assert isinstance(node, ast.Dict)
    for k, v in zip(node.keys, node.values, strict=True):
        if isinstance(k, ast.Constant) and k.value == key:
            return v
    raise AssertionError(f"no key {key}")


def assigned_names(body: list[ast.stmt]) -> list[str]:
    names = []
    for node in body:
        if isinstance(node, ast.Assign):
            names += [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return names


def test_commands_cfg_has_the_waist_lean_term():
    body = class_body(tree("locomotion/stand_env_cfg.py"), "CommandsCfg")
    assert assigned_names(body) == ["base_velocity", "upper_body", "waist_lean"]
    term = assigned_node(body, "waist_lean")
    assert call_target(term) == "WaistLeanCommandCfg"
    assert keyword_value(term, "asset_name") == "robot"


def test_waist_lean_cfg_defaults_are_the_tested_constants():
    """The config reads every default from ``waist_lean_targets``, which the numpy tests pin."""
    body = class_body(tree("locomotion/commands.py"), "WaistLeanCommandCfg")
    fields = {
        "joint_name": "WAIST_PITCH_JOINT",
        "resampling_time_range": "RESAMPLE_TIME_RANGE_S",
        "max_lean_rad": "MAX_LEAN_RAD",
        "zero_share": "ZERO_SHARE",
        "max_rate_rad_s": "MAX_RATE_RAD_S",
    }
    for field, constant in fields.items():
        assert ast.unparse(assigned_node(body, field)) == f"wlt.{constant}"
        assert hasattr(wlt, constant)
    assert ast.unparse(assigned_node(body, "class_type")) == "WaistLeanCommand"
    assert assigned_value(body, "seed") is None


def test_waist_lean_term_implements_the_command_term_interface():
    module = tree("locomotion/commands.py")
    cls = next(
        n for n in ast.walk(module) if isinstance(n, ast.ClassDef) and n.name == "WaistLeanCommand"
    )
    assert [ast.unparse(b) for b in cls.bases] == ["CommandTerm"]
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    # the three abstract methods of CommandTerm (v3.0.0 command_manager.py:190-202), the abstract
    # ``command`` property, and ``reset`` to finalize the per-episode metric
    required = {"command", "_resample_command", "_update_command", "_update_metrics", "reset"}
    assert required <= methods
    assert "'pitch_error'" in ast.unparse(cls) or '"pitch_error"' in ast.unparse(cls)
    # sampling and ramping come from the numpy module, not from re-implemented torch code
    called = {call_target(n) for n in ast.walk(cls) if isinstance(n, ast.Call)}
    assert {"wlt.sample_lean_targets", "wlt.slew_toward"} <= called


def test_policy_observes_the_waist_lean_command_as_one_value():
    body = class_body(tree("locomotion/stand_env_cfg.py"), "PolicyCfg")
    term = assigned_node(body, "waist_lean_command")
    assert call_target(term) == "ObsTerm"
    assert ast.unparse(keyword_node(term, "func")) == "mdp.generated_commands"
    assert keyword_value(term, "params") == {"command_name": "waist_lean"}
    assert not [kw for kw in term.keywords if kw.arg == "noise"]  # commands carry no noise
    (layout_term,) = [t for t in obs_layout.FRAME_TERMS if t.name == "waist_lean_command"]
    assert layout_term.dim == 1


def test_every_command_name_in_the_env_config_exists():
    module = tree("locomotion/stand_env_cfg.py")
    defined = set(assigned_names(class_body(module, "CommandsCfg")))
    used = set()
    for node in ast.walk(module):
        if isinstance(node, ast.keyword) and node.arg == "command_name":
            used.add(ast.literal_eval(node.value))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "command_name":
                    used.add(ast.literal_eval(value))
    assert {"base_velocity", "upper_body", "waist_lean"} <= used
    assert used <= defined, used - defined


def test_waist_reward_tracks_the_command_instead_of_zero():
    module = tree("locomotion/stand_env_cfg.py")
    term = assigned_node(class_body(module, "RewardsCfg"), "waist_deviation")
    assert call_target(term) == "RewTerm"
    assert ast.unparse(keyword_node(term, "func")) == "waist_tracking_l1"
    assert keyword_value(term, "weight") == -0.1  # provisional, #5 tunes it
    params = keyword_node(term, "params")
    assert ast.literal_eval(dict_entry(params, "command_name")) == "waist_lean"
    assert ast.unparse(dict_entry(params, "asset_cfg")) == "_joints(list(WAIST))"
    # the zero-pulling reward is gone from the config and from the reward module
    assert "waist_deviation_l1" not in {n.id for n in ast.walk(module) if isinstance(n, ast.Name)}
    rewards = tree("locomotion/rewards.py")
    functions = {n.name: n for n in rewards.body if isinstance(n, ast.FunctionDef)}
    assert "waist_deviation_l1" not in functions
    assert [a.arg for a in functions["waist_tracking_l1"].args.args] == [
        "env",
        "command_name",
        "asset_cfg",
    ]


def test_waist_reward_reads_the_waist_joints_in_joint_maps_order():
    """The reward indexes the columns by position, so the selection must keep ``WAIST`` order."""
    default = assigned_node(tree("locomotion/rewards.py").body, "WAIST_JOINTS")
    assert call_target(default) == "SceneEntityCfg"
    assert ast.unparse(keyword_node(default, "joint_names")) == "list(WAIST)"
    assert keyword_value(default, "preserve_order") is True
    # the environment config builds the same selection through its own ``_joints`` helper
    env = tree("locomotion/stand_env_cfg.py")
    helper = next(n for n in env.body if isinstance(n, ast.FunctionDef) and n.name == "_joints")
    assert "preserve_order=True" in ast.unparse(helper)


def test_the_numpy_module_imports_no_simulator_package():
    roots = set()
    for node in ast.walk(tree("locomotion/waist_lean_targets.py")):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert not {r for r in roots if r.startswith(SIM_PREFIXES)}, roots


TRAIN_YAML = REPO / "configs" / "train" / "stand_g1_dex3.yaml"


def test_training_yaml_names_the_physx_preset_and_the_budget():
    cfg = yaml.safe_load(TRAIN_YAML.read_text(encoding="utf-8"))
    assert cfg["physics"] == kitchenbot.PHYSICS_PRESET == "isaacsim_physx"
    assert cfg["task"] == "KitchenBot-Stand-G1-Dex3"
    assert cfg["rl_library"] == "rsl_rl"
    assert cfg["num_envs"] == 1024
    assert cfg["max_iterations"] == 1500
    assert isinstance(cfg["seed"], int)


def test_training_yaml_mirrors_the_code():
    cfg = yaml.safe_load(TRAIN_YAML.read_text(encoding="utf-8"))
    upper = cfg["commands"]["upper_body"]
    assert tuple(upper["resampling_time_s"]) == ubt.RESAMPLE_TIME_RANGE_S
    assert upper["range_factor_start"] == ubt.RANGE_FACTOR_START
    assert upper["range_factor_end"] == ubt.RANGE_FACTOR_END
    assert upper["reach_episode_fraction"] == ubt.REACH_EPISODE_FRACTION
    assert tuple(upper["reach_duration_s"]) == ubt.REACH_DURATION_RANGE_S
    lean = cfg["commands"]["waist_lean"]
    assert lean["joint"] == wlt.WAIST_PITCH_JOINT
    assert tuple(lean["resampling_time_s"]) == wlt.RESAMPLE_TIME_RANGE_S
    assert lean["max_lean_rad"] == wlt.MAX_LEAN_RAD
    assert lean["zero_share"] == wlt.ZERO_SHARE
    assert lean["max_rate_rad_s"] == wlt.MAX_RATE_RAD_S
    obs = cfg["observation"]
    assert obs["frame_dim"] == obs_layout.FRAME_DIM
    assert obs["history_length"] == obs_layout.HISTORY_LENGTH
    assert obs["flat_dim"] == obs_layout.FLAT_DIM

    agent = class_body(tree("locomotion/agents/rsl_rl_ppo_cfg.py"), "StandPPORunnerCfg")
    assert assigned_value(agent, "max_iterations") == cfg["max_iterations"]
    assert assigned_value(agent, "num_steps_per_env") == cfg["agent"]["num_steps_per_env"]
    assert assigned_value(agent, "seed") == cfg["seed"]
    hidden = assigned_value(tree("locomotion/agents/rsl_rl_ppo_cfg.py").body, "HIDDEN_DIMS")
    assert hidden == cfg["agent"]["actor_hidden_dims"] == cfg["agent"]["critic_hidden_dims"]
