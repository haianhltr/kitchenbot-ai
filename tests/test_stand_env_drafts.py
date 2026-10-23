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

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "source" / "kitchenbot"
HEADER = (
    "# UNTESTED DRAFT (#46): written against Isaac Lab v3.0.0 source without a simulator; "
    "#5 runs and fixes it."
)
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
    # The header is 106 characters; a trailing ``# noqa: E501`` keeps ruff's 100-column rule.
    assert first.split("  # noqa")[0] == HEADER
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
