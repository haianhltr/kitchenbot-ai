"""Cross-module contracts (#69): numbers, enums, frames and units that live in more than one place.

Each test pins one equality between two or more modules, or between a module and the document it
cites, so that the next "14.9 cm against 15 cm" (#60) fails here instead of in a simulation run.
The full audit, with the mismatches that are reported rather than pinned, is the findings comment
on issue #69. #78 added the pins of B7 (a ``--config`` run without ``--num_envs`` uses the YAML's
``num_envs``) and B8 (the fixed-pelvis YAMLs are the section 1 scenarios with a fixed pelvis).

Pure Python: no simulator, no ``pxr`` (the tines file is read as text), the UNTESTED DRAFTS only
parsed with ``ast``. Deliberately absent: the dry run's outcome counts and the grasp-frame rim
point (#66 and #67 move them), the stand environment's waist command other than its maximum (#68;
the maximum is one constant since #77 and pinned here) and the cabinet constants of
``kitchen_dims`` (#70). Values are compared with each other, never with a literal,
except where a document states the number in words.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import math
import re
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest
import yaml

from kitchenbot.controllers import reachability, target_filter
from kitchenbot.controllers.hand import kinematic_follow
from kitchenbot.controllers.hand.dex3_synergy import Dex3SynergyHand
from kitchenbot.controllers.hand.mock import MockHand
from kitchenbot.evaluation import dry_run, fake, judges, metrics, stats
from kitchenbot.evaluation.cli import main as evaluate_main
from kitchenbot.evaluation.failure_tags import FAILURE_TAG_VALUES, FailureTag
from kitchenbot.evaluation.runner import (
    BASES,
    FIXED_BASE_SCENARIOS,
    EvalConfig,
    get_scenario,
    scenario_names,
)
from kitchenbot.evaluation.schema import load_schema
from kitchenbot.evaluation.seeds import SPLITS, default_seeds
from kitchenbot.geometry import Pose, quat_from_yaw, tilt_from_vertical, to_frame
from kitchenbot.locomotion import waist_lean_targets
from kitchenbot.planning import fakes
from kitchenbot.planning.plate_to_rack import (
    ARCH_9,
    EVAL_3,
    PlannerConfig,
    PlannerState,
    plan_waypoints,
)
from kitchenbot.robots.joint_limits import SOFT_LIMIT_FRACTION, WAIST_LEAN_MAX_RAD, WAIST_LIMITS
from kitchenbot.scene import dishwasher_frames as df
from kitchenbot.scene import kitchen_dims as kd
from kitchenbot.scene.randomization import (
    SCENARIOS,
    RandomizationRanges,
    SpawnDims,
    sample_episode,
)

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "source" / "kitchenbot"
EVAL_CONFIGS = REPO / "configs" / "eval"
TRAIN_YAML = REPO / "configs" / "train" / "stand_g1_dex3.yaml"
RACK_TINES_USDA = PKG / "scene" / "assets" / "dishwasher" / "rack_tines.usda"

CFG = PlannerConfig()


def _section(doc: str, number: str) -> str:
    """The text of section ``number`` (a level-2 heading ``## <number>. ...``) of ``docs/<doc>``."""
    text = (REPO / "docs" / doc).read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(number)}\. .*?(?=^## |\Z)", text, re.M | re.S)
    assert match, f"docs/{doc} has no section {number}"
    return match.group(0)


def _flat(text: str) -> str:
    """Whitespace collapsed, so a phrase broken across Markdown lines still matches."""
    return " ".join(text.split())


def _cm(metres: float) -> str:
    return f"{metres * 100:g}"


ARCH_5 = _flat(_section("architecture.md", "5"))
ARCH_6 = _flat(_section("architecture.md", "6"))
ARCH_7 = _flat(_section("architecture.md", "7"))
ARCH_8 = _flat(_section("architecture.md", "8"))
ARCH_9_TEXT = _section("architecture.md", "9")
ARCH_9_FLAT = _flat(ARCH_9_TEXT)
EVAL_1 = _flat(_section("evaluation.md", "1"))
EVAL_2 = _flat(_section("evaluation.md", "2"))
EVAL_3_FLAT = _flat(_section("evaluation.md", "3"))
EVAL_7 = _section("evaluation.md", "7")


# ------------------------------------------------------------------------------------------------
# The planner, the judges and the protocol
# ------------------------------------------------------------------------------------------------


def test_lift_height_is_one_number_in_the_planner_the_judges_and_the_protocol():
    assert CFG.lift_height_m == judges.GRASP_LIFT_HEIGHT_M
    assert judges.THRESHOLDS["grasp_lift_height_m"].value == CFG.lift_height_m
    assert f"lifts {_cm(CFG.lift_height_m)} cm" in EVAL_1
    assert f"plate lifted {_cm(CFG.lift_height_m)} cm" in EVAL_3_FLAT
    assert f"lift(+{_cm(CFG.lift_height_m)} cm)" in ARCH_9_FLAT
    # #60: the lift state exits within one position tolerance of its waypoint, so the waypoint
    # must be aimed at least that far above the judge's height.
    grasp = Pose((0.1, 0.3, kd.COUNTER_HEIGHT_M))
    edge = Pose((0.3, 0.0, kd.COUNTER_HEIGHT_M))
    lift = plan_waypoints(grasp, Pose((0.35, -0.26, 0.74)), edge, CFG)["lift"]
    worst_exit = float(lift.position[2]) - CFG.position_tolerance_m
    assert worst_exit - float(grasp.position[2]) >= judges.GRASP_LIFT_HEIGHT_M - 1e-9


def test_judge_thresholds_are_the_planner_numbers_and_quote_section_3():
    pairs = {
        "placement_vertical_tol_deg": "vertical_tolerance_deg",
        "placement_stable_s": "hold_check_s",
        "rack_collision_force_n": "rack_collision_force_n",
        "grasp_lift_height_m": "lift_height_m",
    }
    for judge_name, planner_name in pairs.items():
        assert judges.THRESHOLDS[judge_name].value == getattr(CFG, planner_name), judge_name
    assert judges.THRESHOLDS["grasp_hold_s"].source == EVAL_3
    assert f"held for {judges.GRASP_HOLD_S:g} s" in EVAL_3_FLAT
    for q in (quat_from_yaw(0.3), (0.9, 0.2, -0.3, 0.1), (0.5, 0.5, 0.5, 0.5)):
        assert judges.vertical_error_deg(q) == pytest.approx(math.degrees(tilt_from_vertical(q)))


def test_planner_config_sources_quote_the_documents_they_cite():
    phrases = {
        "pre_grasp_height_m": (ARCH_9_FLAT, f"+ {_cm(CFG.pre_grasp_height_m)} cm above rim"),
        "lift_height_m": (ARCH_9_FLAT, f"lift(+{_cm(CFG.lift_height_m)} cm)"),
        "above_slot_height_m": (ARCH_9_FLAT, f"at +{_cm(CFG.above_slot_height_m)} cm"),
        "retreat_height_m": (ARCH_9_FLAT, f"retreat {_cm(CFG.retreat_height_m)} cm up"),
        "vertical_tolerance_deg": (
            EVAL_3_FLAT,
            f"within {CFG.vertical_tolerance_deg:g} deg of vertical",
        ),
        "rack_collision_force_n": (
            EVAL_3_FLAT,
            f"contact force above {CFG.rack_collision_force_n:g} N before release",
        ),
        "hold_check_s": (EVAL_3_FLAT, f"stable for {CFG.hold_check_s:g} s after release"),
        "total_budget_s": (EVAL_3_FLAT, f"within the {CFG.total_budget_s:g} s timeout"),
    }
    cites = {ARCH_9: ARCH_9_FLAT, EVAL_3: EVAL_3_FLAT}
    cited = {
        f.name: f.metadata["source"]
        for f in dataclasses.fields(PlannerConfig)
        if f.metadata["source"] in cites
    }
    assert set(cited) == set(phrases), "a field citing arch s9 or eval s3 needs its phrase here"
    assert all(f.metadata["source"] for f in dataclasses.fields(PlannerConfig))
    for name, (text, phrase) in phrases.items():
        assert cites[cited[name]] is text, f"{name} cites {cited[name]}"
        assert phrase in text, f"{name}: {phrase!r} not in {cited[name]}"


def test_provisional_planner_numbers_are_the_ones_architecture_section_9_lists():
    c = CFG
    for phrase in (
        f"tolerances ({_cm(c.position_tolerance_m)} cm, {c.orientation_tolerance_deg:g} deg)",
        f"the {c.seat_contact_force_n:g} N seat contact",
        f"the {_cm(c.insertion_depth_m)} cm insertion depth",
        f"the {_cm(c.counter_edge_clearance_m)} cm counter-edge clearance",
        f"rack contact of {c.seat_contact_force_n:g} N to {c.rack_collision_force_n:g} N, or palm "
        f"{_cm(c.insertion_depth_m)} cm below the slot entry",
        f"A {c.total_budget_s:g} s budget",
        f"The timeouts sum to {sum(c.state_timeouts().values()):g} s",
    ):
        assert phrase in ARCH_9_FLAT, phrase
    rows = re.findall(r"^\| `(\w+)` \| ([0-9.]+) s \|", ARCH_9_TEXT, re.M)
    table = {PlannerState(name): float(value) for name, value in rows}
    assert table == c.state_timeouts()
    assert list(table) == list(c.state_timeouts())


def test_planner_states_follow_the_architecture_fsm():
    block = re.search(r"```\n(.*?)```", ARCH_9_TEXT, re.S).group(1)
    chain = [s.strip() for s in re.sub(r"\([^()]*\)", "", block).split("→")]
    assert chain == [s.value for s in PlannerState if s is not PlannerState.FAILED]


def test_architecture_section_9_failure_tags_are_failure_tag_values():
    states = {s.value for s in PlannerState}
    tags = {t.value for t in FailureTag}
    for line in ARCH_9_TEXT.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        for token in re.findall(r"`([a-z_]+)`", cells[3]):
            assert token in tags or token in states, (cells[0], token)


def test_failure_tags_agree_in_the_code_the_schema_and_section_7():
    defs = load_schema()["$defs"]
    episode = defs["episode"]
    assert episode["properties"]["failure_tag"]["enum"] == [t.value for t in FailureTag]
    assert episode["then"]["properties"]["failure_tag"]["const"] == FailureTag.SUCCESS.value
    assert episode["else"]["properties"]["failure_tag"]["enum"] == list(FAILURE_TAG_VALUES)
    counts = defs["aggregates"]["properties"]["failure_tag_counts"]["properties"]
    assert list(counts) == list(FAILURE_TAG_VALUES)
    listed = re.search(r"logged in the JSON:\n\n(.*?)\n\n", EVAL_7, re.S).group(1)
    assert re.findall(r"`([a-z_]+)`", listed) == list(FAILURE_TAG_VALUES)


def test_success_definitions_carry_the_section_3_numbers():
    defs = metrics.SUCCESS_DEFINITIONS
    grasp = (
        f"lifted {_cm(judges.GRASP_LIFT_HEIGHT_M)} cm and held for {judges.GRASP_HOLD_S:g} s "
        "without contact with the counter"
    )
    assert grasp in defs["S2"] and grasp in EVAL_3_FLAT
    assert f"within {judges.PLACEMENT_VERTICAL_TOL_DEG:g} deg of vertical" in defs["S3"]
    assert f"stable {judges.PLACEMENT_STABLE_S:g} s" in defs["S3"]
    budget = f"within the {CFG.total_budget_s:g} s timeout"
    assert budget in defs["S4"] and budget in EVAL_3_FLAT
    assert f"under {metrics.HAND_TARGET_THRESHOLD_CM:g} cm" in defs["S1"]


def test_fake_scenario_timeout_is_the_task_budget():
    assert fake.TIMEOUT_S == CFG.total_budget_s


def test_hand_target_threshold_and_the_planner_tolerance_under_it():
    assert f"fraction under {metrics.HAND_TARGET_THRESHOLD_CM:g} cm" in EVAL_3_FLAT
    assert f"under_{metrics.HAND_TARGET_THRESHOLD_CM:g}cm" in metrics.HAND_TARGET_RATE
    # PlannerConfig.position_tolerance_m says it is under the section 3 hand-target error.
    assert CFG.position_tolerance_m * 100 < metrics.HAND_TARGET_THRESHOLD_CM


def test_wilson_interval_level_agrees_in_stats_and_the_schema():
    aggregates = load_schema()["$defs"]["aggregates"]["properties"]
    level = aggregates["ci_level"]["const"]
    assert aggregates["ci_method"]["const"] == "wilson"
    assert stats.Z_95 == pytest.approx(NormalDist().inv_cdf(0.5 + level / 2), abs=1e-3)
    episode = {"seed": 0, "success": True, "failure_tag": "success", "duration_s": 1.0,
               "metrics": {"fell": False, "grasp_success": True}}  # fmt: skip
    out = stats.aggregate_episodes("S2", [episode])
    assert (out["ci_method"], out["ci_level"]) == ("wilson", level)


# ------------------------------------------------------------------------------------------------
# Evaluator plumbing: YAMLs, extras, synthetic scenarios
# ------------------------------------------------------------------------------------------------


EVAL_YAML_VARIANTS = [(sc, sp, "") for sc in SCENARIOS for sp in SPLITS] + [
    (sc, sp, "_fixed") for sc in FIXED_BASE_SCENARIOS for sp in SPLITS
]  # the ten default-base files and the four fixed-pelvis ones (#78)


@pytest.mark.parametrize(("scenario", "split", "suffix"), EVAL_YAML_VARIANTS)
def test_eval_yaml_randomization_block_is_the_sampler_ranges(scenario, split, suffix):
    path = EVAL_CONFIGS / f"{scenario.lower()}_{split}{suffix}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert (data["scenario"], data["split"]) == (scenario, split)
    ranges = RandomizationRanges.from_dict(data["randomization"])
    assert ranges == RandomizationRanges.for_split(split)
    if scenario in dry_run.MIRRORABLE_SCENARIOS:
        assert dry_run.read_eval_yaml(path, split) == (scenario, ranges)


def test_fixed_pelvis_configs_are_the_section_1_scenarios_run_with_a_fixed_pelvis():
    """#69 B8, #78: section 1 runs S2 and S3 "fixed pelvis, then balance policy"; each has a
    committed ``base: fixed`` YAML per split that is its default-base YAML otherwise."""
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in _section("evaluation.md", "1").splitlines()
        if re.match(r"\|\s*S\d\s*\|", line)
    ]
    assert [row[0] for row in rows] == list(SCENARIOS)
    assert [row[0] for row in rows if "fixed pelvis" in row[2]] == list(FIXED_BASE_SCENARIOS)
    fixed_files = sorted(p.name for p in EVAL_CONFIGS.glob("*_fixed.yaml"))
    assert fixed_files == sorted(
        f"{sc.lower()}_{sp}_fixed.yaml" for sc in FIXED_BASE_SCENARIOS for sp in SPLITS
    )
    for scenario in FIXED_BASE_SCENARIOS:
        for split in SPLITS:
            stem = f"{scenario.lower()}_{split}"
            default = yaml.safe_load((EVAL_CONFIGS / f"{stem}.yaml").read_text(encoding="utf-8"))
            fixed = yaml.safe_load((EVAL_CONFIGS / f"{stem}_fixed.yaml").read_text("utf-8"))
            assert default["base"] == EvalConfig(scenario=scenario).base
            assert fixed["base"] == "fixed" and fixed["base"] in BASES
            assert {**fixed, "base": default["base"]} == default


@pytest.mark.parametrize("path", sorted(EVAL_CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_a_config_run_without_the_flag_uses_the_yaml_num_envs(path, tmp_path):
    """#69 B7, #78: ``evaluate.py --config`` without ``--num_envs`` runs with the YAML's value."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = tmp_path / "result.json"
    argv = ["--scenario", "fake", "--split", data["split"], "--seeds", "2", "--base",
            data["base"], "--config", str(path), "--out", str(out)]  # fmt: skip
    assert evaluate_main(argv) == 0
    run = json.loads(out.read_text(encoding="utf-8"))["run"]
    assert run["num_envs"] == data["num_envs"] == data["seeds"]["count"]
    assert data["seeds"]["count"] == len(default_seeds(data["scenario"], data["split"]))
    assert (run["base"], run["config_path"]) == (data["base"], path.as_posix())


def test_dry_run_extra_keys_are_registered_and_are_not_eval_config_fields():
    assert get_scenario("dry").extra_keys == dry_run.EXTRA_KEYS
    run_fields = {f.name for f in dataclasses.fields(EvalConfig)}
    assert run_fields.isdisjoint(dry_run.EXTRA_KEYS)
    assert set(EvalConfig(scenario="dry").to_dict()).isdisjoint(dry_run.EXTRA_KEYS)
    # Extras are not run fields, so they reach run.config_hash only through resolved_config().
    gate_on = dry_run.DryRunBackend(EvalConfig(scenario="dry")).resolved_config()
    off = dry_run.DryRunBackend(EvalConfig(scenario="dry", extras={"reach_gate": False}))
    assert off.resolved_config() != gate_on


def test_synthetic_scenarios_agree_in_the_catalogue_the_registry_and_the_schema():
    synthetic_rule = load_schema()["$defs"]["run"]["allOf"][0]
    assert synthetic_rule["if"]["properties"]["scenario"]["enum"] == list(
        metrics.SYNTHETIC_SCENARIOS
    )
    for name in scenario_names():
        spec = get_scenario(name)
        assert spec.synthetic == (name in metrics.SYNTHETIC_SCENARIOS), name
        assert spec.uses_gpu == (name not in metrics.SYNTHETIC_SCENARIOS), name
    for split in SPLITS:  # dry mirrors S4, trial counts included
        assert default_seeds("dry", split) == default_seeds("S4", split)
    assert metrics.metrics_for("dry") == metrics.metrics_for("S4")


# ------------------------------------------------------------------------------------------------
# Scene: rack layout, tines file, frames, quaternions
# ------------------------------------------------------------------------------------------------

_CAPSULE = re.compile(
    r'def Capsule "Tine_(neg|pos)_\d+".*?double height = ([-0-9.e]+).*?'
    r"double radius = ([-0-9.e]+).*?xformOp:translate = \(([^)]*)\)",
    re.S,
)
_SLOT_XFORM = re.compile(r'def Xform "Slot_(\d+)"\s*\{\s*double3 xformOp:translate = \(([^)]*)\)')


def _xyz(text: str) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in text.split(","))
    return x, y, z


def test_rack_tines_file_has_the_slot_count_pitch_and_entry_height_of_the_layout():
    layout = kd.RackSlotLayout()
    text = RACK_TINES_USDA.read_text(encoding="utf-8")
    tines = [(row, float(h), float(r), _xyz(t)) for row, h, r, t in _CAPSULE.findall(text)]
    assert len(tines) == 2 * (kd.RACK_SLOTS + 1)
    for row, row_y in zip(("neg", "pos"), layout.row_ys, strict=True):
        xs = sorted(p[0] for name, _, _, p in tines if name == row)
        assert xs == pytest.approx(layout.tine_xs, abs=1e-6)
        assert all(p[1] == pytest.approx(row_y, abs=1e-6) for name, _, _, p in tines if name == row)
    for _, height, radius, (_, _, z) in tines:
        assert radius == pytest.approx(layout.tine_radius, abs=1e-6)
        assert z + height / 2 + radius == pytest.approx(layout.slot_entry_z, abs=1e-6)
    slots = {int(i): _xyz(t) for i, t in _SLOT_XFORM.findall(text)}
    assert sorted(slots) == list(range(1, kd.RACK_SLOTS + 1))
    for i, xyz in slots.items():
        assert xyz == pytest.approx(layout.slot_pose(i)[:3], abs=1e-6)
    xs = [slots[i][0] for i in sorted(slots)]
    assert np.diff(xs) == pytest.approx([kd.RACK_SLOT_PITCH_M] * (kd.RACK_SLOTS - 1), abs=1e-6)


def test_the_dry_run_and_the_sampler_use_the_layout_the_scene_places():
    layout = kd.RackSlotLayout()
    assert layout == df.slot_layout(kd.RACK_SCALE) == SpawnDims().rack
    backend = dry_run.DryRunBackend(EvalConfig(scenario="dry", extras={"reach_gate": False}))
    assert backend.layout == layout
    kitchen = df.slot_poses_kitchen()
    assert len(kitchen) == layout.slots
    steps = [math.dist(a[:3], b[:3]) for a, b in zip(kitchen, kitchen[1:], strict=False)]
    assert steps == pytest.approx([layout.pitch] * (layout.slots - 1))


def test_fakes_stand_off_and_pelvis_height_are_the_scene_numbers():
    assert fakes.PELVIS_HEIGHT_M == df.PELVIS_HEIGHT_M
    defaults = {
        "fakes.pelvis_pose": (fakes.pelvis_pose, "stand_off_m", "height_m"),
        "Fakes.from_episode": (fakes.Fakes.from_episode, "stand_off_m", None),
        "dry_run.simulate_episode": (dry_run.simulate_episode, "stand_off_m", None),
        "df.kitchen_to_pelvis": (df.kitchen_to_pelvis, "standoff", "pelvis_height"),
        "df.slot_row_from_pelvis": (df.slot_row_from_pelvis, "standoff", "pelvis_height"),
    }
    for name, (func, standoff, height) in defaults.items():
        params = inspect.signature(func).parameters
        assert params[standoff].default == kd.ROBOT_STANDOFF_M, name
        if height is not None:
            assert params[height].default == df.PELVIS_HEIGHT_M, name
    # The stand environment draft trains the base height to the same number (parsed, not imported).
    module = ast.parse((PKG / "locomotion" / "stand_env_cfg.py").read_text(encoding="utf-8"))
    targets = [
        ast.literal_eval(value)
        for node in ast.walk(module)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "target_height"
    ]
    assert targets == [df.PELVIS_HEIGHT_M]


def test_the_fakes_pelvis_frame_is_the_kitchen_to_pelvis_of_dishwasher_frames():
    spec = sample_episode(0, "seen", "S4")
    nominal = dataclasses.replace(spec, pelvis_offset_m=(0.0, 0.0), pelvis_yaw_offset_deg=0.0)
    pelvis = fakes.pelvis_pose(nominal)
    for point in ((0.35, -0.26, 0.65), (0.10, 0.30, 0.90), (-0.20, 0.50, 1.20)):
        assert to_frame(Pose(point), pelvis).position == pytest.approx(df.kitchen_to_pelvis(point))


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2.0, -math.pi / 2.0, 2.0])
def test_isaac_lab_quaternions_are_the_repository_ones_reordered(yaw):
    assert df.yaw_quat_xyzw(yaw) == pytest.approx(tuple(np.roll(quat_from_yaw(yaw), -1)))


# ------------------------------------------------------------------------------------------------
# Controllers: slew, attach, close, control rate; the training config against the protocol
# ------------------------------------------------------------------------------------------------


def test_slew_limits_agree_in_the_fakes_the_filter_and_section_7():
    assert fakes.ARM_LINEAR_SPEED_MPS == target_filter.MAX_LIN_SPEED_MPS
    assert fakes.ARM_ANGULAR_SPEED_RPS == target_filter.MAX_ANG_SPEED_RPS
    pipeline = inspect.signature(target_filter.ArmTargetPipeline).parameters
    assert pipeline["max_lin_speed"].default == target_filter.MAX_LIN_SPEED_MPS
    assert pipeline["max_ang_speed"].default == target_filter.MAX_ANG_SPEED_RPS
    lin, ang = target_filter.MAX_LIN_SPEED_MPS, target_filter.MAX_ANG_SPEED_RPS
    assert f"at most {lin:g} m/s and {ang:g} rad/s" in ARCH_7
    assert f"({lin:g} m/s, {ang:g} rad/s slew)" in ARCH_9_FLAT


def test_hand_numbers_agree_in_the_fakes_the_controllers_and_section_8():
    assert fakes.ATTACH_RADIUS_M == kinematic_follow.DEFAULT_ATTACH_DISTANCE_M
    assert f"within {_cm(fakes.ATTACH_RADIUS_M)} cm of the rim" in ARCH_8
    dex3 = inspect.signature(Dex3SynergyHand).parameters
    mock = inspect.signature(MockHand).parameters
    assert (
        fakes.CLOSE_DURATION_S
        == fakes.FakeHand().close_duration
        == dex3["close_duration"].default
        == mock["close_duration"].default
    )
    assert f"over {fakes.CLOSE_DURATION_S:g} s" in ARCH_8
    assert dex3["min_contacts"].default == 2 and "two or more fingertips" in ARCH_8


def test_control_rate_agrees_in_the_dry_run_the_training_config_and_section_5():
    timing = yaml.safe_load(TRAIN_YAML.read_text(encoding="utf-8"))["timing"]
    policy_hz = 1.0 / (timing["physics_dt_s"] * timing["decimation"])
    assert dry_run.CONTROL_HZ == pytest.approx(policy_hz)
    assert inspect.signature(fakes.run_episode).parameters["hz"].default == dry_run.CONTROL_HZ
    physics_hz = 1.0 / timing["physics_dt_s"]
    assert f"{dry_run.CONTROL_HZ:g} Hz policy, {physics_hz:g} Hz physics" in ARCH_5


def test_training_config_randomization_and_fall_tilt_match_the_protocol():
    cfg = yaml.safe_load(TRAIN_YAML.read_text(encoding="utf-8"))
    seen = RandomizationRanges.for_split("seen")
    lo, hi = seen.friction_range
    assert tuple(cfg["randomization"]["friction"]) == seen.friction_range
    assert f"uniform in [{lo}, {hi}]" in EVAL_2 and f"friction {lo} to {hi}" in ARCH_6
    assert cfg["randomization"]["push_interval_s"] == seen.push_interval_s
    assert f"every {seen.push_interval_s:g} s" in EVAL_2
    tilt = cfg["terminations"]["max_tilt_deg"]
    assert f"torso tilt exceeds {tilt:g} deg" in EVAL_3_FLAT
    assert f"tilt over {tilt:g} degrees" in ARCH_6
    module = ast.parse((PKG / "locomotion" / "stand_env_cfg.py").read_text(encoding="utf-8"))
    limits = [
        value
        for node in ast.walk(module)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "limit_angle"
    ]
    assert len(limits) == 1 and isinstance(limits[0], ast.Call)
    assert ast.unparse(limits[0].func) == "math.radians"
    assert ast.literal_eval(limits[0].args[0]) == tilt


def test_the_waist_lean_maximum_is_one_constant_in_the_planner_the_training_and_the_envelope():
    """D012 as amended 2026-10-24 (#69 E8, #77): the soft limit of ``waist_pitch_joint``."""
    upper = WAIST_LIMITS["waist_pitch_joint"].upper
    assert WAIST_LEAN_MAX_RAD == round(SOFT_LIMIT_FRACTION * upper, 3) < upper
    # the planner's lean (#67) and the training command (#68)
    assert CFG.waist_lean_rad == waist_lean_targets.MAX_LEAN_RAD == WAIST_LEAN_MAX_RAD
    lean = yaml.safe_load(TRAIN_YAML.read_text(encoding="utf-8"))["commands"]["waist_lean"]
    assert lean["max_lean_rad"] == WAIST_LEAN_MAX_RAD
    # the draft command term's default is the numpy constant (parsed, not imported)
    module = ast.parse((PKG / "locomotion" / "commands.py").read_text(encoding="utf-8"))
    cfg_class = next(
        n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "WaistLeanCommandCfg"
    )
    defaults = {
        n.target.id: ast.unparse(n.value)
        for n in cfg_class.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None
    }
    assert defaults["max_lean_rad"] == "wlt.MAX_LEAN_RAD"
    # the waist envelope's pitch range, in the code and in the committed file the dry run gates on
    pitch = (0.0, math.degrees(WAIST_LEAN_MAX_RAD))
    assert reachability.WAIST_ENVELOPE_RANGE_DEG[2] == pitch
    assert reachability.ReachabilityEnvelope.load("waist").waist_range_deg[2] == pitch
    assert f"forward waist pitch in [0, {WAIST_LEAN_MAX_RAD:g}] rad" in ARCH_6
