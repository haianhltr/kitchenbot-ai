"""``scripts/evaluate.py --extra KEY=VALUE`` fills ``EvalConfig.extras`` (#60)."""

import re
import subprocess
import sys

import pytest
from eval_helpers import REPO

from kitchenbot.evaluation import dry_run
from kitchenbot.evaluation.cli import build_parser, parse_extras
from kitchenbot.evaluation.cli import main as evaluate_main
from kitchenbot.evaluation.results import load_result
from kitchenbot.evaluation.runner import (
    EvalConfig,
    ScenarioConfigError,
    get_scenario,
    run_evaluation,
)

CONFIG = REPO / "configs" / "eval" / "s4_seen.yaml"


def _run(tmp_path, *extra_args, scenario="dry", seeds="0-4"):
    argv = ["--scenario", scenario, "--seeds", seeds, "--out", str(tmp_path), *extra_args]
    if scenario == "dry":
        argv += ["--config", str(CONFIG)]
    return evaluate_main(argv)


# ----------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("item", "value"),
    [
        ("reach_gate=false", False),
        ("reach_gate=False", False),
        ("reach_gate=true", True),
        ("x=0.3", 0.3),
        ("x=7", 7),
        ("x=hello", "hello"),
        ("x=two words", "two words"),
        ("x='false'", "false"),
        ("x=a=b", "a=b"),
        ("x=", None),
        ("x=null", None),
        (" x =1", 1),
    ],
)
def test_values_are_yaml_scalars(item, value):
    ((key, parsed),) = parse_extras([item]).items()
    assert key == "x" or key == "reach_gate"
    assert parsed == value and type(parsed) is type(value)


def test_the_flag_is_repeatable_and_keeps_the_order():
    assert list(parse_extras(["b=1", "a=false", "c=text"]).items()) == [
        ("b", 1),
        ("a", False),
        ("c", "text"),
    ]
    assert parse_extras([]) == {}
    args = build_parser().parse_args(["--scenario", "dry", "--extra", "a=1", "--extra", "b=2"])
    assert args.extra == ["a=1", "b=2"]
    assert build_parser().parse_args(["--scenario", "dry"]).extra == []


@pytest.mark.parametrize(
    ("items", "message"),
    [
        (["novalue"], "KEY=VALUE"),
        (["=1"], "KEY=VALUE"),
        (["a=1", "a=2"], "a given twice"),
        (["a=[unclosed"], "YAML"),
    ],
)
def test_bad_items_are_value_errors(items, message):
    with pytest.raises(ValueError, match=message):
        parse_extras(items)


# ----------------------------------------------------------------------------- through the CLI


def test_extra_reach_gate_false_runs_the_dry_scenario_without_the_gate(tmp_path, capsys):
    assert _run(tmp_path, "--extra", "reach_gate=false") == 0
    out = capsys.readouterr().out
    assert "success: 5/5 = 100.0%" in out and "ik_infeasible 0" in out
    assert "extras reach_gate=False" in out
    off = load_result(next(tmp_path.glob("*_dry_seen.json"))).doc

    other = tmp_path / "on"
    assert _run(other) == 0  # the default is the gate on
    out = capsys.readouterr().out
    # seeds 0 to 4 with the default waist0 gate at the #67 geometry: 1 and 3 succeed
    assert "success: 2/5 = 40.0%" in out and "ik_infeasible 3" in out and "extras" not in out
    on = load_result(next(other.glob("*_dry_seen.json"))).doc

    assert off["run"]["config_hash"] != on["run"]["config_hash"]  # the option is hashed
    differing = {k for k in off["run"] if off["run"][k] != on["run"][k]}
    assert differing <= {"config_hash", "started_at", "wall_clock_s"}  # nothing else in `run`
    assert set(off["run"]) == set(on["run"])


def test_extra_true_is_the_default(tmp_path):
    assert _run(tmp_path, "--extra", "reach_gate=true") == 0
    explicit = load_result(next(tmp_path.glob("*_dry_seen.json"))).doc
    default_dir = tmp_path / "default"
    assert _run(default_dir) == 0
    default = load_result(next(default_dir.glob("*_dry_seen.json"))).doc
    assert explicit["run"]["config_hash"] == default["run"]["config_hash"]
    assert explicit["episodes"] == default["episodes"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--extra", "reach_gate=maybe"], "must be a bool, got 'maybe'"),
        (
            ["--extra", "speed=1"],
            "unknown extras \\['speed'\\].*reads \\['reach_gate', 'gate_envelope'\\]",
        ),
        (["--extra", "reach_gate=false", "--extra", "speed=1"], "unknown extras"),
        (["--extra", "gate_envelope=waist30"], "gate_envelope'\\] must be one of"),
    ],
)
def test_a_value_or_key_the_scenario_rejects_is_a_usage_error(tmp_path, capsys, args, message):
    assert _run(tmp_path, *args) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert re.search(message, err), err
    assert list(tmp_path.glob("*.json")) == []


def test_a_scenario_without_extras_rejects_them(tmp_path, capsys):
    assert _run(tmp_path, "--extra", "reach_gate=false", scenario="fake") == 2
    err = capsys.readouterr().err
    assert "scenario fake takes no extras" in err and list(tmp_path.glob("*.json")) == []


@pytest.mark.parametrize("item", ["novalue", "=1", "a=[unclosed"])
def test_a_malformed_item_is_an_argparse_error(tmp_path, capsys, item):
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, "--extra", item)
    assert exc.value.code == 2
    assert "--extra" in capsys.readouterr().err


def test_the_script_takes_the_flag(tmp_path):
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "evaluate.py"),
        "--scenario",
        "dry",
        "--config",
        str(CONFIG),
        "--extra",
        "reach_gate=false",
        "--seeds",
        "0-2",
        "--out",
        str(tmp_path),
    ]
    done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True)
    assert "success: 3/3 = 100.0%" in done.stdout and "extras reach_gate=False" in done.stdout


# ----------------------------------------------------------------------------- the registry


def test_the_dry_scenario_lists_the_extras_it_reads():
    assert get_scenario("dry").extra_keys == dry_run.EXTRA_KEYS == ("reach_gate", "gate_envelope")
    assert get_scenario("fake").extra_keys == () and get_scenario("S4").extra_keys == ()


def test_run_evaluation_checks_extras_before_loading():
    config = EvalConfig(scenario="fake", seeds=(0,), extras={"anything": 1})
    with pytest.raises(ScenarioConfigError, match="scenario fake takes no extras"):
        run_evaluation(config)
    # a placeholder scenario is rejected for its extras, not for being unimplemented
    with pytest.raises(ScenarioConfigError, match="takes no extras"):
        run_evaluation(EvalConfig(scenario="S4", seeds=(0,), extras={"x": 1}))
