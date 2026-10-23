"""scripts/evaluate.py end to end, through main(argv) and as a subprocess."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from kitchenbot.evaluation.cli import EXIT_NOT_IMPLEMENTED, HELD_OUT_NOTICE, build_parser, main
from kitchenbot.evaluation.schema import validate_result

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "evaluate.py"
NAME = re.compile(r"^\d{8}T\d{6}Z_([0-9a-f]{7}|nogit)_fake_seen\.json$")


def _only_json(directory: Path) -> dict:
    files = list(directory.glob("*.json"))
    assert len(files) == 1, files
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_fake_writes_a_schema_valid_json_into_a_directory(tmp_path, capsys):
    assert main(["--scenario", "fake", "--out", str(tmp_path)]) == 0
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1 and NAME.match(files[0].name), files
    doc = json.loads(files[0].read_text(encoding="utf-8"))
    validate_result(doc)
    assert doc["run"]["scenario"] == "fake" and doc["run"]["synthetic"] is True
    assert doc["run"]["split"] == "seen" and doc["run"]["seeds"] == list(range(100))
    assert doc["aggregates"]["num_episodes"] == 100
    out = capsys.readouterr().out
    assert "SYNTHETIC RUN" in out and "success: 78/100" in out and "95% Wilson" in out
    assert files[0].as_posix() in out
    assert HELD_OUT_NOTICE not in out


def test_split_defaults_to_seen():
    assert build_parser().parse_args(["--scenario", "fake"]).split == "seen"


def test_unseen_split_prints_the_held_out_notice_before_the_summary(tmp_path, capsys):
    argv = ["--scenario", "fake", "--split", "unseen", "--out", str(tmp_path)]
    assert main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == HELD_OUT_NOTICE == "held-out split: report only, do not tune on these seeds"
    assert lines[1].startswith("scenario fake | split unseen")
    assert lines.count(HELD_OUT_NOTICE) == 1
    doc = _only_json(tmp_path)
    assert doc["run"]["split"] == "unseen" and doc["aggregates"]["num_episodes"] == 50
    assert "_fake_unseen.json" in next(tmp_path.glob("*.json")).name


def test_explicit_file_and_options_are_recorded(tmp_path):
    target = tmp_path / "deep" / "result.json"
    argv = [
        "--scenario",
        "fake",
        "--split",
        "seen",
        "--seeds",
        "12",
        "--num_envs",
        "5",
        "--headless",
        "--policy",
        "agile",
        "--hand",
        "dex1",
        "--base",
        "fixed",
        "--out",
        str(target),
    ]
    assert main(argv) == 0
    doc = json.loads(target.read_text(encoding="utf-8"))
    validate_result(doc)
    run = doc["run"]
    assert run["seeds"] == list(range(12))
    assert (run["split"], run["num_envs"], run["headless"]) == ("seen", 5, True)
    assert (run["policy"], run["hand"], run["base"]) == ("agile", "dex1", "fixed")
    assert len(doc["episodes"]) == 12


@pytest.mark.parametrize(
    ("seeds", "split", "expected"),
    [
        ("1000-1009", "unseen", list(range(1000, 1010))),
        ("1001,1003,1010-1011", "unseen", [1001, 1003, 1010, 1011]),
        ("3", "unseen", [1000, 1001, 1002]),
        ("7-7", "seen", [7]),
    ],
)
def test_seed_forms(tmp_path, seeds, split, expected):
    assert (
        main(["--scenario", "fake", "--split", split, "--seeds", seeds, "--out", str(tmp_path)])
        == 0
    )
    assert _only_json(tmp_path)["run"]["seeds"] == expected


def test_default_output_directory_is_outputs_eval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["--scenario", "fake", "--seeds", "3"]) == 0
    files = list((tmp_path / "outputs" / "eval").glob("*.json"))
    assert len(files) == 1 and NAME.match(files[0].name)


def test_unimplemented_scenario_exits_3_and_writes_nothing(tmp_path, capsys):
    assert main(["--scenario", "S4", "--out", str(tmp_path)]) == EXIT_NOT_IMPLEMENTED
    assert "#22" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--scenario", "S9"],
        ["--scenario", "fake", "--split", "val"],
        ["--scenario", "fake", "--policy", "random"],
        ["--scenario", "fake", "--hand", "claw"],
        ["--scenario", "fake", "--base", "wheeled"],
        ["--scenario", "fake", "--seeds", "abc"],
        ["--scenario", "fake", "--seeds", "1000,1001"],  # unseen seeds on the default seen split
        ["--scenario", "fake", "--seeds", "101"],
        ["--scenario", "fake", "--split", "unseen", "--seeds", "0,1"],
        ["--scenario", "fake", "--split", "unseen", "--seeds", "51"],
        ["--scenario", "fake", "--num_envs", "0"],
        ["--scenario", "fake", "--num_envs", "many"],
    ],
)
def test_bad_arguments_exit_2(argv, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main([*argv, "--out", str(tmp_path)])
    assert exc.value.code == 2
    assert not list(tmp_path.glob("*.json"))
    capsys.readouterr()


def test_script_runs_as_a_subprocess(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--scenario",
            "fake",
            "--seeds",
            "10",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    assert "wrote " in proc.stdout
    doc = _only_json(tmp_path)
    validate_result(doc)
    assert len(doc["episodes"]) == 10


def test_script_exposes_main_argv(tmp_path):
    spec = importlib.util.spec_from_file_location("evaluate_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--scenario", "fake", "--seeds", "2", "--out", str(tmp_path)]) == 0
    assert len(_only_json(tmp_path)["episodes"]) == 2


def test_script_help_lists_the_flags():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=True
    )
    for flag in (
        "--scenario",
        "--split",
        "--seeds",
        "--num_envs",
        "--headless",
        "--policy",
        "--hand",
        "--base",
        "--out",
    ):
        assert flag in proc.stdout
