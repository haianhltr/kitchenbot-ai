"""The evaluator's config hook (#52): ``--config`` records the file's SHA256 in the result.

``EvalConfig.config_path`` / ``scripts/evaluate.py --config`` record the file's path and SHA256 in
``run.config_path`` and ``run.config_sha256``; nothing else about a run changes. The result schema
is strict, so the two optional fields come with ``schema_version`` 2.

Since #78 the command line also takes the file's ``num_envs`` when ``--num_envs`` is not given;
the flag overrides it and ``run.num_envs`` records the value used.
"""

import copy
import hashlib
import json

import pytest
from eval_helpers import REPO

from kitchenbot.evaluation import cli, runner
from kitchenbot.evaluation.cli import build_parser, format_summary, resolve_num_envs, yaml_num_envs
from kitchenbot.evaluation.cli import main as evaluate_main
from kitchenbot.evaluation.runner import EvalConfig, file_sha256, run_evaluation, write_result
from kitchenbot.evaluation.schema import load_schema, validate_result
from kitchenbot.evaluation.seeds import SPLITS, default_seeds
from kitchenbot.scene.randomization import SCENARIOS

CONFIG_DIR = REPO / "configs" / "eval"
BOTH = [(sc, sp) for sc in SCENARIOS for sp in SPLITS]


# ------------------------------------------------------------------ the evaluator hook


def _fake_config(**kwargs):
    return EvalConfig(scenario="fake", split="seen", seeds=tuple(range(10)), num_envs=4, **kwargs)


def test_without_a_config_path_the_result_has_no_config_fields():
    run = run_evaluation(_fake_config())["run"]
    assert "config_path" not in run and "config_sha256" not in run


def test_with_a_config_path_the_result_records_its_sha256():
    path = CONFIG_DIR / "s4_seen.yaml"
    doc = run_evaluation(_fake_config(config_path=path))
    validate_result(doc)
    run = doc["run"]
    assert run["config_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert run["config_sha256"] == file_sha256(path)
    assert run["config_path"] == path.as_posix()


def test_only_the_two_fields_change_and_config_hash_is_unaffected():
    plain = run_evaluation(_fake_config())
    with_config = run_evaluation(_fake_config(config_path=CONFIG_DIR / "s4_seen.yaml"))
    assert with_config["run"]["config_hash"] == plain["run"]["config_hash"]
    assert with_config["episodes"] == plain["episodes"]
    assert with_config["aggregates"] == plain["aggregates"]
    differing = {k for k in with_config["run"] if with_config["run"][k] != plain["run"].get(k)}
    assert differing - {"started_at", "wall_clock_s"} == {"config_path", "config_sha256"}
    assert "config_path" not in _fake_config(config_path=CONFIG_DIR / "s4_seen.yaml").to_dict()


def test_the_hash_follows_the_file_content(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_bytes(b"physics: isaacsim_physx\n")
    b.write_bytes(b"physics: isaacsim_physx\nnum_envs: 3\n")
    ha = run_evaluation(_fake_config(config_path=a))["run"]["config_sha256"]
    hb = run_evaluation(_fake_config(config_path=b))["run"]["config_sha256"]
    assert ha != hb and ha == hashlib.sha256(b"physics: isaacsim_physx\n").hexdigest()


def test_config_path_is_normalised_and_must_exist(tmp_path):
    cfg = _fake_config(config_path=str(CONFIG_DIR / "s0_seen.yaml"))
    assert cfg.config_path == CONFIG_DIR / "s0_seen.yaml"
    with pytest.raises(ValueError, match="not a file"):
        _fake_config(config_path=tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="not a file"):
        _fake_config(config_path=tmp_path)  # a directory


@pytest.mark.parametrize(("scenario", "split"), BOTH)
def test_every_committed_config_can_be_hashed_by_a_fake_run(scenario, split):
    path = CONFIG_DIR / f"{scenario.lower()}_{split}.yaml"
    seeds = tuple(default_seeds("fake", split)[:3])
    doc = run_evaluation(
        EvalConfig(scenario="fake", split=split, seeds=seeds, config_path=path)
    )
    assert doc["run"]["config_sha256"] == file_sha256(path)


# ------------------------------------------------------------------ the schema


def test_schema_version_is_one_number_in_the_code_and_the_schema():
    assert runner.SCHEMA_VERSION == 2
    assert load_schema()["properties"]["schema_version"] == {"const": runner.SCHEMA_VERSION}
    assert run_evaluation(_fake_config())["schema_version"] == runner.SCHEMA_VERSION


def test_schema_describes_the_two_optional_fields():
    run = load_schema()["$defs"]["run"]
    assert {"config_path", "config_sha256"} <= set(run["properties"])
    assert not {"config_path", "config_sha256"} & set(run["required"])
    assert run["additionalProperties"] is False  # still strict about everything else


def _valid_with_config():
    return run_evaluation(_fake_config(config_path=CONFIG_DIR / "s4_seen.yaml"))


def test_schema_accepts_both_fields_or_neither():
    doc = _valid_with_config()
    validate_result(doc)
    bare = copy.deepcopy(doc)
    del bare["run"]["config_path"], bare["run"]["config_sha256"]
    validate_result(bare)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda run: run.pop("config_sha256"),
        lambda run: run.pop("config_path"),
        lambda run: run.update(config_sha256="xyz"),
        lambda run: run.update(config_sha256="A" * 64),
        lambda run: run.update(config_path=""),
        lambda run: run.update(config_path=3),
        lambda run: run.update(config_hash_of_yaml="a" * 64),
    ],
    ids=[
        "path without hash",
        "hash without path",
        "short hash",
        "upper-case hash",
        "empty path",
        "path not a string",
        "unknown run field",
    ],
)
def test_schema_rejects_bad_config_fields(mutate):
    from jsonschema.exceptions import ValidationError

    doc = _valid_with_config()
    mutate(doc["run"])
    with pytest.raises(ValidationError):
        validate_result(doc)


# ------------------------------------------------------------------ the command line


def _evaluate(tmp_path, *extra):
    out = tmp_path / "result.json"
    argv = ["--scenario", "fake", "--seeds", "0-9", "--num_envs", "4", "--headless"]
    code = evaluate_main([*argv, "--out", str(out), *extra])
    return code, out


def test_cli_without_config_writes_no_hash(tmp_path, capsys):
    code, out = _evaluate(tmp_path)
    assert code == 0
    doc = json.loads(out.read_text("utf-8"))
    validate_result(doc)
    assert doc["schema_version"] == 2
    assert "config_sha256" not in doc["run"] and "sha256" not in capsys.readouterr().out


def test_cli_with_config_records_path_and_hash_and_prints_them(tmp_path, capsys):
    path = CONFIG_DIR / "s4_seen.yaml"
    code, out = _evaluate(tmp_path, "--config", str(path))
    assert code == 0
    doc = json.loads(out.read_text("utf-8"))
    validate_result(doc)
    assert doc["run"]["config_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert doc["run"]["config_path"] == path.as_posix()
    printed = capsys.readouterr().out
    assert f"sha256 {doc['run']['config_sha256'][:12]}" in printed
    assert "config " in printed and path.name in printed


def test_cli_rejects_a_missing_config_with_status_2(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        _evaluate(tmp_path, "--config", str(tmp_path / "nope.yaml"))
    assert exc.value.code == 2
    assert "not a file" in capsys.readouterr().err
    assert not (tmp_path / "result.json").exists()


def test_written_result_keeps_the_hash_after_a_round_trip(tmp_path):
    doc = _valid_with_config()
    path = write_result(doc, tmp_path / "r.json")
    back = json.loads(path.read_text("utf-8"))
    assert back == doc
    validate_result(back)
    assert f"sha256 {doc['run']['config_sha256'][:12]}" in format_summary(back)


# ------------------------------------------------------------------ num_envs from the YAML (#78)


def _run_without_num_envs(tmp_path, *extra):
    """A fake run with ``extra`` arguments and no ``--num_envs``; returns (status, run block)."""
    out = tmp_path / "result.json"
    argv = ["--scenario", "fake", "--seeds", "0-9", "--headless", "--out", str(out), *extra]
    code = evaluate_main(argv)
    return code, (json.loads(out.read_text("utf-8"))["run"] if out.exists() else None)


def test_without_the_flag_a_config_run_uses_the_yaml_num_envs(tmp_path, capsys):
    path = CONFIG_DIR / "s2_seen.yaml"
    code, run = _run_without_num_envs(tmp_path, "--config", str(path))
    assert code == 0
    assert run["num_envs"] == 100 == yaml_num_envs(path)  # the S2 seen trial count
    assert run["config_sha256"] == file_sha256(path)
    assert "num_envs 100" in capsys.readouterr().out


def test_the_flag_overrides_the_yaml_num_envs(tmp_path):
    out = tmp_path / "result.json"
    argv = ["--scenario", "fake", "--seeds", "0-9", "--num_envs", "7", "--out", str(out)]
    assert evaluate_main([*argv, "--config", str(CONFIG_DIR / "s2_seen.yaml")]) == 0
    assert json.loads(out.read_text("utf-8"))["run"]["num_envs"] == 7


def test_without_the_flag_or_a_config_the_default_is_used(tmp_path):
    code, run = _run_without_num_envs(tmp_path)
    assert code == 0
    assert run["num_envs"] == runner.DEFAULT_NUM_ENVS == EvalConfig(scenario="fake").num_envs
    assert build_parser().parse_args(["--scenario", "fake"]).num_envs is None


def test_a_yaml_without_num_envs_falls_back_to_the_default(tmp_path):
    path = tmp_path / "scratch.yaml"
    path.write_text("physics: isaacsim_physx\n", encoding="utf-8")
    assert yaml_num_envs(path) is None
    code, run = _run_without_num_envs(tmp_path, "--config", str(path))
    assert code == 0 and run["num_envs"] == runner.DEFAULT_NUM_ENVS


def test_a_real_scenario_gets_the_yaml_num_envs_too(monkeypatch, tmp_path):
    """S0 to S4 are placeholders today; the config they would run with carries the YAML's value."""
    seen = []

    def capture(config, **_):
        seen.append(config)
        raise NotImplementedError("placeholder")

    monkeypatch.setattr(cli, "run_evaluation", capture)
    argv = ["--scenario", "S3", "--split", "unseen", "--out", str(tmp_path)]
    assert evaluate_main([*argv, "--config", str(CONFIG_DIR / "s3_unseen.yaml")]) == 3
    assert seen[0].num_envs == 50 and seen[0].config_path == CONFIG_DIR / "s3_unseen.yaml"
    assert evaluate_main(argv) == 3  # no config: the default
    assert seen[1].num_envs == runner.DEFAULT_NUM_ENVS


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("num_envs: 0\n", "num_envs must be a positive integer, got 0"),
        ("num_envs: many\n", "num_envs must be a positive integer, got 'many'"),
        ("num_envs: true\n", "num_envs must be a positive integer, got True"),
        ("num_envs: 2.5\n", "num_envs must be a positive integer, got 2.5"),
        ("- num_envs: 4\n", "expected a mapping at the top level"),
        ("num_envs: [\n", "cannot read as YAML"),
    ],
    ids=["zero", "text", "bool", "float", "list", "broken"],
)
def test_a_bad_yaml_num_envs_is_a_usage_error_unless_the_flag_is_given(
    tmp_path, capsys, text, message
):
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _run_without_num_envs(tmp_path, "--config", str(path))
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "result.json").exists()
    # With the flag the file is hashed, not read, as before #78.
    assert resolve_num_envs(3, path) == 3
    code, _ = _evaluate(tmp_path, "--config", str(path))
    assert code == 0
