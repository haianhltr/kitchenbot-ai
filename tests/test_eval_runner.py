"""Runner: registry, the fake scenario, run metadata, and the lazy-import guarantee."""

import hashlib
import json
import subprocess
import sys

import pytest

from kitchenbot.evaluation import fake, runner
from kitchenbot.evaluation.failure_tags import FAILURE_TAG_VALUES, FailureTag
from kitchenbot.evaluation.metrics import SCENARIO_NAMES, metrics_for
from kitchenbot.evaluation.runner import (
    EpisodeResult,
    EvalConfig,
    ScenarioSpec,
    config_hash,
    default_output_path,
    detect_gpu_name,
    get_scenario,
    git_info,
    register_scenario,
    run_evaluation,
    scenario_names,
    write_result,
)
from kitchenbot.evaluation.schema import validate_result
from kitchenbot.evaluation.seeds import split_seeds

# ----------------------------------------------------------------------------- registry


def test_registry_has_s0_to_s4_and_fake():
    assert scenario_names() == ["S0", "S1", "S2", "S3", "S4", "fake"]
    assert scenario_names() == list(SCENARIO_NAMES)


@pytest.mark.parametrize(
    ("name", "issue"), [("S0", 5), ("S1", 14), ("S2", 17), ("S3", 21), ("S4", 22)]
)
def test_real_scenarios_are_placeholders_naming_their_issue(name, issue):
    spec = get_scenario(name)
    assert not spec.synthetic and spec.uses_gpu
    with pytest.raises(NotImplementedError, match=f"#{issue}\\b") as err:
        spec.loader(EvalConfig(scenario=name))
    assert name in str(err.value)
    with pytest.raises(NotImplementedError):
        run_evaluation(EvalConfig(scenario=name))


def test_registering_twice_needs_replace(monkeypatch):
    monkeypatch.setattr(runner, "_REGISTRY", dict(runner._REGISTRY))
    spec = get_scenario("fake")
    with pytest.raises(ValueError, match="already registered"):
        register_scenario(spec)
    register_scenario(spec, replace=True)
    with pytest.raises(ValueError, match="unknown scenario"):
        register_scenario(ScenarioSpec(name="S9", title="x", loader=spec.loader))
    with pytest.raises(ValueError):
        get_scenario("S9")


def test_importing_the_runner_pulls_in_no_simulator():
    code = (
        "import sys\n"
        "import kitchenbot.evaluation.cli, kitchenbot.evaluation.runner\n"
        "import kitchenbot.evaluation.fake, kitchenbot.evaluation.schema\n"
        "banned = {'isaaclab', 'isaacsim', 'omni', 'torch', 'carb', 'warp'}\n"
        "hits = sorted(m for m in sys.modules if m.split('.')[0] in banned)\n"
        "assert not hits, hits\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ----------------------------------------------------------------------------- fake scenario


def _strip_run(doc):
    return {"episodes": doc["episodes"], "aggregates": doc["aggregates"]}


def test_fake_run_is_schema_valid_and_marked_synthetic():
    doc = run_evaluation(EvalConfig(scenario="fake"))
    validate_result(doc)
    run = doc["run"]
    assert run["synthetic"] is True and run["gpu_name"] == "none"
    assert run["scenario"] == "fake" and run["split"] == "seen"
    assert run["seeds"] == split_seeds("seen")
    assert [e["seed"] for e in doc["episodes"]] == run["seeds"]
    assert doc["aggregates"]["num_episodes"] == 100


def test_fake_is_deterministic_and_independent_of_num_envs():
    a = run_evaluation(EvalConfig(scenario="fake", num_envs=50))
    b = run_evaluation(EvalConfig(scenario="fake", num_envs=50))
    c = run_evaluation(EvalConfig(scenario="fake", num_envs=7))
    d = run_evaluation(EvalConfig(scenario="fake", num_envs=1))
    assert _strip_run(a) == _strip_run(b) == _strip_run(c) == _strip_run(d)
    assert a["run"]["config_hash"] == b["run"]["config_hash"]
    assert a["run"]["config_hash"] != c["run"]["config_hash"]  # num_envs is part of the config


def test_fake_output_is_pinned():
    """CI output must not drift: a digest of the episodes for the unseen and seen runs."""

    def digest(doc):
        blob = json.dumps(doc["episodes"], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    unseen = run_evaluation(EvalConfig(scenario="fake", split="unseen"))
    seen = run_evaluation(EvalConfig(scenario="fake", split="seen"))
    assert unseen["aggregates"]["success"]["count"] == 36
    assert seen["aggregates"]["success"]["count"] == 78
    assert digest(unseen) == "17932ae46108aff6dfad831ae421ab0e7ebef46c9ed528424be0cd5fbc3118a3"
    assert digest(seen) == "dad901e830398e3502a69b1d8b7743f2e928f7a413a83775b82922991270926d"


def test_fake_exercises_the_taxonomy_and_both_outcomes():
    doc = run_evaluation(EvalConfig(scenario="fake", split="seen"))
    counts = doc["aggregates"]["failure_tag_counts"]
    assert list(counts) == list(FAILURE_TAG_VALUES)
    assert sum(1 for n in counts.values() if n > 0) >= 4
    assert 0 < doc["aggregates"]["success"]["count"] < 100
    for e in doc["episodes"]:
        assert e["success"] == (e["failure_tag"] == "success")
        assert (e["metrics"]["completion_time_s"] is not None) == e["success"]
        assert e["metrics"]["fell"] == (e["failure_tag"] == "fall")


def test_synthetic_episode_is_a_pure_function_of_seed_and_split():
    assert fake.synthetic_episode(1003, "unseen") == fake.synthetic_episode(1003, "unseen")
    assert fake.synthetic_episode(3, "seen") != fake.synthetic_episode(4, "seen")
    ep = fake.synthetic_episode(1003, "unseen")
    assert set(ep.metrics) == set(metrics_for("fake"))


def test_fake_precedence_is_applied():
    """Whenever a fall is drawn, the tag is fall, whatever else was drawn."""
    for seed in range(2000):
        ep = fake.synthetic_episode(seed, "unseen")
        if ep.metrics["fell"]:
            assert ep.failure_tag is FailureTag.FALL and not ep.success


def test_seed_subset_and_order_are_respected():
    doc = run_evaluation(EvalConfig(scenario="fake", split="unseen", seeds=(1010, 1002, 1033)))
    assert [e["seed"] for e in doc["episodes"]] == [1010, 1002, 1033]
    full = run_evaluation(EvalConfig(scenario="fake", split="unseen"))
    by_seed = {e["seed"]: e for e in full["episodes"]}
    assert doc["episodes"] == [by_seed[s] for s in (1010, 1002, 1033)]


# ----------------------------------------------------------------------------- config checks


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scenario": "S9"},
        {"scenario": "fake", "split": "val"},
        {"scenario": "fake", "policy": "random"},
        {"scenario": "fake", "hand": "claw"},
        {"scenario": "fake", "base": "wheeled"},
        {"scenario": "fake", "num_envs": 0},
        {"scenario": "fake", "split": "unseen", "seeds": (0, 1)},
        {"scenario": "fake", "split": "seen", "seeds": (1000,)},
        {"scenario": "fake", "split": "seen", "seeds": (1, 1)},
    ],
)
def test_eval_config_rejects(kwargs):
    with pytest.raises(ValueError):
        EvalConfig(**kwargs)


def test_eval_config_defaults_to_the_default_seeds():
    assert EvalConfig(scenario="S4", split="seen").seeds == tuple(range(50))
    assert EvalConfig(scenario="S0", split="seen").seeds == tuple(range(100))
    assert EvalConfig(scenario="S0", split="unseen").seeds == tuple(range(1000, 1050))


def test_eval_config_split_defaults_to_seen():
    """The held-out split is never the default, for the Python API as for the CLI."""
    assert EvalConfig(scenario="S0").split == "seen"
    assert EvalConfig(scenario="S0").seeds == tuple(range(100))
    assert EvalConfig(scenario="S4").seeds == tuple(range(50))


# ----------------------------------------------------------------------------- custom backend


class _Backend:
    """A stand-in for a real scenario backend: S1 episodes, records how it was driven."""

    def __init__(self, config, bad_metrics=False, fail_in_batch=False):
        self.batches = []
        self.closed = 0
        self._bad = bad_metrics
        self._fail = fail_in_batch

    def resolved_config(self):
        return {"env": "stand-and-reach", "target_range_m": 0.4}

    def run_batch(self, seeds):
        self.batches.append(list(seeds))
        if self._fail:
            raise RuntimeError("simulator crashed")
        out = []
        for s in seeds:
            ok = s % 2 == 0
            metrics = {
                "fell": False,
                "torso_orientation_error_deg": 1.0,
                "hand_target_error_cm": 1.0 if ok else 5.0,
            }
            if self._bad:
                metrics.pop("fell")
            out.append(
                EpisodeResult(s, ok, FailureTag.SUCCESS if ok else FailureTag.TIMEOUT, 4.0, metrics)
            )
        return out

    def close(self):
        self.closed += 1


def _register_backend(monkeypatch, **kwargs):
    made = []

    def loader(config):
        made.append(_Backend(config, **kwargs))
        return made[-1]

    monkeypatch.setattr(runner, "_REGISTRY", dict(runner._REGISTRY))
    register_scenario(ScenarioSpec(name="S1", title="Stand and reach", loader=loader), replace=True)
    return made


def test_registered_backend_is_batched_by_num_envs_and_closed(monkeypatch):
    made = _register_backend(monkeypatch)
    doc = run_evaluation(EvalConfig(scenario="S1", split="seen", seeds=tuple(range(7)), num_envs=3))
    assert made[0].batches == [[0, 1, 2], [3, 4, 5], [6]]
    assert made[0].closed == 1
    validate_result(doc)
    run = doc["run"]
    assert run["synthetic"] is False and run["num_envs"] == 3
    assert doc["aggregates"]["rates"]["hand_target_under_3cm_rate"]["count"] == 4
    assert doc["aggregates"]["failure_tag_counts"]["timeout"] == 3
    expected_hash = config_hash(run_options(doc), {"env": "stand-and-reach", "target_range_m": 0.4})
    assert run["config_hash"] == expected_hash


def run_options(doc):
    keys = ("scenario", "split", "seeds", "num_envs", "policy", "hand", "base", "headless")
    return {k: doc["run"][k] for k in keys}


def test_backend_with_wrong_metrics_is_rejected_and_still_closed(monkeypatch):
    made = _register_backend(monkeypatch, bad_metrics=True)
    with pytest.raises(ValueError, match="metrics"):
        run_evaluation(EvalConfig(scenario="S1", split="seen", seeds=(0, 1)))
    assert made[0].closed == 1


def test_backend_crash_closes_the_simulator(monkeypatch):
    made = _register_backend(monkeypatch, fail_in_batch=True)
    with pytest.raises(RuntimeError, match="crashed"):
        run_evaluation(EvalConfig(scenario="S1", split="seen", seeds=(0, 1)))
    assert made[0].closed == 1


def test_inconsistent_success_and_tag_is_rejected():
    bad = EpisodeResult(0, True, FailureTag.FALL, 1.0, dict.fromkeys(metrics_for("fake")))
    with pytest.raises(ValueError, match="failure_tag"):
        runner._episode_record(bad, "fake")


# ----------------------------------------------------------------------------- metadata helpers


def test_config_hash_is_canonical_and_sensitive():
    a = config_hash({"x": 1, "y": [1, 2]}, {"k": "v"})
    assert a == config_hash({"y": [1, 2], "x": 1}, {"k": "v"})
    assert len(a) == 64 and int(a, 16) >= 0
    assert a != config_hash({"x": 2, "y": [1, 2]}, {"k": "v"})
    assert a != config_hash({"x": 1, "y": [1, 2]}, {"k": "w"})


def test_detect_gpu_name(monkeypatch):
    class Done:
        stdout = "NVIDIA GeForce RTX 5080\nNVIDIA Other\n"

    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: Done())
    assert detect_gpu_name() == "NVIDIA GeForce RTX 5080"

    class Empty:
        stdout = "\n"

    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: Empty())
    assert detect_gpu_name() == "none"

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(runner.subprocess, "run", missing)
    assert detect_gpu_name() == "none"


def test_git_info(monkeypatch):
    commit, dirty = git_info()
    assert commit == "unknown" or len(commit) == 40
    assert dirty is None or isinstance(dirty, bool)

    def missing(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(runner.subprocess, "run", missing)
    assert git_info() == ("unknown", None)


def test_default_output_path_and_write_result(tmp_path):
    doc = run_evaluation(EvalConfig(scenario="fake", split="unseen", seeds=(1000,)))
    doc["run"]["started_at"] = "2026-10-23T08:09:10Z"
    doc["run"]["git_commit"] = "abcdef0123456789abcdef0123456789abcdef01"
    path = default_output_path(doc, tmp_path)
    assert path.name == "20261023T080910Z_abcdef0_fake_unseen.json"
    doc["run"]["git_commit"] = "unknown"
    assert default_output_path(doc, tmp_path).name == "20261023T080910Z_nogit_fake_unseen.json"

    out = write_result(doc, tmp_path / "nested" / "r.json")
    raw = out.read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    assert json.loads(raw) == doc

    doc["run"]["wall_clock_s"] = float("nan")
    with pytest.raises(ValueError):
        write_result(doc, tmp_path / "nan.json")
