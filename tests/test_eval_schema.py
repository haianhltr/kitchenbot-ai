"""schema.json: valid schema, in step with the metric catalogue, rejects bad results."""

import copy

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from kitchenbot.evaluation.failure_tags import ALL_TAG_VALUES, FAILURE_TAG_VALUES
from kitchenbot.evaluation.metrics import (
    SCENARIO_NAMES,
    continuous_names_for,
    metrics_for,
    rate_names_for,
)
from kitchenbot.evaluation.runner import BASES, HANDS, POLICIES, EvalConfig, run_evaluation
from kitchenbot.evaluation.schema import load_schema, validate_result
from kitchenbot.evaluation.seeds import SPLITS
from kitchenbot.evaluation.stats import aggregate_episodes

_DEFAULTS = {
    "fell": False,
    "standing_duration_s": 60.0,
    "torso_orientation_error_deg": 2.0,
    "hand_target_error_cm": 1.5,
    "grasp_success": True,
    "placement_success": True,
    "rack_collision": False,
    "completion_time_s": 12.0,
}


def make_doc(scenario: str) -> dict:
    """A hand-built, valid document for any scenario: one success and one fall."""
    metrics = {name: _DEFAULTS[name] for name in metrics_for(scenario)}
    failed = dict(metrics)
    if "fell" in failed:
        failed["fell"] = True
    if "completion_time_s" in failed:
        failed["completion_time_s"] = None
    episodes = [
        {
            "seed": 1000,
            "success": True,
            "failure_tag": "success",
            "duration_s": 14.0,
            "metrics": metrics,
        },
        {
            "seed": 1001,
            "success": False,
            "failure_tag": "fall",
            "duration_s": 5.5,
            "metrics": failed,
        },
    ]
    return {
        "schema_version": 1,
        "run": {
            "scenario": scenario,
            "split": "unseen",
            "seeds": [1000, 1001],
            "num_envs": 2,
            "policy": "own",
            "hand": "dex3",
            "base": "floating",
            "headless": True,
            "synthetic": scenario == "fake",
            "git_commit": "0" * 40,
            "git_dirty": False,
            "config_hash": "a" * 64,
            "gpu_name": "none" if scenario == "fake" else "NVIDIA GeForce RTX 5080",
            "host": "test-host",
            "started_at": "2026-10-23T12:00:00Z",
            "wall_clock_s": 1.5,
            "package_version": "0.1.0",
        },
        "episodes": episodes,
        "aggregates": aggregate_episodes(scenario, episodes),
    }


def test_schema_is_valid_draft_2020_12():
    schema = load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(schema)


def test_top_level_fields():
    schema = load_schema()
    assert set(schema["required"]) == {"schema_version", "run", "episodes", "aggregates"}
    assert set(schema["properties"]) == set(schema["required"])
    assert schema["additionalProperties"] is False


def test_enums_match_the_code():
    defs = load_schema()["$defs"]
    run = defs["run"]["properties"]
    assert run["scenario"]["enum"] == list(SCENARIO_NAMES)
    assert run["split"]["enum"] == list(SPLITS)
    assert run["policy"]["enum"] == list(POLICIES)
    assert run["hand"]["enum"] == list(HANDS)
    assert run["base"]["enum"] == list(BASES)
    assert defs["episode"]["properties"]["failure_tag"]["enum"] == list(ALL_TAG_VALUES)
    counts = defs["aggregates"]["properties"]["failure_tag_counts"]
    assert counts["required"] == list(FAILURE_TAG_VALUES)
    assert set(counts["properties"]) == set(FAILURE_TAG_VALUES)


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_scenario_rules_match_the_metric_catalogue(scenario):
    rules = [
        r["then"]["properties"]
        for r in load_schema()["allOf"]
        if r["if"]["properties"]["run"]["properties"]["scenario"]["const"] == scenario
    ]
    assert len(rules) == 1
    metrics = rules[0]["episodes"]["items"]["properties"]["metrics"]
    assert metrics["required"] == list(metrics_for(scenario))
    assert metrics["propertyNames"]["enum"] == list(metrics_for(scenario))
    agg = rules[0]["aggregates"]["properties"]
    assert agg["rates"]["required"] == list(rate_names_for(scenario))
    assert agg["continuous"]["required"] == list(continuous_names_for(scenario))


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_valid_document_for_every_scenario(scenario):
    validate_result(make_doc(scenario))


def test_a_produced_document_validates():
    doc = run_evaluation(
        EvalConfig(scenario="fake", split="unseen", seeds=(1000, 1001, 1002)), validate=False
    )
    validate_result(doc)


def _set(path, value):
    def mutate(doc):
        node = doc
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return mutate


def _delete(path):
    def mutate(doc):
        node = doc
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]

    return mutate


S4_MUTATIONS = {
    "unknown top-level field": _set(["extra"], 1),
    "missing run": _delete(["run"]),
    "bad schema_version": _set(["schema_version"], 2),
    "unknown scenario": _set(["run", "scenario"], "S9"),
    "bad split": _set(["run", "split"], "val"),
    "bad policy": _set(["run", "policy"], "random"),
    "bad hand": _set(["run", "hand"], "claw"),
    "bad base": _set(["run", "base"], "wheeled"),
    "headless not bool": _set(["run", "headless"], "yes"),
    "short git commit": _set(["run", "git_commit"], "abc123"),
    "bad config hash": _set(["run", "config_hash"], "xyz"),
    "bad timestamp": _set(["run", "started_at"], "yesterday"),
    "negative wall clock": _set(["run", "wall_clock_s"], -1),
    "unknown run field": _set(["run", "extra"], 1),
    "missing gpu": _delete(["run", "gpu_name"]),
    "duplicate seeds": _set(["run", "seeds"], [1000, 1000]),
    "no episodes": _set(["episodes"], []),
    "unknown failure tag": _set(["episodes", 1, "failure_tag"], "exploded"),
    "success with a failure tag": _set(["episodes", 0, "failure_tag"], "fall"),
    "failure tagged success": _set(["episodes", 1, "failure_tag"], "success"),
    "missing metric": _delete(["episodes", 0, "metrics", "fell"]),
    "metric from another scenario": _set(["episodes", 0, "metrics", "standing_duration_s"], 5.0),
    "metric of wrong type": _set(["episodes", 0, "metrics", "fell"], "no"),
    "negative metric": _set(["episodes", 0, "metrics", "torso_orientation_error_deg"], -1),
    "completion time on a failure": _set(["episodes", 1, "metrics", "completion_time_s"], 9.0),
    "unknown episode field": _set(["episodes", 0, "extra"], 1),
    "rate above one": _set(["aggregates", "success", "rate"], 1.5),
    "missing rate": _delete(["aggregates", "rates", "fall_rate"]),
    "rate of another scenario": _set(
        ["aggregates", "rates", "hand_target_under_3cm_rate"],
        {"count": 0, "n": 2, "rate": 0.0, "ci_low": 0.0, "ci_high": 0.66},
    ),
    "missing failure tag count": _delete(["aggregates", "failure_tag_counts", "drop"]),
    "success in failure tag counts": _set(["aggregates", "failure_tag_counts", "success"], 1),
    "wrong ci method": _set(["aggregates", "ci_method"], "normal"),
}


@pytest.mark.parametrize("name", S4_MUTATIONS)
def test_schema_rejects(name):
    doc = make_doc("S4")
    validate_result(doc)
    S4_MUTATIONS[name](doc)
    with pytest.raises(ValidationError):
        validate_result(doc)


def test_schema_rejects_a_metric_missing_for_another_scenario():
    doc = make_doc("S0")
    del doc["episodes"][0]["metrics"]["standing_duration_s"]
    with pytest.raises(ValidationError):
        validate_result(doc)


def test_fake_must_be_marked_synthetic_and_real_must_not():
    doc = make_doc("fake")
    doc["run"]["synthetic"] = False
    with pytest.raises(ValidationError):
        validate_result(doc)
    doc = make_doc("S4")
    doc["run"]["synthetic"] = True
    with pytest.raises(ValidationError):
        validate_result(doc)


def test_cross_field_checks():
    doc = make_doc("S4")
    bad = copy.deepcopy(doc)
    bad["aggregates"]["success"]["count"] = 2
    bad["aggregates"]["success"]["rate"] = 1.0
    with pytest.raises(ValueError, match="aggregates"):
        validate_result(bad)

    bad = copy.deepcopy(doc)
    bad["run"]["seeds"] = [1001, 1000]
    with pytest.raises(ValueError, match="seeds"):
        validate_result(bad)

    bad = copy.deepcopy(doc)
    bad["run"]["split"] = "seen"
    with pytest.raises(ValueError, match="split"):
        validate_result(bad)


def _with_traces(doc, traces):
    doc = copy.deepcopy(doc)
    doc["episodes"][0]["traces"] = traces
    return doc


def test_traces_are_optional_and_validated_when_present():
    doc = make_doc("S4")
    assert all("traces" not in e for e in doc["episodes"])
    validate_result(doc)  # no traces: still valid
    good = {"sample_rate_hz": 10.0, "torso_orientation_error_deg": [1.0, 2.5, 0.0]}
    validate_result(_with_traces(doc, good))
    validate_result(_with_traces(doc, {**good, "sample_rate_hz": 0.5}))


@pytest.mark.parametrize(
    "traces",
    [
        {"torso_orientation_error_deg": [1.0]},
        {"sample_rate_hz": 10.0},
        {"sample_rate_hz": 0, "torso_orientation_error_deg": [1.0]},
        {"sample_rate_hz": -10.0, "torso_orientation_error_deg": [1.0]},
        {"sample_rate_hz": "10", "torso_orientation_error_deg": [1.0]},
        {"sample_rate_hz": 10.0, "torso_orientation_error_deg": []},
        {"sample_rate_hz": 10.0, "torso_orientation_error_deg": [1.0, -0.5]},
        {"sample_rate_hz": 10.0, "torso_orientation_error_deg": [1.0, None]},
        {"sample_rate_hz": 10.0, "torso_orientation_error_deg": [1.0], "extra": []},
        [1.0, 2.0],
    ],
)
def test_bad_traces_are_rejected(traces):
    with pytest.raises(ValidationError):
        validate_result(_with_traces(make_doc("S4"), traces))
