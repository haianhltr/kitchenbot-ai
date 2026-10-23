"""Seed splits and the failure taxonomy (docs/evaluation.md sections 1, 4 and 7)."""

import re
from pathlib import Path

import pytest

from kitchenbot.evaluation.failure_tags import (
    FAILURE_PRECEDENCE,
    FAILURE_TAG_VALUES,
    FailureTag,
    primary_failure_tag,
)
from kitchenbot.evaluation.seeds import (
    SEEN_SEEDS,
    UNSEEN_SEEDS,
    default_seeds,
    parse_seeds,
    split_of_seed,
    split_seeds,
)

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- seeds


def test_split_definitions_match_the_doc():
    assert split_seeds("seen") == list(range(0, 100))
    assert split_seeds("unseen") == list(range(1000, 1050))
    assert len(SEEN_SEEDS) == 100 and len(UNSEEN_SEEDS) == 50
    assert not set(SEEN_SEEDS) & set(UNSEEN_SEEDS)


def test_split_seeds_returns_a_copy_and_rejects_unknown():
    seeds = split_seeds("seen")
    seeds.append(-1)
    assert split_seeds("seen")[-1] == 99
    with pytest.raises(ValueError):
        split_seeds("val")


def test_split_of_seed():
    assert split_of_seed(0) == "seen" and split_of_seed(99) == "seen"
    assert split_of_seed(1000) == "unseen" and split_of_seed(1049) == "unseen"
    assert split_of_seed(100) is None and split_of_seed(1050) is None


def test_default_trial_counts_follow_section_4():
    for scenario in ("S0", "S1", "S2", "S3", "fake"):
        assert default_seeds(scenario, "seen") == list(range(100))
    assert default_seeds("S4", "seen") == list(range(50))
    for scenario in ("S0", "S1", "S2", "S3", "S4", "fake"):
        assert default_seeds(scenario, "unseen") == list(range(1000, 1050))


def test_parse_seeds_forms():
    assert parse_seeds(None, "S4", "unseen") == list(range(1000, 1050))
    assert parse_seeds("", "S0", "seen") == list(range(100))
    assert parse_seeds("10", "S0", "seen") == list(range(10))  # a bare integer is a count
    assert parse_seeds("1000-1049", "S4", "unseen") == list(range(1000, 1050))
    assert parse_seeds("0,5,10-12", "S0", "seen") == [0, 5, 10, 11, 12]
    assert parse_seeds("1003-1003", "S0", "unseen") == [1003]
    assert parse_seeds(" 7 - 8 ", "S0", "seen") == [7, 8]


@pytest.mark.parametrize(
    ("spec", "split"),
    [
        ("0", "seen"),  # count of zero
        ("51", "unseen"),  # more than the split holds
        ("a,b", "seen"),
        ("5-3", "seen"),
        ("1,1", "seen"),
        ("0-5,3", "seen"),
        ("1000", "seen"),  # explicit comma-free integer is a count, 1000 > 100
        ("5,1000", "seen"),  # unseen seed in the seen split
        ("0,1", "unseen"),  # seen seeds in the unseen split
        ("0-5000", "seen"),  # range larger than the split
        ("1,,2", "seen"),
    ],
)
def test_parse_seeds_rejects(spec, split):
    with pytest.raises(ValueError):
        parse_seeds(spec, "S0", split)


# ----------------------------------------------------------------------------- failure tags


def test_failure_tags_match_section_7_of_the_doc():
    """The enum is the taxonomy in docs/evaluation.md section 7, in the doc order."""
    text = (REPO / "docs" / "evaluation.md").read_text(encoding="utf-8")
    section = text.split("## 7. Failure taxonomy", 1)[1].split("## 8.", 1)[0]
    paragraph = next(p for p in section.split("\n\n") if p.startswith("`fall`"))
    in_doc = re.findall(r"`([a-z_]+)`", paragraph)
    assert in_doc == list(FAILURE_TAG_VALUES)
    assert FailureTag.SUCCESS.value == "success"
    assert len(FailureTag) == len(in_doc) + 1


def test_enum_values_are_json_strings():
    assert FailureTag("fall") is FailureTag.FALL
    assert str(FailureTag.GRASP_MISS) == "grasp_miss"
    assert FailureTag.DROP == "drop"


def test_precedence_is_enum_order_with_timeout_last():
    assert FAILURE_PRECEDENCE[0] is FailureTag.FALL
    assert FAILURE_PRECEDENCE[-1] is FailureTag.TIMEOUT
    assert FailureTag.SUCCESS not in FAILURE_PRECEDENCE


def test_primary_failure_tag_picks_highest_precedence():
    assert primary_failure_tag([FailureTag.TIMEOUT, FailureTag.FALL]) is FailureTag.FALL
    assert primary_failure_tag(["misplace", "rack_collision"]) is FailureTag.RACK_COLLISION
    assert primary_failure_tag([FailureTag.DROP, FailureTag.TIMEOUT]) is FailureTag.DROP
    assert primary_failure_tag([FailureTag.TIMEOUT]) is FailureTag.TIMEOUT
    assert primary_failure_tag([FailureTag.SUCCESS, FailureTag.GRASP_MISS]) is FailureTag.GRASP_MISS
    for tag in FAILURE_PRECEDENCE:
        assert primary_failure_tag(FAILURE_PRECEDENCE[FAILURE_PRECEDENCE.index(tag) :]) is tag


def test_primary_failure_tag_needs_a_cause():
    with pytest.raises(ValueError):
        primary_failure_tag([])
    with pytest.raises(ValueError):
        primary_failure_tag([FailureTag.SUCCESS])
    with pytest.raises(ValueError):
        primary_failure_tag(["not_a_tag"])
