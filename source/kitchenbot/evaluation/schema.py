"""Load ``schema.json`` and validate result documents against it.

``validate_result`` does two things:

1. JSON Schema validation (draft 2020-12) against ``schema.json``, which pins the field names and
   types, the enums for scenario, split, policy, hand, base and failure tag, which metrics each
   scenario carries, and that ``success`` agrees with ``failure_tag``.
2. Cross-field checks a schema cannot express: the episodes are the run's seeds in order, the
   seeds belong to the run's split, and ``aggregates`` is exactly what the episodes aggregate to.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from importlib import resources
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, best_match

from kitchenbot.evaluation.seeds import split_seeds
from kitchenbot.evaluation.stats import aggregate_episodes

__all__ = ["ValidationError", "load_schema", "validate_result"]


@cache
def load_schema() -> dict[str, Any]:
    """The parsed ``schema.json`` shipped next to this module."""
    text = resources.files("kitchenbot.evaluation").joinpath("schema.json").read_text("utf-8")
    return json.loads(text)


@cache
def _validator() -> Draft202012Validator:
    return Draft202012Validator(load_schema())


def validate_result(doc: Mapping[str, Any]) -> None:
    """Raise if ``doc`` is not a valid, internally consistent result document.

    Schema violations raise ``jsonschema.ValidationError`` (the most relevant one); cross-field
    violations raise ``ValueError``.
    """
    errors = sorted(_validator().iter_errors(doc), key=lambda e: [str(p) for p in e.path])
    if errors:
        raise best_match(errors)
    run = doc["run"]
    episodes = doc["episodes"]
    if [e["seed"] for e in episodes] != run["seeds"]:
        raise ValueError("episode seeds are not the run seeds in order")
    outside = sorted(set(run["seeds"]) - set(split_seeds(run["split"])))
    if outside:
        raise ValueError(f"seeds {outside[:5]} are not in split {run['split']!r}")
    expected = aggregate_episodes(run["scenario"], episodes)
    if doc["aggregates"] != expected:
        raise ValueError("aggregates do not match the episodes")
