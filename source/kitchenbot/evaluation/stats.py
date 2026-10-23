"""Statistics for evaluation results: Wilson intervals and per-run aggregates.

docs/evaluation.md section 4, step 3: every reported rate carries a 95 percent Wilson confidence
interval. Pure standard library, so it runs on CI without numpy.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from kitchenbot.evaluation.failure_tags import FAILURE_TAG_VALUES
from kitchenbot.evaluation.metrics import (
    HAND_TARGET_RATE,
    HAND_TARGET_THRESHOLD_CM,
    METRICS,
    continuous_names_for,
    metrics_for,
)

#: z for a two-sided 95 percent interval.
Z_95 = 1.96

#: Decimal places kept for floats in aggregates, so the JSON stays short and stable.
_DIGITS = 6


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval ``(low, high)`` for ``successes`` out of ``n`` trials.

    With ``p = successes / n``::

        centre = (p + z^2 / 2n) / (1 + z^2 / n)
        half   = z * sqrt(p (1 - p) / n + z^2 / 4n^2) / (1 + z^2 / n)

    The result lies in ``[0, 1]`` and contains ``p``. ``n`` must be at least 1 and
    ``0 <= successes <= n``.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0 <= successes <= n:
        raise ValueError("successes must satisfy 0 <= successes <= n")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    # The interval always contains p; the clamps only remove floating point error at 0 and 1.
    return max(0.0, min(p, centre - half)), min(1.0, max(p, centre + half))


def summarize_rate(count: int, n: int) -> dict[str, Any]:
    """A rate with its 95 percent Wilson interval, in the shape ``schema.json`` expects."""
    low, high = wilson_interval(count, n)
    return {
        "count": count,
        "n": n,
        "rate": round(count / n, _DIGITS),
        "ci_low": round(low, _DIGITS),
        "ci_high": round(high, _DIGITS),
    }


def summarize_values(values: Sequence[float]) -> dict[str, Any]:
    """``n``, mean, sample standard deviation (``n - 1``) and median of ``values``.

    Statistics that are undefined are ``None``: everything for ``n == 0``, and ``std`` for
    ``n == 1``.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "median": None}
    return {
        "n": n,
        "mean": round(statistics.fmean(values), _DIGITS),
        "std": round(statistics.stdev(values), _DIGITS) if n > 1 else None,
        "median": round(statistics.median(values), _DIGITS),
    }


def aggregate_episodes(scenario: str, episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the ``aggregates`` object from episode records.

    ``episodes`` are the per-episode dicts of the result document (``seed``, ``success``,
    ``failure_tag``, ``duration_s``, ``metrics``). Boolean metrics become rates over all episodes,
    numeric metrics are summarised over the episodes where they are not ``None`` (so
    ``completion_time_s`` covers successes only), and ``hand_target_under_3cm_rate`` is derived
    from ``hand_target_error_cm``.
    """
    n = len(episodes)
    if n < 1:
        raise ValueError("cannot aggregate zero episodes")
    rates: dict[str, Any] = {}
    for name in metrics_for(scenario):
        spec = METRICS[name]
        if spec.kind == "bool" and spec.rate_name:
            count = sum(1 for e in episodes if e["metrics"][name])
            rates[spec.rate_name] = summarize_rate(count, n)
        if name == "hand_target_error_cm":
            count = sum(
                1
                for e in episodes
                if e["metrics"][name] is not None and e["metrics"][name] < HAND_TARGET_THRESHOLD_CM
            )
            rates[HAND_TARGET_RATE] = summarize_rate(count, n)
    continuous = {
        name: summarize_values(
            [e["metrics"][name] for e in episodes if e["metrics"][name] is not None]
        )
        for name in continuous_names_for(scenario)
    }
    counts = dict.fromkeys(FAILURE_TAG_VALUES, 0)
    for e in episodes:
        if not e["success"]:
            counts[e["failure_tag"]] += 1
    return {
        "num_episodes": n,
        "ci_method": "wilson",
        "ci_level": 0.95,
        "success": summarize_rate(sum(1 for e in episodes if e["success"]), n),
        "rates": rates,
        "continuous": continuous,
        "failure_tag_counts": counts,
    }
