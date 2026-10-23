"""Wilson intervals and aggregates (docs/evaluation.md section 4, step 3)."""

import math

import pytest

from kitchenbot.evaluation.stats import (
    aggregate_episodes,
    summarize_rate,
    summarize_values,
    wilson_interval,
)

# Hand-checked with z = 1.96. For 0 of 10: z^2 = 3.8416, centre = (0 + 3.8416 / 20) / (1 + 0.38416)
# = 0.138773 and half-width = 1.96 * sqrt(3.8416 / 400) / 1.38416 = 0.138773, so the interval is
# [0, 0.2775]. 10 of 10 is its mirror image. For 50 of 100 the centre is 0.5 and the half-width is
# 1.96 * sqrt(0.0025 + 3.8416 / 40000) / 1.038416 = 0.096168. The 3 of 10 and 40 of 50 rows match
# the standard Wilson tables.
KNOWN = [
    (0, 10, 0.0, 0.2775),
    (10, 10, 0.7225, 1.0),
    (50, 100, 0.4038, 0.5962),
    (3, 10, 0.1078, 0.6032),
    (40, 50, 0.6696, 0.8876),
    (0, 1, 0.0, 0.7935),
]


@pytest.mark.parametrize(("k", "n", "low", "high"), KNOWN)
def test_wilson_known_values(k, n, low, high):
    got_low, got_high = wilson_interval(k, n)
    assert got_low == pytest.approx(low, abs=5e-5)
    assert got_high == pytest.approx(high, abs=5e-5)


def test_wilson_other_z():
    low, high = wilson_interval(50, 100, z=1.0)
    assert (low, high) == (pytest.approx(0.4502, abs=5e-5), pytest.approx(0.5498, abs=5e-5))


@pytest.mark.parametrize(("k", "n"), [(0, 7), (3, 7), (7, 7), (21, 50), (50, 50), (99, 100)])
def test_wilson_bounds_solve_the_defining_equation(k, n):
    """The bounds are the p where (p_hat - p)^2 = z^2 p (1 - p) / n, an independent check."""
    z = 1.96
    p_hat = k / n
    for p in wilson_interval(k, n):
        if 0.0 < p < 1.0:
            assert (p_hat - p) ** 2 == pytest.approx(z * z * p * (1 - p) / n, rel=1e-9)


def test_wilson_properties():
    for n in (1, 5, 50, 100):
        for k in range(n + 1):
            low, high = wilson_interval(k, n)
            assert 0.0 <= low <= k / n <= high <= 1.0
            m_low, m_high = wilson_interval(n - k, n)
            assert (low, high) == (pytest.approx(1 - m_high), pytest.approx(1 - m_low))
    widths = [high - low for low, high in (wilson_interval(n // 2, n) for n in (10, 50, 100, 500))]
    assert widths == sorted(widths, reverse=True)


def test_wilson_fifty_trials_is_roughly_twelve_points():
    """Section 4 says about plus or minus 12 points at 50 trials; 80 percent gives 11 points."""
    low, high = wilson_interval(40, 50)
    assert (high - low) / 2 == pytest.approx(0.109, abs=0.005)


@pytest.mark.parametrize(("k", "n"), [(0, 0), (-1, 5), (6, 5)])
def test_wilson_rejects_bad_input(k, n):
    with pytest.raises(ValueError):
        wilson_interval(k, n)


def test_summarize_rate():
    assert summarize_rate(1, 4) == {
        "count": 1,
        "n": 4,
        "rate": 0.25,
        "ci_low": pytest.approx(0.0456, abs=5e-5),
        "ci_high": pytest.approx(0.6994, abs=5e-5),
    }


def test_summarize_values():
    s = summarize_values([1.0, 2.0, 3.0, 10.0])
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(4.0)
    assert s["median"] == pytest.approx(2.5)
    assert s["std"] == pytest.approx(math.sqrt((9 + 4 + 1 + 36) / 3))  # sample std
    assert summarize_values([5.0]) == {"n": 1, "mean": 5.0, "std": None, "median": 5.0}
    assert summarize_values([]) == {"n": 0, "mean": None, "std": None, "median": None}


def _episode(seed, success, tag, **metrics):
    return {
        "seed": seed,
        "success": success,
        "failure_tag": tag,
        "duration_s": 1.0,
        "metrics": metrics,
    }


def test_aggregate_s0():
    episodes = [
        _episode(
            0,
            True,
            "success",
            fell=False,
            standing_duration_s=60.0,
            torso_orientation_error_deg=2.0,
        ),
        _episode(
            1,
            True,
            "success",
            fell=False,
            standing_duration_s=60.0,
            torso_orientation_error_deg=4.0,
        ),
        _episode(
            2, False, "fall", fell=True, standing_duration_s=12.0, torso_orientation_error_deg=30.0
        ),
        _episode(
            3,
            False,
            "timeout",
            fell=False,
            standing_duration_s=20.0,
            torso_orientation_error_deg=6.0,
        ),
    ]
    agg = aggregate_episodes("S0", episodes)
    assert agg["num_episodes"] == 4
    assert (agg["success"]["count"], agg["success"]["n"], agg["success"]["rate"]) == (2, 4, 0.5)
    assert set(agg["rates"]) == {"fall_rate"}
    assert agg["rates"]["fall_rate"]["count"] == 1
    assert set(agg["continuous"]) == {"standing_duration_s", "torso_orientation_error_deg"}
    assert agg["continuous"]["standing_duration_s"]["mean"] == pytest.approx(38.0)
    assert agg["failure_tag_counts"]["fall"] == 1
    assert agg["failure_tag_counts"]["timeout"] == 1
    assert sum(agg["failure_tag_counts"].values()) == 2  # failures only


def test_aggregate_s1_hand_target_fraction_under_3_cm():
    errors = [0.5, 2.9, 3.0, 8.0]
    episodes = [
        _episode(
            i,
            e < 3.0,
            "success" if e < 3.0 else "timeout",
            fell=False,
            torso_orientation_error_deg=1.0,
            hand_target_error_cm=e,
        )
        for i, e in enumerate(errors)
    ]
    agg = aggregate_episodes("S1", episodes)
    under = agg["rates"]["hand_target_under_3cm_rate"]
    assert (under["count"], under["n"]) == (2, 4)  # exactly 3.0 cm is not under 3 cm
    assert agg["continuous"]["hand_target_error_cm"]["median"] == pytest.approx(2.95)


def test_aggregate_s4_completion_time_covers_successes_only():
    base = {
        "fell": False,
        "torso_orientation_error_deg": 1.0,
        "grasp_success": True,
        "placement_success": True,
        "rack_collision": False,
    }
    episodes = [
        _episode(0, True, "success", completion_time_s=10.0, **base),
        _episode(1, True, "success", completion_time_s=30.0, **base),
        _episode(2, False, "misplace", completion_time_s=None, **base),
    ]
    agg = aggregate_episodes("S4", episodes)
    assert agg["continuous"]["completion_time_s"]["n"] == 2
    assert agg["continuous"]["completion_time_s"]["median"] == pytest.approx(20.0)
    assert agg["rates"]["collision_rate"]["count"] == 0
    assert set(agg["rates"]) == {
        "fall_rate",
        "grasp_success_rate",
        "placement_success_rate",
        "collision_rate",
    }


def test_aggregate_rejects_empty():
    with pytest.raises(ValueError):
        aggregate_episodes("S0", [])
