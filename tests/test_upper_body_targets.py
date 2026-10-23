"""The upper-body command math (#46): sampling, timing, interpolation, reach, curriculum."""

import json
from pathlib import Path

import numpy as np
import pytest

from kitchenbot.locomotion import upper_body_targets as ubt

FIXTURE = Path(__file__).parent / "fixtures" / "g1_dex3_joints_urdf.json"
JOINTS = {j["name"]: j for j in json.loads(FIXTURE.read_text(encoding="utf-8"))["joints"]}
LOWER = np.array([JOINTS[n]["lower"] for n in ubt.UPPER_BODY_JOINTS])
UPPER = np.array([JOINTS[n]["upper"] for n in ubt.UPPER_BODY_JOINTS])
SOFT = 0.9  # G1_29DOF_CFG soft_joint_pos_limit_factor
MID, HALF = (LOWER + UPPER) / 2, (UPPER - LOWER) / 2
SOFT_LOWER, SOFT_UPPER = MID - SOFT * HALF, MID + SOFT * HALF
DEFAULT = np.zeros(len(ubt.UPPER_BODY_JOINTS))  # G1_29DOF_CFG leaves arms and hands at 0


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_joint_order_and_right_arm_slice():
    assert len(ubt.UPPER_BODY_JOINTS) == 28
    right = ubt.UPPER_BODY_JOINTS[ubt.RIGHT_ARM_SLICE]
    assert right == tuple(ubt.REACH_BOX)
    assert all(name.startswith("right_") for name in right)


@pytest.mark.parametrize("factor", [0.0, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("limits", ["hard", "soft"])
def test_every_sample_is_inside_the_limits(factor, limits):
    lower, upper = (LOWER, UPPER) if limits == "hard" else (SOFT_LOWER, SOFT_UPPER)
    targets = ubt.sample_targets(lower, upper, DEFAULT, factor, rng(), 20_000)
    assert targets.shape == (20_000, 28)
    assert np.all(targets >= lower) and np.all(targets <= upper)


def test_soft_limits_exclude_a_zero_default_on_the_dex3():
    # Several Dex3 joints have 0 as a hard limit, so a zero default lies outside the soft range;
    # sampling clips the default into the limits first and still stays inside.
    outside = (DEFAULT < SOFT_LOWER) | (DEFAULT > SOFT_UPPER)
    assert outside.sum() == 10  # thumb_2, middle_0/1, index_0/1 on both hands
    low, high = ubt.sampling_bounds(SOFT_LOWER, SOFT_UPPER, DEFAULT, 0.2)
    assert np.all(low >= SOFT_LOWER) and np.all(high <= SOFT_UPPER) and np.all(low <= high)


def test_range_factor_one_spans_the_whole_range():
    low, high = ubt.sampling_bounds(LOWER, UPPER, DEFAULT, 1.0)
    np.testing.assert_allclose(low, LOWER)
    np.testing.assert_allclose(high, UPPER)
    targets = ubt.sample_targets(LOWER, UPPER, DEFAULT, 1.0, rng(), 50_000)
    span = targets.max(axis=0) - targets.min(axis=0)
    assert np.all(span > 0.98 * (UPPER - LOWER))


def test_range_factor_point_two_is_a_fifth_of_the_range_about_the_default():
    default = MID.copy()
    low, high = ubt.sampling_bounds(LOWER, UPPER, default, 0.2)
    np.testing.assert_allclose(high - low, 0.2 * (UPPER - LOWER))
    np.testing.assert_allclose((low + high) / 2, default)
    targets = ubt.sample_targets(LOWER, UPPER, default, 0.2, rng(), 20_000)
    assert np.all(targets >= low - 1e-12) and np.all(targets <= high + 1e-12)


def test_range_factor_zero_returns_the_default():
    default = MID.copy()
    targets = ubt.sample_targets(LOWER, UPPER, default, 0.0, rng(), 10)
    np.testing.assert_allclose(targets, np.broadcast_to(default, targets.shape))


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        ubt.sampling_bounds(LOWER, UPPER, DEFAULT, 1.5)
    with pytest.raises(ValueError):
        ubt.sampling_bounds(UPPER, LOWER, DEFAULT, 0.5)
    with pytest.raises(ValueError):
        ubt.sampling_bounds(LOWER, UPPER, DEFAULT[:3], 0.5)
    with pytest.raises(ValueError):
        ubt.sample_durations(rng(), 3, (0.0, 1.0))


def test_resample_times_are_in_one_to_three_seconds():
    durations = ubt.sample_durations(rng(), 50_000)
    assert durations.min() >= 1.0 and durations.max() <= 3.0
    assert abs(durations.mean() - 2.0) < 0.02


def test_interpolation_endpoints_and_midpoint():
    previous = np.zeros((2, 3))
    target = np.array([[1.0, 2.0, -3.0], [4.0, 4.0, 4.0]])
    duration = np.array([2.0, 1.0])
    np.testing.assert_allclose(ubt.interpolate(previous, target, np.zeros(2), duration), previous)
    np.testing.assert_allclose(ubt.interpolate(previous, target, duration, duration), target)
    half = ubt.interpolate(previous, target, duration / 2, duration)
    np.testing.assert_allclose(half, target / 2)
    # clipped outside [0, duration]
    np.testing.assert_allclose(ubt.interpolate(previous, target, duration * 3, duration), target)
    np.testing.assert_allclose(ubt.interpolate(previous, target, -duration, duration), previous)


def test_interpolated_path_stays_inside_the_limits():
    r = rng(3)
    previous = ubt.sample_targets(LOWER, UPPER, DEFAULT, 1.0, r, 100)
    target = ubt.sample_targets(LOWER, UPPER, DEFAULT, 1.0, r, 100)
    duration = ubt.sample_durations(r, 100)
    for fraction in np.linspace(0.0, 1.0, 11):
        path = ubt.interpolate(previous, target, fraction * duration, duration)
        assert np.all(path >= LOWER - 1e-12) and np.all(path <= UPPER + 1e-12)


def test_reach_pattern_frequency_over_many_episodes():
    mask = ubt.sample_reach_episodes(rng(7), 100_000)
    assert mask.dtype == bool
    assert abs(mask.mean() - 0.2) < 0.005
    assert not ubt.sample_reach_episodes(rng(), 1000, 0.0).any()
    assert ubt.sample_reach_episodes(rng(), 1000, 1.0).all()


def test_reach_moves_only_the_right_arm_inside_its_box():
    targets = ubt.sample_reach_targets(SOFT_LOWER, SOFT_UPPER, DEFAULT, rng(), 5000)
    held = np.clip(DEFAULT, SOFT_LOWER, SOFT_UPPER)
    others = np.ones(28, dtype=bool)
    others[ubt.RIGHT_ARM_SLICE] = False
    np.testing.assert_allclose(targets[:, others], np.broadcast_to(held[others], (5000, 21)))
    right = targets[:, ubt.RIGHT_ARM_SLICE]
    for column, (low, high) in enumerate(ubt.REACH_BOX.values()):
        assert right[:, column].min() >= low and right[:, column].max() <= high
    assert np.all(targets >= SOFT_LOWER) and np.all(targets <= SOFT_UPPER)


def test_reach_durations_are_slow():
    durations = ubt.sample_durations(rng(), 1000, ubt.REACH_DURATION_RANGE_S)
    assert durations.min() >= 3.0 and durations.max() <= 5.0


def test_sampling_is_deterministic_given_the_generator():
    def run(seed):
        r = rng(seed)
        return (
            ubt.sample_targets(LOWER, UPPER, DEFAULT, 0.6, r, 50),
            ubt.sample_durations(r, 50),
            ubt.sample_reach_episodes(r, 50),
            ubt.sample_reach_targets(LOWER, UPPER, DEFAULT, r, 5),
        )

    for a, b in zip(run(11), run(11), strict=True):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(run(11)[0], run(12)[0])


def test_fall_rate_smoothing():
    assert ubt.update_fall_rate(0.4, 0, 0) == 0.4  # no finished episode: unchanged
    assert ubt.update_fall_rate(0.4, 10, 10, smoothing=1.0) == 1.0
    assert ubt.update_fall_rate(0.4, 0, 10, smoothing=0.5) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        ubt.update_fall_rate(0.4, 5, 3)


def test_curriculum_moves_from_point_two_to_one_as_falls_stop():
    factor, rate, history = ubt.RANGE_FACTOR_START, 0.5, []
    for _ in range(40):  # batches with no falls
        rate = ubt.update_fall_rate(rate, 0, 1024)
        factor = ubt.next_range_factor(factor, rate)
        history.append(factor)
    assert history[0] == pytest.approx(0.2)  # rate 0.25: in the dead band, no step yet
    assert all(b >= a for a, b in zip(history, history[1:], strict=False))
    assert history[-1] == 1.0
    assert max(history) <= 1.0


def test_curriculum_backs_off_when_falls_return_and_never_drops_below_point_two():
    factor = 0.6
    factor = ubt.next_range_factor(factor, 0.5)
    assert factor == pytest.approx(0.55)
    for _ in range(50):
        factor = ubt.next_range_factor(factor, 0.9)
    assert factor == pytest.approx(0.2)
    assert ubt.next_range_factor(0.4, 0.2) == pytest.approx(0.4)  # dead band
