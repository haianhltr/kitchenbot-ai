"""The waist-lean command math (#68, D012): sampling, ramp, tracking error, command timeline."""

import numpy as np
import pytest

from kitchenbot.locomotion import waist_lean_targets as wlt
from kitchenbot.robots import joint_maps
from kitchenbot.robots.joint_limits import SOFT_LIMIT_FRACTION, WAIST_LEAN_MAX_RAD, WAIST_LIMITS

DT = 0.02  # the policy step: decimation 4 at 200 Hz


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_constants_match_d012_and_the_urdf():
    # D012 as amended 2026-10-24 (#69 E8, #77): the lean is capped at the soft limit, 0.9 x the
    # 0.52 rad URDF limit of waist_pitch_joint rounded to 3 decimals, the planner's constant
    pitch = WAIST_LIMITS["waist_pitch_joint"]
    assert wlt.MAX_LEAN_RAD == WAIST_LEAN_MAX_RAD == round(SOFT_LIMIT_FRACTION * pitch.upper, 3)
    assert wlt.MAX_LEAN_RAD == 0.468 and pitch.upper == 0.52
    assert wlt.MAX_LEAN_RAD < pitch.upper  # never a target at the hard stop
    assert pitch.lower == -0.52  # the command never goes backward
    assert wlt.ZERO_SHARE == 0.5
    assert wlt.RESAMPLE_TIME_RANGE_S == (1.0, 3.0)
    assert wlt.MAX_RATE_RAD_S == 1.0
    assert wlt.WAIST_PITCH_JOINT == "waist_pitch_joint"


def test_waist_indices_follow_joint_maps_order():
    names = [joint_maps.WAIST[i] for i in (wlt.WAIST_YAW_INDEX, wlt.WAIST_ROLL_INDEX)]
    assert names == ["waist_yaw_joint", "waist_roll_joint"]
    assert joint_maps.WAIST[wlt.WAIST_PITCH_INDEX] == wlt.WAIST_PITCH_JOINT
    assert sorted((wlt.WAIST_YAW_INDEX, wlt.WAIST_ROLL_INDEX, wlt.WAIST_PITCH_INDEX)) == [0, 1, 2]


def test_targets_are_in_range_and_half_are_exactly_zero():
    targets = wlt.sample_lean_targets(rng(), 200_000)
    assert targets.shape == (200_000,)
    assert targets.min() >= 0.0 and targets.max() <= wlt.MAX_LEAN_RAD
    zero_share = np.mean(targets == 0.0)
    assert abs(zero_share - 0.5) < 0.01  # 9 standard errors at this sample size
    lean = targets[targets > 0.0]
    assert abs(lean.mean() - wlt.MAX_LEAN_RAD / 2) < 0.005  # uniform on [0, 0.468]
    assert lean.max() > 0.99 * wlt.MAX_LEAN_RAD and lean.min() < 0.01 * wlt.MAX_LEAN_RAD
    # uniform: each tenth of the range holds a tenth of the non-zero draws
    counts, _ = np.histogram(lean, bins=10, range=(0.0, wlt.MAX_LEAN_RAD))
    assert np.all(np.abs(counts / lean.size - 0.1) < 0.01)


@pytest.mark.parametrize("share", [0.0, 0.25, 1.0])
def test_zero_share_is_a_parameter(share):
    targets = wlt.sample_lean_targets(rng(1), 100_000, zero_share=share)
    assert abs(np.mean(targets == 0.0) - share) < 0.01


def test_max_lean_is_a_parameter():
    targets = wlt.sample_lean_targets(rng(2), 50_000, max_lean=0.3)
    assert targets.max() <= 0.3 and targets.max() > 0.29
    assert np.all(wlt.sample_lean_targets(rng(), 100, max_lean=0.0) == 0.0)


def test_same_seed_same_targets():
    a = wlt.sample_lean_targets(rng(7), 100)
    b = wlt.sample_lean_targets(rng(7), 100)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, wlt.sample_lean_targets(rng(8), 100))


def test_bad_sampling_arguments_are_rejected():
    with pytest.raises(ValueError):
        wlt.sample_lean_targets(rng(), 10, max_lean=-0.1)
    with pytest.raises(ValueError):
        wlt.sample_lean_targets(rng(), 10, zero_share=1.5)
    with pytest.raises(ValueError):
        wlt.sample_lean_targets(rng(), 10, zero_share=-0.1)


def test_slew_moves_at_most_one_step_and_never_overshoots():
    command = np.array([0.0, 0.5, 0.3, 0.3, 0.0])
    target = np.array([0.468, 0.0, 0.3, 0.31, 0.0])
    out = wlt.slew_toward(command, target, 0.02)
    np.testing.assert_allclose(out, [0.02, 0.48, 0.3, 0.31, 0.0])
    assert np.all(np.abs(out - command) <= 0.02 + 1e-12)
    assert np.all(np.abs(out - target) <= np.abs(command - target) + 1e-12)


def test_slew_arrives_exactly_at_zero():
    # 0.1234567 is not a multiple of the step, so the last step is a partial one.
    command, steps = np.array([0.1234567]), 0
    while command[0] != 0.0:
        command = wlt.slew_toward(command, np.zeros(1), 0.02)
        steps += 1
        assert steps < 100
    assert steps == 7  # ceil(0.1234567 / 0.02)


def test_a_full_lean_takes_about_half_a_second():
    command, steps = np.zeros(1), 0
    while command[0] < wlt.MAX_LEAN_RAD - 1e-9:
        command = wlt.slew_toward(command, np.full(1, wlt.MAX_LEAN_RAD), wlt.MAX_RATE_RAD_S * DT)
        steps += 1
    assert steps == 24  # 0.468 rad at 1 rad/s = 0.468 s: 23 full steps of 20 ms and a partial one
    assert steps * DT < wlt.RESAMPLE_TIME_RANGE_S[0]  # the ramp ends before the next target


def test_tracking_error_is_pitch_error_plus_yaw_and_roll():
    pitch = np.array([0.2, 0.0, 0.52, 0.1])  # joint positions; 0.52 is the hard stop, not a command
    yaw = np.array([0.0, 0.1, -0.2, 0.0])
    roll = np.array([0.0, 0.0, 0.05, -0.3])
    command = np.array([0.2, 0.0, 0.0, 0.3])
    np.testing.assert_allclose(
        wlt.tracking_error(pitch, yaw, roll, command), [0.0, 0.1, 0.77, 0.5]
    )


def test_tracking_error_is_zero_only_on_the_commanded_posture():
    command = np.array([0.0, 0.3, wlt.MAX_LEAN_RAD])
    zeros = np.zeros(3)
    assert np.all(wlt.tracking_error(command, zeros, zeros, command) == 0.0)
    # the old reward pulled the waist to zero even when a lean was commanded; this one does not
    assert np.all(wlt.tracking_error(zeros, zeros, zeros, command)[1:] > 0.0)


# -- the command timeline ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def trace():
    """512 environments for one 20 s episode."""
    return wlt.simulate_command(rng(3), num_envs=512, num_steps=1000, dt=DT)


def test_trace_shapes(trace):
    assert trace.command.shape == trace.target.shape == trace.resampled.shape == (1000, 512)
    assert trace.dt == DT


def test_command_stays_in_range_and_starts_upright(trace):
    assert trace.command.min() >= 0.0 and trace.command.max() <= wlt.MAX_LEAN_RAD
    assert np.all(trace.command[0] <= wlt.MAX_RATE_RAD_S * DT + 1e-12)


def test_command_never_moves_faster_than_the_rate(trace):
    steps = np.abs(np.diff(trace.command, axis=0, prepend=0.0))
    assert steps.max() <= wlt.MAX_RATE_RAD_S * DT + 1e-12
    assert steps.max() > 0.99 * wlt.MAX_RATE_RAD_S * DT  # and it does use the full rate


def test_resampled_targets_are_half_zero(trace):
    drawn = trace.target[trace.resampled]
    assert drawn.size > 4000  # about ten draws per 20 s episode
    assert abs(np.mean(drawn == 0.0) - 0.5) < 0.03
    assert drawn.min() >= 0.0 and drawn.max() <= wlt.MAX_LEAN_RAD


def test_resampling_interval_is_one_to_three_seconds(trace):
    intervals = []
    for env in range(trace.resampled.shape[1]):
        times = np.flatnonzero(trace.resampled[:, env]) * trace.dt
        intervals.extend(np.diff(times))
    intervals = np.array(intervals)
    assert intervals.size > 3000
    # a draw of T seconds fires on the first step at or after T, so the gap lies in [T, T + dt)
    assert intervals.min() >= wlt.RESAMPLE_TIME_RANGE_S[0] - 1e-9
    assert intervals.max() < wlt.RESAMPLE_TIME_RANGE_S[1] + DT + 1e-9
    assert abs(intervals.mean() - (2.0 + DT / 2)) < 0.05


def test_command_arrives_at_the_target_before_the_next_draw(trace):
    """After 25 steps (0.50 s, one more than the longest ramp, 24 steps for 0.468 rad) the command
    equals its target, and it is exactly 0.0 whenever that target is zero."""
    checked = zero_checked = 0
    for env in range(trace.resampled.shape[1]):
        draws = np.flatnonzero(trace.resampled[:, env])
        ends = list(draws[1:]) + [trace.command.shape[0]]
        for start, end in zip(draws, ends, strict=True):
            for step in range(start + 25, end):
                assert abs(trace.command[step, env] - trace.target[step, env]) < 1e-12
            if end > start + 25:
                checked += 1
                if trace.target[start, env] == 0.0:
                    assert trace.command[end - 1, env] == 0.0
                    zero_checked += 1
    assert checked > 2000 and zero_checked > 1000


def test_command_only_approaches_its_target_between_draws(trace):
    """No overshoot and no wobble: the distance to the target never grows except at a redraw."""
    distance = np.abs(trace.command - trace.target)
    grew = np.diff(distance, axis=0) > 1e-12
    assert not grew[~trace.resampled[1:]].any()
    assert grew[trace.resampled[1:]].any()  # a new target does start a new distance


def test_step_mode_follows_the_target_immediately():
    step = wlt.simulate_command(rng(4), 16, 400, DT, max_rate=None)
    assert np.array_equal(step.command, step.target)
    assert np.abs(np.diff(step.command, axis=0)).max() > wlt.MAX_RATE_RAD_S * DT  # it jumps


def test_timeline_is_deterministic_and_validates_its_arguments():
    a = wlt.simulate_command(rng(5), 4, 100, DT)
    b = wlt.simulate_command(rng(5), 4, 100, DT)
    assert np.array_equal(a.command, b.command) and np.array_equal(a.resampled, b.resampled)
    for bad in (dict(num_envs=0), dict(num_steps=0), dict(dt=0.0)):
        args = dict(num_envs=1, num_steps=1, dt=DT) | bad
        with pytest.raises(ValueError):
            wlt.simulate_command(rng(), **args)
