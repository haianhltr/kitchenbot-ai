"""Seeded randomization sampler (#52): determinism, split separation, ranges, plain data.

No simulator. The ranges under test are the rows of docs/evaluation.md section 2.
"""

import dataclasses
import json
import math
import subprocess
import sys

import pytest
import yaml

from kitchenbot.evaluation.metrics import SCENARIO_NAMES
from kitchenbot.evaluation.seeds import SPLITS, default_seeds, split_seeds
from kitchenbot.scene import kitchen_dims as kd
from kitchenbot.scene import randomization as rz
from kitchenbot.scene.randomization import (
    PLATE_NOMINAL_XY_M,
    SCENARIOS,
    EpisodeSpec,
    RandomizationRanges,
    SpawnDims,
    sample_episode,
    sample_split,
)

N_SEEDS = 1000  # "over 1,000 seeds per split"
EPS = 1e-12


def _specs(split: str, scenario: str, n: int = N_SEEDS) -> list[EpisodeSpec]:
    return [sample_episode(seed, split, scenario) for seed in range(n)]


@pytest.fixture(scope="module")
def drawn() -> dict[tuple[str, str], list[EpisodeSpec]]:
    """1000 specs for each split, for S4 (all but the push) and S0 (the push)."""
    return {(sp, sc): _specs(sp, sc) for sp in SPLITS for sc in ("S0", "S4")}


# ------------------------------------------------------------------ the table of section 2


def test_seen_ranges_are_the_table():
    r = RandomizationRanges.for_split("seen")
    assert r.split == "seen" and r == RandomizationRanges()
    assert r.plate_box_size_m == 0.20 and r.plate_box_shift_toward_robot_m == 0.0
    assert r.plate_yaw_half_range_deg == 30.0
    assert r.plate_diameters_m == (0.22, 0.25)
    assert r.rack_slots == (1, 2, 3, 4)
    assert r.pelvis_position_noise_m == 0.02 and r.pelvis_yaw_noise_deg == 5.0
    assert r.friction_range == (0.6, 1.0)
    assert r.push_force_range_n == (10.0, 30.0) and r.push_interval_s == 3.0
    assert r.push_scenarios == ("S0",)


def test_unseen_ranges_change_exactly_the_unseen_only_column():
    seen, unseen = RandomizationRanges.for_split("seen"), RandomizationRanges.for_split("unseen")
    changed = {
        f.name
        for f in dataclasses.fields(RandomizationRanges)
        if getattr(seen, f.name) != getattr(unseen, f.name)
    }
    assert changed == {
        "split",
        "plate_box_shift_toward_robot_m",
        "plate_diameters_m",
        "rack_slots",
        "push_force_range_n",
    }
    assert unseen.split == "unseen"
    assert unseen.plate_box_shift_toward_robot_m == 0.05
    assert unseen.plate_diameters_m == (0.27,)
    assert unseen.rack_slots == (5,)
    assert unseen.push_force_range_n == (40.0, 40.0)


def test_held_out_values_agree_with_kitchen_dims():
    seen, unseen = RandomizationRanges.for_split("seen"), RandomizationRanges.for_split("unseen")
    assert sorted(seen.plate_diameters_m + unseen.plate_diameters_m) == list(kd.PLATE_DIAMETERS_M)
    assert seen.rack_slots == kd.SEEN_SLOTS and unseen.rack_slots == kd.UNSEEN_SLOTS
    assert sorted(seen.rack_slots + unseen.rack_slots) == list(range(1, kd.RACK_SLOTS + 1))


def test_scenarios_are_s0_to_s4():
    assert SCENARIOS == tuple(n for n in SCENARIO_NAMES if n not in ("fake", "dry"))


def test_nominal_plate_point_is_on_the_counter_right_of_the_sink():
    x, y = PLATE_NOMINAL_XY_M
    assert y == kd.COUNTER_DEPTH_M / 2.0
    sink_right = kd.SINK_CENTRE_X_M + kd.SINK_SIZE_M[0] / 2.0
    assert x > sink_right  # "to the right of the sink": +x is the robot's right


def test_ranges_dict_round_trip_and_validation():
    for split in SPLITS:
        r = RandomizationRanges.for_split(split)
        d = r.to_dict()
        assert list(d)[0] == "split"
        assert RandomizationRanges.from_dict(json.loads(json.dumps(d))) == r
        assert RandomizationRanges.from_dict(yaml.safe_load(yaml.safe_dump(d))) == r
    with pytest.raises(ValueError, match="unknown randomization keys"):
        RandomizationRanges.from_dict({"split": "seen", "plate_colour": "red"})
    with pytest.raises(ValueError, match="unknown split"):
        RandomizationRanges.for_split("val")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"split": "val"},
        {"plate_box_size_m": 0.0},
        {"plate_box_shift_toward_robot_m": -0.05},
        {"plate_yaw_half_range_deg": -1.0},
        {"plate_diameters_m": ()},
        {"plate_diameters_m": (0.30,)},
        {"rack_slots": ()},
        {"rack_slots": (0,)},
        {"rack_slots": (6,)},
        {"rack_slots": (True,)},
        {"friction_range": (1.0, 0.6)},
        {"friction_range": (-0.1, 0.5)},
        {"friction_range": (0.6,)},
        {"push_force_range_n": (30.0, 10.0)},
        {"push_interval_s": 0.0},
        {"push_scenarios": ("S9",)},
        {"pelvis_position_noise_m": float("nan")},
    ],
)
def test_ranges_reject_nonsense(kwargs):
    with pytest.raises(ValueError):
        RandomizationRanges(**kwargs)


# ------------------------------------------------------------------ determinism


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_same_seed_same_spec(split, scenario):
    seed = 1000 if split == "unseen" else 7
    assert sample_episode(seed, split, scenario) == sample_episode(seed, split, scenario)


def test_an_episode_does_not_depend_on_what_else_was_sampled():
    """The spec of a seed is the same alone, after other draws, and inside sample_split."""
    alone = sample_episode(37, "seen", "S4")
    for seed in (5, 0, 99, 50):  # draw others first, in a different order
        sample_episode(seed, "seen", "S4")
        sample_episode(seed, "seen", "S0")
    assert sample_episode(37, "seen", "S4") == alone
    by_seed = {s.seed: s for s in sample_split("seen", "S4", seeds=range(100))}
    assert by_seed[37] == alone
    assert sample_split("seen", "S4", seeds=[37]) == (alone,)
    assert sample_split("seen", "S4", seeds=[99, 37])[1] == alone


def test_different_seeds_differ():
    specs = _specs("seen", "S4", 200)
    assert len({s.plate_position_m for s in specs}) == 200
    assert len({s.pelvis_offset_m for s in specs}) == 200
    assert len({(s.floor_friction, s.object_friction) for s in specs}) == 200


def test_the_same_seed_differs_across_scenarios_and_splits():
    base = sample_episode(7, "seen", "S4")
    for scenario in ("S0", "S1", "S2", "S3"):
        other = sample_episode(7, "seen", scenario)
        assert other.plate_position_m != base.plate_position_m
        assert other.pelvis_offset_m != base.pelvis_offset_m
    other = sample_episode(7, "unseen", "S4")  # same seed number, other sub-seed
    assert other.plate_position_m != base.plate_position_m


def test_sub_seed_rule_is_the_documented_one():
    """episode_rng is Generator(PCG64(SeedSequence([52, scenario number, split index, seed])))."""
    import numpy as np

    expected = np.random.Generator(np.random.PCG64(np.random.SeedSequence([52, 3, 1, 1017])))
    assert rz.episode_rng(1017, "unseen", "S3").random(4).tolist() == expected.random(4).tolist()
    assert rz.SEED_DOMAIN == 52


def test_pinned_draws():
    """The draw layout is frozen once M4 starts: a change here changes every episode."""
    a = sample_episode(0, "seen", "S4")
    assert a.plate_position_m == pytest.approx(
        (0.05054851062538507, 0.3094634527361313, 0.9), abs=EPS
    )
    assert a.plate_yaw_deg == pytest.approx(27.369331238651718, abs=EPS)
    assert (a.plate_diameter_m, a.rack_slot) == (0.25, 4)
    assert a.pelvis_offset_m == pytest.approx(
        (0.010879313881449141, -0.013265360878593965), abs=EPS
    )
    assert a.pelvis_yaw_offset_deg == pytest.approx(1.5516265293659783, abs=EPS)
    assert (a.floor_friction, a.object_friction) == pytest.approx(
        (0.8806875661491443, 0.8174217829179362), abs=EPS
    )
    assert a.push_times_s == ()

    b = sample_episode(1049, "unseen", "S0")
    assert b.plate_position_m == pytest.approx(
        (0.12896085230872917, 0.15267271517827344, 0.9), abs=EPS
    )
    assert (b.plate_diameter_m, b.rack_slot, b.rack_slot_pose) == (0.27, 5, (0.1, 0.0, 0.09, 0.0))
    assert b.push_times_s[:2] == (3.0, 6.0) and b.push_forces_n[:2] == (40.0, 40.0)

    c = sample_episode(5, "seen", "S0")
    assert c.push_forces_n[:3] == pytest.approx(
        (28.427647297680526, 29.238022892107814, 27.55791435445013), abs=EPS
    )


# ------------------------------------------------------------------ sample_split


@pytest.mark.parametrize(
    ("split", "scenario", "count"),
    [("seen", "S0", 100), ("seen", "S3", 100), ("seen", "S4", 50), ("unseen", "S1", 50)],
)
def test_sample_split_iterates_the_evaluation_seeds(split, scenario, count):
    specs = sample_split(split, scenario)
    assert [s.seed for s in specs] == default_seeds(scenario, split)
    assert len(specs) == count
    assert all(s.split == split and s.scenario == scenario for s in specs)
    assert set(s.seed for s in specs) <= set(split_seeds(split))


# ------------------------------------------------------------------ ranges over 1000 seeds


def _in(value, lo, hi):
    return lo - EPS <= value <= hi + EPS


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("scenario", ["S0", "S4"])
def test_every_drawn_value_is_inside_its_range(drawn, split, scenario):
    r = RandomizationRanges.for_split(split)
    cx, cy = r.plate_box_centre_xy_m
    half = r.plate_box_size_m / 2.0
    specs = drawn[(split, scenario)]
    assert len(specs) == N_SEEDS
    for s in specs:
        x, y, z = s.plate_position_m
        assert _in(x, cx - half, cx + half) and _in(y, cy - half, cy + half)
        assert z == kd.COUNTER_HEIGHT_M
        assert _in(s.plate_yaw_deg, -30.0, 30.0)
        assert s.plate_diameter_m in r.plate_diameters_m
        assert s.rack_slot in r.rack_slots
        assert s.rack_slot_pose == kd.RackSlotLayout().slot_pose(s.rack_slot)
        assert _in(s.pelvis_offset_m[0], -0.02, 0.02) and _in(s.pelvis_offset_m[1], -0.02, 0.02)
        assert _in(s.pelvis_yaw_offset_deg, -5.0, 5.0)
        assert _in(s.floor_friction, 0.6, 1.0) and _in(s.object_friction, 0.6, 1.0)


@pytest.mark.parametrize("split", SPLITS)
def test_the_whole_range_is_used(drawn, split):
    """A sampler stuck near the middle would pass the range test; check the spread."""
    specs = drawn[(split, "S4")]
    r = RandomizationRanges.for_split(split)
    cx, cy = r.plate_box_centre_xy_m
    for values, lo, hi in (
        ([s.plate_position_m[0] for s in specs], cx - 0.1, cx + 0.1),
        ([s.plate_position_m[1] for s in specs], cy - 0.1, cy + 0.1),
        ([s.plate_yaw_deg for s in specs], -30.0, 30.0),
        ([s.pelvis_offset_m[0] for s in specs], -0.02, 0.02),
        ([s.pelvis_offset_m[1] for s in specs], -0.02, 0.02),
        ([s.pelvis_yaw_offset_deg for s in specs], -5.0, 5.0),
        ([s.floor_friction for s in specs], 0.6, 1.0),
        ([s.object_friction for s in specs], 0.6, 1.0),
    ):
        width = hi - lo
        assert min(values) < lo + 0.02 * width and max(values) > hi - 0.02 * width
        assert abs(sum(values) / len(values) - (lo + hi) / 2) < 0.03 * width


def test_the_two_frictions_are_independent(drawn):
    specs = drawn[("seen", "S4")]
    assert any(s.floor_friction != s.object_friction for s in specs)
    n = len(specs)
    mf = sum(s.floor_friction for s in specs) / n
    mo = sum(s.object_friction for s in specs) / n
    cov = sum((s.floor_friction - mf) * (s.object_friction - mo) for s in specs) / n
    assert abs(cov) < 0.1 * (0.4**2 / 12)  # |correlation| well under 0.1


def test_seen_specs_never_use_an_unseen_only_value_and_vice_versa(drawn):
    seen = drawn[("seen", "S4")] + drawn[("seen", "S0")]
    unseen = drawn[("unseen", "S4")] + drawn[("unseen", "S0")]
    assert {s.plate_diameter_m for s in seen} == {0.22, 0.25}
    assert {s.rack_slot for s in seen} == {1, 2, 3, 4}
    assert {s.plate_diameter_m for s in unseen} == {0.27}
    assert {s.rack_slot for s in unseen} == {5}
    assert all(s.split == "seen" for s in seen) and all(s.split == "unseen" for s in unseen)
    for s in seen:
        assert all(_in(f, 10.0, 30.0) for f in s.push_forces_n)
        assert _in(s.plate_position_m[1], 0.2, 0.4)  # the unshifted box
    for s in unseen:
        assert all(f == 40.0 for f in s.push_forces_n)
        assert _in(s.plate_position_m[1], 0.15, 0.35)  # shifted 5 cm toward the robot
    # The shift is toward the robot: the unseen box reaches 5 cm closer to the counter's front edge.
    assert min(s.plate_position_m[1] for s in unseen) < min(s.plate_position_m[1] for s in seen)
    assert max(s.plate_position_m[1] for s in unseen) < max(s.plate_position_m[1] for s in seen)


def test_both_seen_diameters_and_all_seen_slots_are_drawn(drawn):
    from collections import Counter

    specs = drawn[("seen", "S4")]
    for counts, n in (
        (Counter(s.plate_diameter_m for s in specs), 2),
        (Counter(s.rack_slot for s in specs), 4),
    ):
        assert len(counts) == n
        assert min(counts.values()) > 0.8 * len(specs) / n  # roughly uniform


@pytest.mark.parametrize("split", SPLITS)
def test_the_plate_stays_on_the_counter_and_off_the_sink(drawn, split):
    """Every drawn plate lies entirely on the counter top and clear of the sink opening."""
    sink_lo = (
        kd.SINK_CENTRE_X_M - kd.SINK_SIZE_M[0] / 2.0,
        kd.SINK_CENTRE_Y_M - kd.SINK_SIZE_M[1] / 2.0,
    )
    sink_hi = (
        kd.SINK_CENTRE_X_M + kd.SINK_SIZE_M[0] / 2.0,
        kd.SINK_CENTRE_Y_M + kd.SINK_SIZE_M[1] / 2.0,
    )
    for s in drawn[(split, "S4")]:
        x, y, _ = s.plate_position_m
        radius = s.plate_diameter_m / 2.0
        assert radius <= y <= kd.COUNTER_DEPTH_M - radius
        assert abs(x) + radius <= kd.COUNTER_WIDTH_M / 2.0
        # the plate's bounding square misses the sink opening's rectangle
        assert x - radius > sink_hi[0] or x + radius < sink_lo[0] or y - radius > sink_hi[1]


# ------------------------------------------------------------------ pushes


def test_pushes_are_s0_only_every_three_seconds(drawn):
    for split in SPLITS:
        for s in drawn[(split, "S0")]:
            assert s.push_times_s == tuple(3.0 * k for k in range(1, 20))  # 3, 6, ..., 57
            assert len(s.push_forces_n) == len(s.push_directions_xy) == 19
        for scenario in ("S1", "S2", "S3", "S4"):
            s = sample_episode(3, split, scenario)
            assert s.push_times_s == s.push_forces_n == s.push_directions_xy == ()


def test_push_directions_are_horizontal_unit_vectors_in_every_direction(drawn):
    vectors = [d for s in drawn[("seen", "S0")] for d in s.push_directions_xy]
    assert len(vectors) == N_SEEDS * 19
    assert all(abs(math.hypot(*v) - 1.0) < 1e-12 for v in vectors)
    quadrants = [sum(1 for vx, vy in vectors if (vx >= 0) == qx and (vy >= 0) == qy)
                 for qx in (True, False) for qy in (True, False)]
    assert all(0.22 < q / len(vectors) < 0.28 for q in quadrants)
    forces = [f for s in drawn[("seen", "S0")] for f in s.push_forces_n]
    assert min(forces) < 10.5 and max(forces) > 29.5


def test_push_schedule_follows_the_interval_and_horizon():
    r = RandomizationRanges(push_interval_s=5.0, push_horizon_s=20.0)
    assert sample_episode(0, "seen", "S0", r).push_times_s == (5.0, 10.0, 15.0)  # not 20.0
    r = RandomizationRanges(push_scenarios=("S0", "S2"))
    assert len(sample_episode(0, "seen", "S2", r).push_times_s) == 19
    r = RandomizationRanges(push_horizon_s=3.0)
    assert sample_episode(0, "seen", "S0", r).push_times_s == ()


# ------------------------------------------------------------------ the spec is plain data


def _leaf_types(value):
    if isinstance(value, dict):
        return {t for v in value.values() for t in _leaf_types(v)}
    if isinstance(value, list):
        return {t for v in value for t in _leaf_types(v)}
    return {type(value)}


@pytest.mark.parametrize("scenario", ["S0", "S4"])
def test_spec_is_plain_data_with_a_to_dict(scenario):
    spec = sample_episode(1003, "unseen", scenario)
    d = spec.to_dict()
    assert list(d) == [f.name for f in dataclasses.fields(EpisodeSpec)]
    assert _leaf_types(d) <= {int, float, str}  # no numpy scalars, no tuples
    assert json.loads(json.dumps(d)) == d
    assert yaml.safe_load(yaml.safe_dump(d)) == d
    # the dataclass itself holds only plain values too
    for f in dataclasses.fields(spec):
        value = getattr(spec, f.name)
        assert type(value).__module__ == "builtins", f.name
    assert hash(spec) == hash(sample_episode(1003, "unseen", scenario))


def test_spec_is_frozen():
    spec = sample_episode(0, "seen", "S4")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.rack_slot = 2  # type: ignore[misc]


# ------------------------------------------------------------------ dims and validation


def test_dims_feed_the_counter_height_and_the_slot_pose():
    layout = kd.RackSlotLayout(scale=1.5, pitch_m=0.05 / 1.5, tine_height_m=0.08 / 1.5)
    dims = SpawnDims(counter_height_m=0.95, rack=layout)
    spec = sample_episode(12, "unseen", "S3", dims=dims)
    assert spec.plate_position_m[2] == 0.95
    assert spec.rack_slot_pose == layout.slot_pose(5)
    assert sample_episode(12, "unseen", "S3").plate_position_m[2] == kd.COUNTER_HEIGHT_M
    assert SpawnDims() == SpawnDims(kd.COUNTER_HEIGHT_M, kd.RackSlotLayout())


def test_a_slot_the_rack_does_not_have_is_an_error():
    dims = SpawnDims(rack=kd.RackSlotLayout(slots=4))
    with pytest.raises(ValueError, match="slot index"):
        sample_episode(0, "unseen", "S4", dims=dims)  # unseen draws slot 5


@pytest.mark.parametrize(
    "args",
    [
        (-1, "seen", "S4"),
        (True, "seen", "S4"),
        (1.5, "seen", "S4"),
        ("3", "seen", "S4"),
        (0, "val", "S4"),
        (0, "seen", "fake"),
        (0, "seen", "S5"),
    ],
)
def test_bad_arguments_are_rejected(args):
    with pytest.raises(ValueError):
        sample_episode(*args)


def test_ranges_of_the_wrong_split_are_rejected():
    with pytest.raises(ValueError, match="split"):
        sample_episode(0, "unseen", "S4", RandomizationRanges.for_split("seen"))
    with pytest.raises(ValueError):
        sample_split("val", "S4")
    with pytest.raises(ValueError):
        sample_split("seen", "fake")


def test_importing_the_sampler_pulls_in_no_simulator():
    code = (
        "import sys\n"
        "import kitchenbot.scene.randomization\n"
        "banned = {'isaaclab', 'isaacsim', 'omni', 'torch', 'carb', 'warp', 'pxr'}\n"
        "hits = sorted(m for m in sys.modules if m.split('.')[0] in banned)\n"
        "assert not hits, hits\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
