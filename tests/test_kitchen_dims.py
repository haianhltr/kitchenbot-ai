"""Authored kitchen dimensions and the rack slot layout (issue #47). No simulator, no pxr."""

import math
import re
from pathlib import Path

import pytest
from usd_helpers import REPO, imported_top_level_modules, load_script

from kitchenbot.planning.geometry import plate_vertical_quat, quat_apply, tilt_from_vertical
from kitchenbot.scene import kitchen_dims as kd
from kitchenbot.scene.kitchen_dims import RackSlotLayout

SCALES = [1.3, 1.4, 1.5]  # the range #11 expects to pick from


def test_module_uses_the_standard_library_only():
    imports = imported_top_level_modules(Path(kd.__file__))
    assert imports <= {"__future__", "dataclasses", "math"}, imports


def test_counter_numbers_come_from_issue_10():
    assert kd.COUNTER_HEIGHT_M == 0.90
    assert kd.COUNTER_DEPTH_M == 0.60
    assert kd.ROBOT_STANDOFF_M == 0.30  # D012: from #10's 0.45 m
    assert kd.COUNTER_WIDTH_M > 4 * 0.20  # room for the 20 cm plate box, the sink and a gap


def test_sink_and_floor_fit_the_counter_and_the_kitchen():
    sx, sy = kd.SINK_SIZE_M
    margin = 0.05
    assert kd.SINK_CENTRE_Y_M == kd.COUNTER_DEPTH_M / 2
    assert kd.SINK_CENTRE_X_M - sx / 2 >= -kd.COUNTER_WIDTH_M / 2 + margin
    assert kd.SINK_CENTRE_X_M + sx / 2 <= kd.COUNTER_WIDTH_M / 2 - margin
    assert sy / 2 + margin <= kd.SINK_CENTRE_Y_M <= kd.COUNTER_DEPTH_M - sy / 2 - margin
    assert kd.COUNTER_TOP_THICKNESS_M < kd.SINK_DEPTH_M < kd.COUNTER_HEIGHT_M - 0.5
    floor_x, floor_y = kd.FLOOR_SIZE_M
    assert floor_x >= kd.COUNTER_WIDTH_M + 2.0
    assert floor_y / 2 >= kd.COUNTER_DEPTH_M + kd.ROBOT_STANDOFF_M + 1.0
    assert 0.0 < kd.FLOOR_THICKNESS_M <= 0.2


def test_frictions_are_inside_the_randomisation_range_of_the_evaluation_protocol():
    text = (REPO / "docs" / "evaluation.md").read_text(encoding="utf-8")
    low, high = map(
        float, re.search(r"friction\s*\|\s*uniform in \[([\d.]+), ([\d.]+)\]", text).groups()
    )
    for friction in (kd.COUNTER_FRICTION, kd.FLOOR_FRICTION, kd.RACK_FRICTION):
        assert low <= friction <= high


def test_default_layout_is_five_slots_at_five_centimetres():
    layout = RackSlotLayout()
    assert layout.slots == 5
    assert layout.pitch == pytest.approx(0.05)
    assert layout.tine_height == pytest.approx(0.08)
    assert layout.rack_width_m == 0.366 and layout.rack_depth_m == 0.287  # #29
    assert layout.scale == 1.0
    assert len(layout.tine_xs) == 6  # six tines per row for five slots
    assert layout.row_ys == (-0.09, 0.09)


def test_slot_poses_increase_by_the_pitch_and_slot_3_is_centred():
    layout = RackSlotLayout()
    poses = layout.slot_poses()
    assert len(poses) == 5
    xs = [p[0] for p in poses]
    assert xs == pytest.approx([-0.10, -0.05, 0.0, 0.05, 0.10])
    assert [b - a for a, b in zip(xs, xs[1:], strict=False)] == pytest.approx([0.05] * 4)
    assert layout.slot_pose(3) == (0.0, 0.0, layout.slot_entry_z, 0.0)
    for _x, y, z, yaw in poses:
        assert y == 0.0 and yaw == 0.0
        assert z == layout.slot_entry_z == pytest.approx(0.09)  # base 10 mm + tines 80 mm


@pytest.mark.parametrize("slots", [1, 2, 3, 4, 5, 6, 7])
def test_slots_sit_in_the_gaps_between_tines_for_any_count(slots):
    layout = RackSlotLayout(slots=slots)
    xs = layout.tine_xs
    assert len(xs) == slots + 1
    for i in range(1, slots + 1):
        assert layout.slot_pose(i)[0] == pytest.approx((xs[i - 1] + xs[i]) / 2)
        assert layout.slot_pose(i)[0] == pytest.approx(-layout.slot_pose(slots + 1 - i)[0])
    assert layout.slot_pose((slots + 1) // 2)[0] == pytest.approx(
        0.0 if slots % 2 else -layout.pitch / 2
    )


@pytest.mark.parametrize("bad", [0, 6, -1, True, 2.0, "3"])
def test_slot_index_outside_one_to_slots_is_rejected(bad):
    with pytest.raises(ValueError):
        RackSlotLayout().slot_pose(bad)


@pytest.mark.parametrize("scale", SCALES)
def test_scale_multiplies_every_length(scale):
    base, scaled = RackSlotLayout(), RackSlotLayout(scale=scale)
    for name in (
        "pitch",
        "tine_height",
        "tine_radius",
        "row_spacing",
        "base_thickness",
        "width",
        "depth",
        "gap_width",
        "slot_entry_z",
    ):
        assert getattr(scaled, name) == pytest.approx(scale * getattr(base, name)), name
    assert scaled.tine_xs == pytest.approx(tuple(scale * x for x in base.tine_xs))
    assert scaled.row_ys == pytest.approx(tuple(scale * y for y in base.row_ys))
    for i in range(1, 6):
        x, y, z, yaw = base.slot_pose(i)
        assert scaled.slot_pose(i) == pytest.approx((scale * x, scale * y, scale * z, yaw))


def test_a_fixed_world_pitch_after_scaling_is_asked_for_explicitly():
    scale = 1.4
    layout = RackSlotLayout(scale=scale, pitch_m=0.05 / scale)
    assert layout.pitch == pytest.approx(0.05)
    assert layout.slot_pose(5)[0] - layout.slot_pose(4)[0] == pytest.approx(0.05)


def test_seen_and_unseen_slots_follow_the_evaluation_protocol():
    text = (REPO / "docs" / "evaluation.md").read_text(encoding="utf-8")
    seen_first, seen_last, unseen = map(
        int, re.search(r"Rack slot used \| slots (\d) to (\d) \| slot (\d)", text).groups()
    )
    assert kd.SEEN_SLOTS == tuple(range(seen_first, seen_last + 1)) == (1, 2, 3, 4)
    assert kd.UNSEEN_SLOTS == (unseen,) == (5,)
    assert set(kd.SEEN_SLOTS).isdisjoint(kd.UNSEEN_SLOTS)
    assert set(kd.SEEN_SLOTS) | set(kd.UNSEEN_SLOTS) == set(range(1, kd.RACK_SLOTS + 1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"slots": 0},
        {"slots": 2.5},
        {"scale": 0.0},
        {"scale": -1.0},
        {"pitch_m": float("nan")},
        {"tine_radius_m": 0.025},  # no gap left
        {"pitch_m": 0.08},  # five slots no longer fit across 0.366 m
        {"row_spacing_m": 0.29},  # rows outside the 0.287 m depth
    ],
)
def test_layouts_that_cannot_exist_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RackSlotLayout(**kwargs)


def test_slot_frame_matches_the_planner_convention():
    """The planner's vertical plate has its normal along the slot's x axis, which is the rack's."""
    layout = RackSlotLayout()
    for i in range(1, 6):
        quat = plate_vertical_quat(layout.slot_pose(i)[3])
        normal = quat_apply(quat, (0.0, 0.0, 1.0))
        assert abs(normal[0]) == pytest.approx(1.0)  # across the tines, which is how x runs here
        assert normal[1] == pytest.approx(0.0, abs=1e-12)
        assert tilt_from_vertical(quat) == pytest.approx(0.0, abs=1e-9)
        if i > 1:  # neighbouring slots differ along that same axis: plates face each other
            prev, cur = layout.slot_pose(i - 1), layout.slot_pose(i)
            assert (cur[0] - prev[0], cur[1] - prev[1]) == pytest.approx((0.05, 0.0))


def test_plate_constants_match_the_plate_generator():
    plates = load_script("make_plate_usd")
    assert plates.BODY_THICKNESS + plates.RIM_RISE == pytest.approx(kd.PLATE_HEIGHT_M)
    for diameter in kd.PLATE_DIAMETERS_M:
        name = f"plate_{round(diameter * 100)}.usda"
        assert (REPO / "source/kitchenbot/scene/assets/plates" / name).is_file()


@pytest.mark.parametrize("diameter", kd.PLATE_DIAMETERS_M)
def test_every_plate_fits_and_is_held_by_both_tine_rows(diameter):
    fit = RackSlotLayout().plate_fit(diameter)
    assert fit.fits
    assert fit.gap_width == pytest.approx(0.044)  # 50 mm pitch minus a 6 mm tine
    assert fit.side_clearance == pytest.approx(0.012)  # on each side of the 20 mm plate
    assert fit.depth_margin >= 0.0  # the plate is no wider than the rack is deep
    assert fit.row_margin == pytest.approx(0.0505)
    assert fit.engaged_height > 0.030  # both rows touch the plate over the top 3 cm of the tines
    assert 0.0 < fit.lean_limit_deg < 15.0  # tines stop a lean before the 15 degree criterion
    assert fit.top_z == pytest.approx(0.010 + diameter)


def test_numbers_for_the_largest_plate():
    fit = RackSlotLayout().plate_fit(0.27)
    assert fit.depth_margin == pytest.approx(0.0085)  # (0.287 - 0.27) / 2
    assert fit.engaged_height == pytest.approx(
        0.08 - (0.135 - math.sqrt(0.135**2 - 0.09**2)), abs=1e-9
    )
    assert fit.lean_limit_deg == pytest.approx(math.degrees(math.atan2(0.012, 0.08)))
    assert fit.top_z == pytest.approx(0.28)


def test_a_plate_too_small_for_the_rows_is_reported_as_not_fitting():
    fit = RackSlotLayout().plate_fit(0.15)  # radius 7.5 cm < rows at 9 cm: the rows miss it
    assert not fit.fits
    assert fit.engaged_height == 0.0 and fit.lean_limit_deg is None
    assert not RackSlotLayout().plate_fit(0.30).fits  # wider than the 0.287 m rack depth
    with pytest.raises(ValueError):
        RackSlotLayout().plate_fit(0.0)
