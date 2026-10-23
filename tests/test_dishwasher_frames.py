"""Dishwasher placement, rack fit and slot frames (#59). No simulator, no pxr."""

import math

import pytest

from kitchenbot.scene import dishwasher_frames as df
from kitchenbot.scene import kitchen_dims as kd

PLATE_25 = 0.25
COUNTER_BOX = ((-kd.COUNTER_WIDTH_M / 2, kd.COUNTER_WIDTH_M / 2), (0.0, kd.COUNTER_DEPTH_M))
#: The right leg's joint line is 0.116 to 0.119 m to the right of the pelvis (URDF fixture, #48
#: kinematics at the default pose); with a link radius of about 5 cm its surface is near 0.17 m.
LEG_SURFACE_M = 0.17


# --------------------------------------------------------------------------------------------
# rack_fits_plate
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("scale", "depth_margin"), [(1.0, 0.0185), (1.3, 0.06155), (1.5, 0.09025)])
def test_a_25_cm_plate_fits_the_rack_at_the_candidate_scales(scale, depth_margin):
    fit = kd.rack_fits_plate(scale, PLATE_25)
    assert fit.fits
    assert fit.diameter == PLATE_25
    assert fit.depth_margin == pytest.approx(depth_margin, abs=1e-9)
    assert fit.depth_margin == pytest.approx((kd.RACK_DEPTH_M * scale - PLATE_25) / 2.0)
    # The tines do not scale with the asset: gap, side clearance and lean limit stay the same.
    assert fit.gap_width == pytest.approx(kd.RACK_SLOT_PITCH_M - 2 * kd.RACK_TINE_RADIUS_M)
    assert fit.side_clearance == pytest.approx(0.012)
    assert fit.lean_limit_deg == pytest.approx(8.53, abs=0.01)
    assert fit.row_margin == pytest.approx(
        kd.RACK_DEPTH_M * scale / 2 - (kd.RACK_ROW_SPACING_M / 2 + kd.RACK_TINE_RADIUS_M)
    )


def test_rack_fits_plate_defaults_and_bad_scale():
    assert kd.RACK_SCALE == 1.0  # #59 keeps it; #11 decides
    assert kd.rack_fits_plate() == kd.rack_fits_plate(kd.RACK_SCALE, 0.25)
    assert kd.rack_fits_plate(1.0, 0.27).fits  # the held-out plate, 8.5 mm to spare in depth
    assert kd.rack_fits_plate(1.0, 0.27).depth_margin == pytest.approx(0.0085)
    assert not kd.rack_fits_plate(1.0, 0.30).fits  # wider than the 0.287 m rack
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            kd.rack_fits_plate(bad)


# --------------------------------------------------------------------------------------------
# Tine attachment and slot poses
# --------------------------------------------------------------------------------------------


def test_tine_pose_is_the_rack_centre_on_its_floor():
    x, y, z, yaw = kd.RACK_TINES_POSE_IN_LINK
    assert x * kd.DISHWASHER_LOWER_RACK_LINK_SCALE_X == pytest.approx(kd.RACK_WIDTH_M / 2, abs=5e-4)
    assert y == pytest.approx(kd.RACK_DEPTH_M / 2, abs=5e-4)
    assert 0.0 < z < kd.DISHWASHER_RACK_HEIGHT_M
    assert yaw == 0.0


@pytest.mark.parametrize("scale", [1.0, 1.3, 1.5])
def test_compensating_scale_cancels_link_and_asset_scale(scale):
    sx, sy, sz = df.rack_tines_scale_in_link(scale)
    assert sx * kd.DISHWASHER_LOWER_RACK_LINK_SCALE_X * scale == pytest.approx(1.0, abs=1e-12)
    assert sy * scale == pytest.approx(1.0) and sz * scale == pytest.approx(1.0)


def test_slot_poses_in_the_kitchen():
    poses = df.slot_poses_kitchen()
    assert len(poses) == kd.RACK_SLOTS
    x3, y3, z3, yaw3 = poses[2]
    assert (x3, y3, z3) == pytest.approx((0.3502, -0.2600, 0.7396), abs=1e-3)
    assert yaw3 == pytest.approx(-math.pi / 2)  # a seated plate's normal points along -y
    # The row runs along y at the tine pitch, slot 1 nearest the counter, all at one x and height.
    for (xa, ya, za, _), (xb, yb, zb, _) in zip(poses, poses[1:], strict=False):
        assert xa == pytest.approx(xb) and za == pytest.approx(zb)
        assert ya - yb == pytest.approx(kd.RACK_SLOT_PITCH_M)
    # Entry height: rack floor + base frame + tines.
    rack_floor = kd.DISHWASHER_POSE_KITCHEN[2] + df.rack_frame_in_root()[2]
    assert z3 - rack_floor == pytest.approx(kd.RACK_BASE_THICKNESS_M + kd.RACK_TINE_HEIGHT_M)


def test_slot_poses_follow_the_rack_joint():
    out = df.slot_poses_kitchen(rack_pos=-0.4)
    authored = df.slot_poses_kitchen(rack_pos=-0.2)
    # Pulling the rack out moves the row towards -x of the root link, i.e. towards the robot (-x).
    assert authored[2][0] - out[2][0] == pytest.approx(0.2)
    assert authored[2][1:3] == pytest.approx(out[2][1:3])


@pytest.mark.parametrize(
    ("standoff", "forward", "horizontal"), [(0.45, 0.19, 0.398), (0.25, -0.01, 0.350)]
)
def test_rack_origin_from_the_pelvis_at_both_standoffs(standoff, forward, horizontal):
    origin = df.root_to_kitchen(df.rack_frame_in_root(rack_pos=-kd.DISHWASHER_RACK_TRAVEL_M))
    f, left, up = df.kitchen_to_pelvis(origin, standoff)
    assert f == pytest.approx(forward, abs=2e-3)
    assert left == pytest.approx(-0.350, abs=2e-3)  # to the robot's right
    assert up == pytest.approx(-0.10, abs=2e-3)
    assert math.hypot(f, left) == pytest.approx(horizontal, abs=2e-3)


def test_slot_row_from_the_pelvis():
    row = df.slot_row_from_pelvis(standoff=0.25)
    assert [round(r[0], 3) for r in row] == [0.09, 0.04, -0.01, -0.06, -0.11]
    assert all(r[1] == pytest.approx(-0.350, abs=2e-3) for r in row)
    assert all(0.34 < r[3] < 0.38 for r in row)
    far = df.slot_row_from_pelvis(standoff=0.45, rack_pos=kd.DISHWASHER_RACK_LIMITS_M[0])
    assert all(r[1] == pytest.approx(-0.550, abs=2e-3) for r in far)  # authored travel: too far


# --------------------------------------------------------------------------------------------
# Clearances of the pose (the reasons in the DISHWASHER_POSE_KITCHEN comment)
# --------------------------------------------------------------------------------------------


def _bounds(corners):
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    return (min(xs), max(xs)), (min(ys), max(ys))


@pytest.mark.parametrize("rack_pos", [-kd.DISHWASHER_RACK_TRAVEL_M, kd.DISHWASHER_RACK_LIMITS_M[0]])
def test_dishwasher_is_clear_of_the_counter(rack_pos):
    parts = df.footprints_kitchen(rack_pos=rack_pos)
    for name, (corners, _z) in parts.items():
        (x0, x1), (y0, y1) = _bounds(corners)
        assert y1 < COUNTER_BOX[1][0], f"{name} reaches the counter: y max {y1}"
    (_, _), (_, rack_y1) = _bounds(parts["rack"][0])
    assert rack_y1 <= -0.05  # #56: a vertical path above the slots must not meet the counter edge


def test_open_door_and_rack_clear_the_right_leg():
    parts = df.footprints_kitchen()
    for name in ("door", "rack"):
        (x0, _), _ = _bounds(parts[name][0])
        z0, z1 = parts[name][1]
        assert z1 < df.PELVIS_HEIGHT_M  # at leg height, so the lateral clearance is what counts
        assert x0 - LEG_SURFACE_M >= 0.03, f"{name} only {x0 - LEG_SURFACE_M:.3f} m from the leg"
    (bx0, _), _ = _bounds(parts["body"][0])
    assert bx0 > 0.5


def test_heights_match_the_56_band():
    origin_z = kd.DISHWASHER_POSE_KITCHEN[2] + df.rack_frame_in_root()[2]
    assert 0.5 <= origin_z <= 0.7  # #56: rack base 0.5 to 0.7 m above the floor


def test_rack_travel_clears_the_body_and_the_authored_travel_does_not():
    assert df.min_rack_travel(1.0, 0.27, 0.0) == pytest.approx(0.295, abs=1e-3)
    for scale in (1.0, 1.3, 1.5):
        assert kd.DISHWASHER_RACK_TRAVEL_M >= df.min_rack_travel(scale, 0.27, 0.02)
    assert -kd.DISHWASHER_RACK_LIMITS_M[0] < df.min_rack_travel(1.0, min(kd.PLATE_DIAMETERS_M))
    with pytest.raises(ValueError):
        df.min_rack_travel(1.0, 0.0)


def test_plinth_fills_the_gap_under_the_body():
    (cx, cy, cz), (sx, sy, sz) = df.plinth_box_kitchen()
    assert sz == pytest.approx(kd.DISHWASHER_POSE_KITCHEN[2]) and cz == pytest.approx(sz / 2)
    body = sorted(kd.DISHWASHER_BODY_SIZE_M[:2])
    assert sorted((sx, sy)) == pytest.approx(body)
    (x0, x1), (y0, y1) = _bounds(df.footprints_kitchen()["body"][0])
    assert (cx, cy) == pytest.approx(((x0 + x1) / 2, (y0 + y1) / 2))
    with pytest.raises(ValueError):
        df.plinth_box_kitchen(pose=(0.6, -0.3, 0.5, 0.3))


def test_slot_waypoints_are_in_the_48_envelope_at_the_closer_standoff():
    """Palm points around every slot (insertion, slot, 25 cm above; side or top rim grasp)."""
    from kitchenbot.controllers.reachability import ReachabilityEnvelope

    envelope = ReachabilityEnvelope.load()
    for x, y, z, _ in df.slot_poses_kitchen():
        for grasp in (0.125, 0.25):
            for dz in (-0.05, 0.0, 0.25):
                p = df.kitchen_to_pelvis((x, y, z + grasp + dz), standoff=0.25)
                assert envelope.margin_contains(p, 0.02), (x, y, z, grasp, dz)


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def test_quaternion_order_is_isaac_labs():
    q = df.yaw_quat_xyzw(-math.pi / 2)
    assert q == pytest.approx((0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)))
    assert df.yaw_quat_xyzw(0.0) == (0.0, 0.0, 0.0, 1.0)


def test_door_gains_are_converted_to_per_radian():
    assert df.DOOR_STIFFNESS_PER_RAD == pytest.approx(1145.916, abs=1e-3)
    assert df.DOOR_DAMPING_PER_RAD == pytest.approx(572.958, abs=1e-3)
    assert (df.RACK_STIFFNESS, df.RACK_DAMPING) == (10.0, 10.0)


def test_kitchen_to_pelvis_axes():
    assert df.kitchen_to_pelvis((0.0, -0.45, 0.75), 0.45) == pytest.approx((0.0, 0.0, 0.0))
    assert df.kitchen_to_pelvis((0.3, -0.45, 0.75), 0.45) == pytest.approx((0.0, -0.3, 0.0))
    assert df.kitchen_to_pelvis((0.0, 0.0, 1.0), 0.45) == pytest.approx((0.45, 0.0, 0.25))
