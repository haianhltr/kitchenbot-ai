"""Waist-range envelope and the reach-gap analysis (#56). numpy and matplotlib, no simulator."""

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kitchenbot.controllers import reachability as rc
from kitchenbot.controllers.reachability import ReachabilityEnvelope
from kitchenbot.geometry import Pose, to_frame
from kitchenbot.robots import joint_maps as jm
from kitchenbot.robots.joint_limits import WAIST_LEAN_MAX_RAD, WAIST_LIMITS
from kitchenbot.robots.kinematics import UrdfChain, right_arm_chain
from kitchenbot.scene import kitchen_dims as kd

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "g1_dex3_joints_urdf.json"
ENVELOPE_SCRIPT = REPO / "scripts" / "make_reach_envelope.py"
ANALYSIS_SCRIPT = REPO / "scripts" / "reach_gap_analysis.py"
COMMITTED = rc.DEFAULT_PATH
COMMITTED_WAIST = rc.WAIST_PATH
FIGURE = REPO / "docs" / "figures" / "reach_gap.png"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mre = _load(ENVELOPE_SCRIPT, "make_reach_envelope_56")
rga = _load(ANALYSIS_SCRIPT, "reach_gap_analysis")


@pytest.fixture(scope="module")
def joints():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["joints"]


@pytest.fixture(scope="module")
def chain(joints):
    return right_arm_chain(joints)


@pytest.fixture(scope="module")
def committed():
    return ReachabilityEnvelope.load(COMMITTED)


@pytest.fixture(scope="module")
def committed_waist():
    return ReachabilityEnvelope.load(COMMITTED_WAIST)


def _limits_deg(name):
    limit = WAIST_LIMITS[name]
    return math.degrees(limit.lower), math.degrees(limit.upper)


# --- waist ranges and sampling ----------------------------------------------------------------


def test_waist_ranges_are_clipped_to_the_urdf_limits():
    clipped = rc.clip_waist_range(((-200.0, 200.0), (-40.0, 40.0), (-50.0, 50.0)))
    for name, (lo, hi) in zip(jm.WAIST, clipped, strict=True):
        assert (lo, hi) == pytest.approx(_limits_deg(name)), name
    np.testing.assert_allclose(
        rc.waist_range_deg(20, 0, 0, 30), ((-20.0, 20.0), (0.0, 0.0), (0.0, math.degrees(0.52)))
    )
    assert rc.waist_range_deg() == rc.ZERO_WAIST_DEG
    assert rc.waist_range_deg(-15, 0, 0, 0)[0] == (-15.0, 15.0)  # a half range, sign-free
    with pytest.raises(ValueError):
        rc.clip_waist_range(((0.0, 0.0), (0.0, 0.0), (10.0, 5.0)))
    with pytest.raises(ValueError):
        rc.clip_waist_range(((0.0, 0.0), (0.0, 0.0)))


def test_waist_sampling_stays_inside_the_limits_and_the_requested_range(chain):
    wide = ((-200.0, 200.0), (-40.0, 40.0), (-50.0, 50.0))
    q, waist, positions = rc.sample_configurations(chain, 4000, 3, waist_range=wide)
    assert q.shape == (4000, 7) and waist.shape == (4000, 3) and positions.shape == (4000, 3)
    for i, name in enumerate(jm.WAIST):
        limit = WAIST_LIMITS[name]
        assert np.all(waist[:, i] >= limit.lower) and np.all(waist[:, i] <= limit.upper), name
        # the draws fill the clipped range, they are not piled up on the limits
        assert waist[:, i].min() < limit.lower + 0.05 and waist[:, i].max() > limit.upper - 0.05
    _, waist, _ = rc.sample_configurations(chain, 4000, 3, waist_range=rc.WAIST_ENVELOPE_RANGE_DEG)
    yaw, roll, pitch = waist.T
    assert np.all(np.abs(yaw) <= math.radians(20.0) + 1e-12)
    assert np.all(roll == 0.0)
    # the committed range leans up to the soft limit the planner commands (#77), not the URDF's
    assert np.all(pitch >= 0.0) and np.all(pitch <= WAIST_LEAN_MAX_RAD + 1e-12)
    assert pitch.max() > WAIST_LEAN_MAX_RAD - 0.01


def test_sampled_positions_are_the_forward_kinematics_of_the_sampled_waist(chain):
    q, waist, positions = rc.sample_configurations(
        chain, 50, 5, waist_range=rc.WAIST_ENVELOPE_RANGE_DEG
    )
    for row in range(0, 50, 7):
        np.testing.assert_allclose(chain.fk(q[row], held=waist[row])[0], positions[row], atol=1e-12)
    # the arm draws do not depend on the waist range: the waist has its own stream
    q0, waist0, _ = rc.sample_configurations(chain, 50, 5)
    assert np.array_equal(q, q0) and not np.any(waist0)


def test_positive_waist_pitch_leans_the_shoulder_forward(joints):
    shoulder = UrdfChain.from_joint_dump(
        joints, "pelvis", "right_shoulder_pitch_link", active=["right_shoulder_pitch_joint"]
    )
    upright = shoulder.fk([0.0], held=[0.0, 0.0, 0.0])[0]
    leaning = shoulder.fk([0.0], held=[0.0, 0.0, 0.3])[0]
    assert leaning[0] - upright[0] > 0.05  # forward, along the pelvis x
    assert abs(leaning[1] - upright[1]) < 1e-9


def test_tip_offset_moves_the_sampled_point_in_the_palm_frame(chain):
    offset = (0.05, 0.0, 0.0)
    q, positions = rc.sample_tip_positions(chain, 20, 9, tip_offset_m=offset)
    pos, rot = chain.fk_batch(q)
    np.testing.assert_allclose(positions, pos + rot @ np.array(offset), atol=1e-12)


# --- the zero range is #48 --------------------------------------------------------------------


def test_zero_range_draws_are_the_48_draws(chain):
    # #48's sampler, written out: one default_rng(seed), uniform in fixed batches, waist at zero
    lower, upper = rc.sampling_bounds(chain)
    rng = np.random.default_rng(rc.DEFAULT_SEED)
    q48 = rng.uniform(lower, upper, size=(3000, chain.dof))
    q, waist, positions = rc.sample_configurations(chain, 3000, rc.DEFAULT_SEED)
    assert np.array_equal(q, q48) and not np.any(waist)
    assert np.array_equal(positions, chain.fk_batch(q48)[0])


def test_zero_range_reproduces_the_committed_metadata_and_fill_on_a_small_sample(
    tmp_path, chain, committed
):
    offset = committed.tip_offset_m  # the finger bases since #66
    explicit = mre.generate(FIXTURE, samples=20_000, seed=committed.seed,
                            waist_range=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)), tip_offset_m=offset)
    default = mre.generate(FIXTURE, samples=20_000, seed=committed.seed, tip_offset_m=offset)
    assert explicit.same_as(default)
    assert explicit.fill_fraction == default.fill_fraction
    assert (explicit.seed, explicit.wrist_margin_deg, explicit.frame, explicit.tip_link) == (
        committed.seed, committed.wrist_margin_deg, committed.frame, committed.tip_link
    )
    assert explicit.joint_source == committed.joint_source
    assert explicit.waist_at_zero and committed.waist_at_zero
    assert explicit.waist_range_deg == committed.waist_range_deg == rc.ZERO_WAIST_DEG
    assert explicit.tip_offset_m == committed.tip_offset_m == rc.FINGER_BASE_OFFSET_M
    # the first draws of the seed are the first configurations of the committed sample
    idx = np.argwhere(explicit.occupied)
    centres = explicit.origin + (idx + 0.5) * explicit.spacing
    assert committed.contains_points(centres).all()
    # a zero-range envelope writes no waist key, and re-saving the committed file is a no-op
    resaved = committed.save(tmp_path / "resaved.npz")
    assert resaved.read_bytes() == COMMITTED.read_bytes()
    with np.load(explicit.save(tmp_path / "small.npz"), allow_pickle=False) as data:
        assert "waist_range_deg" not in data.files and "tip_offset_m" in data.files
    # the palm link origin (#48's point) writes exactly the #48 keys
    origin = mre.generate(FIXTURE, samples=2_000, seed=committed.seed)
    with np.load(origin.save(tmp_path / "origin.npz"), allow_pickle=False) as data:
        assert "waist_range_deg" not in data.files and "tip_offset_m" not in data.files


# --- the committed waist envelope -------------------------------------------------------------


def test_committed_waist_envelope_records_its_range_and_is_small(committed, committed_waist):
    assert COMMITTED_WAIST.stat().st_size < 10_000  # #77: 7,274 B, the issue asks under 10 KB
    env = committed_waist
    assert (env.seed, env.samples) == (rc.DEFAULT_SEED, rc.WAIST_ENVELOPE_SAMPLES)
    assert env.waist_range_deg == rc.clip_waist_range(rc.WAIST_ENVELOPE_RANGE_DEG)
    # #77: forward pitch 0 to the 0.468 rad soft limit (26.81 deg), the planner's and the training
    # command's maximum; #56 and #66 sampled to the 0.52 rad URDF limit (29.79 deg). Clipping does
    # not touch it, so the file records exactly degrees(WAIST_LEAN_MAX_RAD).
    assert env.waist_range_deg == rc.WAIST_ENVELOPE_RANGE_DEG
    assert env.waist_range_deg[2] == (0.0, math.degrees(WAIST_LEAN_MAX_RAD))
    assert env.waist_range_deg[0] == (-20.0, 20.0) and env.waist_range_deg[1] == (0.0, 0.0)
    # the finger-base point of D012 (#66), the same point as the waist-at-zero file
    assert not env.waist_at_zero and env.tip_offset_m == rc.FINGER_BASE_OFFSET_M
    assert (env.frame, env.tip_link, env.joint_source, env.tip_offset_m) == (
        committed.frame, committed.tip_link, committed.joint_source, committed.tip_offset_m
    )
    with np.load(COMMITTED_WAIST, allow_pickle=False) as data:
        assert data["waist_range_deg"].shape == (3, 2)
        np.testing.assert_array_equal(data["tip_offset_m"], rc.FINGER_BASE_OFFSET_M)
    # the lean reaches further forward, and keeps (almost) all of the waist-at-zero voxels
    assert env.bounding_box[1][0] - committed.bounding_box[1][0] >= 0.10
    idx = np.argwhere(committed.occupied)
    centres = committed.origin + (idx + 0.5) * committed.spacing
    assert env.contains_points(centres).mean() > 0.995


def test_waist_file_sampled_points_are_inside(chain, committed_waist):
    _, positions = rc.sample_tip_positions(
        chain,
        5000,
        committed_waist.seed,
        waist_range=committed_waist.waist_range_deg,
        tip_offset_m=committed_waist.tip_offset_m,
    )
    assert committed_waist.contains_points(positions).all()


def test_check_prints_the_waist_ranges_and_the_tip_offset_of_both_files():
    for path, expected in (
        (COMMITTED, "yaw [0, 0], roll [0, 0], pitch [0, 0]; waist at zero"),
        # the waist file's pitch ends at the soft limit since #77 (29.79, the URDF limit, before)
        (COMMITTED_WAIST, "yaw [-20, 20], roll [0, 0], pitch [0, 26.81]; waist sampled"),
    ):
        result = subprocess.run(
            [sys.executable, str(ENVELOPE_SCRIPT), "--check", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "waist ranges (deg): " + expected in result.stdout
        assert "URDF limits yaw +-150, roll +-29.79, pitch +-29.79 deg" in result.stdout
        assert "tip offset (0.0777, -0.0016, 0.0000) m" in result.stdout


def test_script_tip_offset_flag_writes_the_offset(tmp_path):
    out = tmp_path / "t.npz"
    argv = ["--samples", "2000", "--seed", "4", "--tip-offset", "0.0777", "-0.0016", "0",
            "--out", str(out)]
    assert mre.main(argv) == 0
    env = ReachabilityEnvelope.load(out)
    assert env.tip_offset_m == rc.FINGER_BASE_OFFSET_M and env.waist_at_zero
    in_memory = mre.generate(FIXTURE, samples=2000, seed=4, tip_offset_m=rc.FINGER_BASE_OFFSET_M)
    assert env.same_as(in_memory)
    origin = tmp_path / "o.npz"
    assert mre.main(["--samples", "2000", "--seed", "4", "--out", str(origin)]) == 0
    assert ReachabilityEnvelope.load(origin).tip_offset_m == (0.0, 0.0, 0.0)  # the default


def test_script_waist_flags_write_the_clipped_range(tmp_path, capsys):
    out = tmp_path / "w.npz"
    argv = ["--samples", "3000", "--seed", "4", "--waist-yaw-deg", "20",
            "--waist-pitch-max-deg", "45", "--out", str(out)]
    assert mre.main(argv) == 0
    printed = capsys.readouterr().out
    assert "clipped to the URDF limits" in printed
    env = ReachabilityEnvelope.load(out)
    assert env.waist_range_deg == rc.waist_range_deg(20, 0, 0, 45)
    assert env.samples == 3000 and env.seed == 4


# --- the analysis -----------------------------------------------------------------------------


def _spec(**changes):
    spec = rga.episode_specs("S4", "seen", 0.30, 1)[0]
    return spec.__class__(**{**spec.__dict__, **changes})


def test_kitchen_poses_follow_the_stated_assumptions():
    spec = _spec(pelvis_offset_m=(0.0, 0.0), pelvis_yaw_offset_deg=0.0)
    setup = rga.Setup(rack_pose=(0.5, 0.1, 0.3, 0.0))
    plate, slot, edge = rga.kitchen_poses(spec, setup)
    x, y, z = spec.plate_position_m
    d = spec.plate_diameter_m
    np.testing.assert_allclose(plate.position, (x, y - d / 2.0, z + kd.PLATE_HEIGHT_M))
    sx, sy, sz, _ = spec.rack_slot_pose
    np.testing.assert_allclose(slot.position, (0.5 + sx, 0.1 + sy, 0.3 + sz + d))
    np.testing.assert_allclose(edge.position, (x, 0.0, kd.COUNTER_HEIGHT_M))
    centre = rga.kitchen_poses(spec, rga.Setup(grasp_point="centre"))[0]
    np.testing.assert_allclose(centre.position[:2], (x, y))
    # the robot faces +y: a kitchen point straight ahead of the pelvis is pelvis +x
    pelvis = rga.pelvis_pose(spec, setup)
    np.testing.assert_allclose(pelvis.position, (0.0, -setup.standoff_m, rga.PELVIS_HEIGHT_M))
    ahead = to_frame(Pose((0.0, 1.0 - setup.standoff_m, rga.PELVIS_HEIGHT_M)), pelvis)
    np.testing.assert_allclose(ahead.position, (1.0, 0.0, 0.0), atol=1e-12)


def test_pelvis_targets_match_to_frame():
    spec = rga.episode_specs("S4", "unseen", 0.30, 2)[1]
    setup = rga.Setup(standoff_m=0.3)
    targets = rga.pelvis_targets(spec, setup)
    plate, slot, edge = rga.kitchen_poses(spec, setup)
    waypoints = rga.plan_waypoints(plate, slot, edge, rga.PlannerConfig())
    pelvis = rga.pelvis_pose(spec, setup)
    for row, name in zip(targets, rga.WAYPOINTS, strict=True):
        np.testing.assert_allclose(row, to_frame(waypoints[name], pelvis).position, atol=1e-12)


def test_plate_box_sweep_only_shifts_the_plates():
    near = rga.episode_specs("S2", "unseen", 0.20, 5)
    far = rga.episode_specs("S2", "unseen", 0.30, 5)
    for a, b in zip(near, far, strict=True):
        assert b.plate_position_m[1] - a.plate_position_m[1] == pytest.approx(0.10)
        assert a.plate_position_m[0] == b.plate_position_m[0]
        assert (a.plate_diameter_m, a.rack_slot, a.pelvis_offset_m) == (
            b.plate_diameter_m, b.rack_slot, b.pelvis_offset_m
        )


def test_analysis_core_writes_tables_and_a_figure(tmp_path, committed, committed_waist):
    envelopes = {"waist0": committed, "waist": committed_waist}
    analysis = rga.run_analysis(
        envelopes,
        standoffs=(0.25, 0.45),
        plate_ys=(0.25, 0.30),
        max_seeds=3,
        rack_zs=(0.09, 0.7),
        rack_xs=(0.25,),
    )
    paths = rga.write_outputs(analysis, tmp_path, focus_standoff=0.25)
    tables = paths["tables"].read_text(encoding="utf-8")
    for heading in ("### Per waypoint at the current constants", "### Whole episodes per stand-off",
                    "### S2 episodes per plate box centre y", "### S3 slot targets",
                    "### Sensitivity"):
        assert heading in tables
    rows = paths["episodes"].read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("envelope,standoff_m,plate_centre_y_m,scenario,split,part")
    # 2 stand-offs x 2 plate ys x 2 envelopes x (S1 to S4: 3 parts, S0: plate and slot) x 2 splits
    assert len(rows) - 1 == 2 * 2 * 2 * (4 * 3 + 2) * 2
    assert 10_000 < paths["figure"].stat().st_size < 300_000
    # at the 0.45 m stand-off of before D012 the waist-at-zero envelope reaches almost none of the
    # plate targets: none with #48's palm link origin (#52's finding); with the finger-base point
    # the committed files hold since #66, none of the first three seen draws and one of the first
    # three unseen S1 and S4 draws
    pairs = analysis.grid[(0.45, 0.30)]
    expected = {"seen": (0.0, 0.0, 0.0), "unseen": (1 / 3, 0.0, 1 / 3)}
    for split, fractions in expected.items():
        for scenario, fraction in zip(("S1", "S2", "S4"), fractions, strict=True):
            r = pairs[(scenario, split)]
            got = rga.episode_fraction(r.inside["waist0"], rga.PLATE_TARGETS[:3])
            assert got == pytest.approx(fraction), (scenario, split)


def test_the_point_sensitivity_resamples_the_point_the_envelopes_do_not_hold():
    def small(offset):
        return {
            "waist0": mre.generate(FIXTURE, samples=500, seed=1, tip_offset_m=offset),
            "waist": mre.generate(FIXTURE, samples=500, seed=1,
                                  waist_range=rc.WAIST_ENVELOPE_RANGE_DEG, tip_offset_m=offset),
        }

    variant = {"pitch 0 to 15": ((0.0, 0.0), (0.0, 0.0), (0.0, 15.0))}
    # the committed files hold the finger bases since #66: the sensitivity is the palm link origin
    finger = rga.resampled_envelopes(small(rc.FINGER_BASE_OFFSET_M), waist_variants=variant)
    assert set(finger) == {"palm-link point", "pitch 0 to 15"}
    assert all(e.tip_offset_m == (0.0, 0.0, 0.0) for e in finger["palm-link point"].values())
    assert finger["pitch 0 to 15"]["waist"].tip_offset_m == rc.FINGER_BASE_OFFSET_M
    # #56's palm-link envelopes: the sensitivity is the finger bases, as in the committed doc
    origin = rga.resampled_envelopes(small(None), waist_variants={})
    assert set(origin) == {"finger-base point"}
    assert all(
        e.tip_offset_m == rc.FINGER_BASE_OFFSET_M for e in origin["finger-base point"].values()
    )


def test_committed_figure_is_under_300_kb():
    assert FIGURE.exists()
    assert FIGURE.stat().st_size < 300_000
