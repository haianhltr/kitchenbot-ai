"""Right-palm reachability envelope (#48): sampling, the committed file, queries. No simulator."""

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
from kitchenbot.robots import joint_maps as jm
from kitchenbot.robots.kinematics import right_arm_chain

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "g1_dex3_joints_urdf.json"
SCRIPT = REPO / "scripts" / "make_reach_envelope.py"
COMMITTED = REPO / "source" / "kitchenbot" / "controllers" / "data" / "right_arm_envelope.npz"


def _load_script():
    spec = importlib.util.spec_from_file_location("make_reach_envelope", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mre = _load_script()


@pytest.fixture(scope="module")
def chain():
    return right_arm_chain(json.loads(FIXTURE.read_text(encoding="utf-8"))["joints"])


@pytest.fixture(scope="module")
def committed():
    return ReachabilityEnvelope.load(COMMITTED)


def cube_envelope(size=10, spacing=0.02):
    """A solid cube of ``size`` voxels per side with its min corner at the origin."""
    ticks = (np.arange(size) + 0.5) * spacing
    points = np.stack(np.meshgrid(ticks, ticks, ticks, indexing="ij"), axis=-1).reshape(-1, 3)
    return ReachabilityEnvelope.from_points(points, spacing, seed=0, samples=len(points))


# --- the committed file -----------------------------------------------------------------------


def test_committed_envelope_is_small_and_records_its_sampling(committed):
    assert COMMITTED == rc.DEFAULT_PATH
    assert COMMITTED.stat().st_size < 200_000
    assert (committed.seed, committed.samples) == (rc.DEFAULT_SEED, rc.DEFAULT_SAMPLES)
    assert committed.spacing == rc.VOXEL_SIZE_M == 0.02
    assert committed.wrist_margin_deg == rc.WRIST_MARGIN_DEG == 5.0
    assert (committed.frame, committed.tip_link) == ("pelvis", jm.PALM_LINKS["right"])
    assert "073aec51f548add65baf482ea8bb3be4cb81b7ad" in committed.joint_source
    with np.load(COMMITTED, allow_pickle=False) as data:
        assert {"packed", "origin", "spacing", "shape", "seed", "samples"} <= set(data.files)
        assert data["packed"].dtype == np.uint8


def test_committed_file_equals_a_fresh_generation_with_the_recorded_seed(committed):
    fresh = mre.generate(
        FIXTURE,
        samples=committed.samples,
        seed=committed.seed,
        spacing=committed.spacing,
        wrist_margin_deg=committed.wrist_margin_deg,
    )
    assert fresh.same_as(committed)


def test_fk_of_sampled_configurations_is_inside(chain, committed):
    # the first draws of the recorded seed are the first configurations of the committed sample
    q, positions = rc.sample_tip_positions(chain, 5000, committed.seed)
    for row, position in zip(q[:20], positions[:20], strict=True):
        np.testing.assert_allclose(chain.fk(row)[0], position, atol=1e-12)
    assert committed.contains_points(positions).all()
    assert committed.contains(chain.fk(np.zeros(7))[0])
    # configurations the envelope never saw: all but a sliver at the sparse boundary are inside
    _, unseen = rc.sample_tip_positions(chain, 5000, committed.seed + 1)
    assert committed.contains_points(unseen).mean() > 0.995


def test_points_far_away_are_outside(committed):
    for p in [(2.0, 0.0, 0.0), (0.0, -2.0, 0.3), (0.0, 0.0, 2.0), (-1.4, 1.4, 0.3)]:
        assert not committed.contains(p)
        assert not committed.margin_contains(p, 0.05)
    lo, hi = committed.bounding_box
    assert np.all(hi - lo < 1.0)  # a 0.45 m arm cannot span a metre in any axis


def test_sampling_bounds_take_the_wrist_margin_off_both_ends(chain):
    lower, upper = rc.sampling_bounds(chain)
    margin = math.radians(5.0)
    for i, name in enumerate(chain.active_joints):
        shrink = margin if "_wrist_" in name else 0.0
        assert lower[i] == pytest.approx(chain.lower[i] + shrink), name
        assert upper[i] == pytest.approx(chain.upper[i] - shrink), name
    assert sum("_wrist_" in n for n in chain.active_joints) == 3
    q, _ = rc.sample_tip_positions(chain, 1000, 7)
    assert np.all(q >= lower) and np.all(q <= upper)


def test_sampling_is_deterministic(chain):
    a = rc.sample_tip_positions(chain, 300, 11)
    b = rc.sample_tip_positions(chain, 300, 11)
    c = rc.sample_tip_positions(chain, 300, 12)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


# --- the envelope type ------------------------------------------------------------------------


def test_save_and_load_round_trip_with_stable_bytes(tmp_path, committed):
    first = committed.save(tmp_path / "a.npz")
    second = committed.save(tmp_path / "b.npz")
    assert first.read_bytes() == second.read_bytes()
    back = ReachabilityEnvelope.load(first)
    assert back.same_as(committed)
    assert back.occupied.dtype == bool and not back.occupied.flags.writeable


def test_contains_and_margin_contains_on_a_cube():
    env = cube_envelope(size=10, spacing=0.02)  # occupied from 0 to 0.2 m on every axis
    assert env.occupied_count == 1000
    lo, hi = env.bounding_box
    np.testing.assert_allclose(lo, 0.0, atol=1e-12)
    np.testing.assert_allclose(hi, 0.2, atol=1e-12)
    assert env.fill_fraction == pytest.approx(1000 / 12**3)  # one voxel of padding per side
    centre = (0.1, 0.1, 0.1)
    near_face = (0.01, 0.1, 0.1)
    assert env.contains(centre) and env.contains(near_face)
    assert not env.contains((-0.01, 0.1, 0.1)) and not env.contains((0.1, 0.1, 0.25))
    assert env.margin_contains(centre, 0.0) and env.margin_contains(centre, 0.09)
    assert not env.margin_contains(centre, 0.11)
    assert env.margin_contains(near_face, 0.0) and not env.margin_contains(near_face, 0.02)
    assert not env.margin_contains((5.0, 5.0, 5.0), 0.01)
    with pytest.raises(ValueError):
        env.margin_contains(centre, -0.01)
    np.testing.assert_array_equal(
        env.contains_points([centre, (1.0, 1.0, 1.0), near_face]), [True, False, True]
    )


def test_a_hole_breaks_the_margin_but_not_its_neighbours():
    env = cube_envelope(size=10, spacing=0.02)
    grid = env.occupied.copy()
    hole = env.voxel_index(np.array([[0.11, 0.11, 0.11]]))[0]
    grid[tuple(hole)] = False
    holed = ReachabilityEnvelope(grid, env.origin, env.spacing, seed=0, samples=0)
    assert not holed.contains((0.11, 0.11, 0.11))
    assert holed.contains((0.13, 0.11, 0.11))
    assert not holed.margin_contains((0.13, 0.11, 0.11), 0.02)
    assert holed.margin_contains((0.17, 0.11, 0.11), 0.02)


# --- the script -------------------------------------------------------------------------------


def test_check_prints_the_summary_and_plot_writes_a_png(tmp_path):
    png = tmp_path / "envelope.png"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", str(COMMITTED), "--plot", str(png)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "fill fraction" in result.stdout and "bounding box" in result.stdout
    assert f"seed {rc.DEFAULT_SEED}, samples {rc.DEFAULT_SAMPLES}" in result.stdout
    assert png.stat().st_size > 10_000


def test_script_generates_a_small_envelope(tmp_path):
    out = tmp_path / "small.npz"
    assert mre.main(["--samples", "2000", "--seed", "3", "--out", str(out)]) == 0
    env = ReachabilityEnvelope.load(out)
    assert (env.seed, env.samples) == (3, 2000)
    assert 0 < env.occupied_count <= 2000
