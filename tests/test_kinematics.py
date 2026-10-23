"""URDF joint origins in the dump (#48) and the forward kinematics built on them. No simulator."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kitchenbot.geometry import quat_from_axis_angle, quat_from_yaw, quat_multiply, quat_to_matrix
from kitchenbot.robots import joint_maps as jm
from kitchenbot.robots.joint_limits import ARM_LEFT_LIMITS, ARM_RIGHT_LIMITS, WAIST_LIMITS
from kitchenbot.robots.kinematics import UrdfChain, right_arm_chain, rpy_to_matrix

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "g1_dex3_joints_urdf.json"
SCRIPT = REPO / "scripts" / "smoke" / "dump_urdf_joints.py"

#: Right palm link origin in the pelvis frame at q = 0 (arm and waist), metres, from the fixture.
ZERO_PALM_POSITION = (0.24127, -0.15164, 0.09523)


@pytest.fixture(scope="module")
def dump():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def joints(dump):
    return dump["joints"]


# --- origins in the dump ----------------------------------------------------------------------

ORIGIN_URDF = """<robot name="origins">
  <link name="base"/>
  <link name="a"/>
  <link name="b"/>
  <link name="c"/>
  <joint name="j_full" type="fixed">
    <origin xyz="0.1 -0.2 0.3" rpy="0.5 0 -1.5"/><parent link="base"/><child link="a"/>
  </joint>
  <joint name="j_xyz_only" type="revolute">
    <origin xyz="0 0 1E-2"/><parent link="a"/><child link="b"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="j_none" type="continuous"><parent link="b"/><child link="c"/></joint>
</robot>
"""


def test_dump_records_origins_with_urdf_defaults(tmp_path):
    urdf = tmp_path / "origins.urdf"
    out = tmp_path / "origins.json"
    urdf.write_text(ORIGIN_URDF, encoding="utf-8")
    subprocess.run([sys.executable, str(SCRIPT), str(urdf), str(out)], check=True)
    full, xyz_only, none = json.loads(out.read_text(encoding="utf-8"))["joints"]
    assert (full["origin_xyz"], full["origin_rpy"]) == ([0.1, -0.2, 0.3], [0.5, 0.0, -1.5])
    assert (xyz_only["origin_xyz"], xyz_only["origin_rpy"]) == ([0.0, 0.0, 0.01], [0.0, 0.0, 0.0])
    assert (none["origin_xyz"], none["origin_rpy"]) == ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_fixture_has_an_origin_on_every_joint(joints):
    for joint in joints:
        assert len(joint["origin_xyz"]) == 3 and len(joint["origin_rpy"]) == 3, joint["name"]
    by_name = {j["name"]: j for j in joints}
    # spot checks against the URDF text at the recorded commit
    assert by_name["right_shoulder_pitch_joint"]["origin_xyz"] == [0.0039563, -0.10021, 0.24778]
    assert by_name["right_shoulder_pitch_joint"]["origin_rpy"] == [-0.27931, 5.4949e-05, 0.00019159]
    assert by_name["right_hand_palm_joint"]["origin_xyz"] == [0.0415, -0.003, 0.0]
    assert by_name["waist_roll_joint"]["origin_xyz"] == [-0.0039635, 0.0, 0.044]


# --- arm and waist limits ---------------------------------------------------------------------


def test_arm_and_waist_limits_equal_the_fixture_on_all_fields(joints):
    by_name = {j["name"]: j for j in joints}
    tables = {
        "waist": (WAIST_LIMITS, jm.WAIST),
        "arm_left": (ARM_LEFT_LIMITS, jm.ARM_LEFT),
        "arm_right": (ARM_RIGHT_LIMITS, jm.ARM_RIGHT),
    }
    for group, (table, names) in tables.items():
        assert list(table) == list(names), group
        for name, limit in table.items():
            joint = by_name[name]
            fields = (limit.lower, limit.upper, limit.effort, limit.velocity)
            expected = (joint["lower"], joint["upper"], joint["effort"], joint["velocity"])
            assert fields == expected, name


def test_only_the_shoulder_roll_range_differs_between_the_arms():
    for left, right in zip(jm.ARM_LEFT, jm.ARM_RIGHT, strict=True):
        a, b = ARM_LEFT_LIMITS[left], ARM_RIGHT_LIMITS[right]
        if left == "left_shoulder_roll_joint":
            assert (a.lower, a.upper) == (-b.upper, -b.lower)
        else:
            assert a == b, left


# --- forward kinematics -----------------------------------------------------------------------
# The expected values below are computed independently of kitchenbot.robots.kinematics: plain
# 4x4 homogeneous matrices built from the fixture with their own elementary rotations.


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _origin_transform(joint) -> np.ndarray:
    roll, pitch, yaw = joint["origin_rpy"]
    t = np.eye(4)
    t[:3, :3] = _rz(yaw) @ _ry(pitch) @ _rx(roll)
    t[:3, 3] = joint["origin_xyz"]
    return t


def _chain_by_hand(joints, base, tip):
    by_child = {j["child"]: j for j in joints}
    chain, link = [], tip
    while link != base:
        chain.append(by_child[link])
        link = by_child[link]["parent"]
    return chain[::-1]


@pytest.fixture(scope="module")
def hand_chain(joints):
    return _chain_by_hand(joints, "pelvis", jm.PALM_LINKS["right"])


@pytest.fixture(scope="module")
def chain(joints):
    return right_arm_chain(joints)


def _zero_transforms(hand_chain):
    """Base-frame transform of every joint frame at the zero configuration."""
    out, t = [], np.eye(4)
    for joint in hand_chain:
        t = t @ _origin_transform(joint)
        out.append(t)
    return out


def test_chain_joint_order_matches_the_joint_map(chain, hand_chain):
    names = [j["name"] for j in hand_chain]
    assert names[-1] == "right_hand_palm_joint" and hand_chain[-1]["type"] == "fixed"
    assert chain.joint_names == (*jm.WAIST, *jm.ARM_RIGHT)
    assert chain.active_joints == jm.ARM_RIGHT
    assert chain.held_joints == jm.WAIST
    assert [j.name for j in chain.joints] == names
    assert chain.dof == 7
    lower = [ARM_RIGHT_LIMITS[n].lower for n in jm.ARM_RIGHT]
    upper = [ARM_RIGHT_LIMITS[n].upper for n in jm.ARM_RIGHT]
    np.testing.assert_array_equal(chain.lower, lower)
    np.testing.assert_array_equal(chain.upper, upper)


def test_zero_configuration_palm_equals_the_composed_origins(chain, hand_chain):
    expected = _zero_transforms(hand_chain)[-1]
    position, quat = chain.fk(np.zeros(7))
    np.testing.assert_allclose(position, expected[:3, 3], atol=1e-12)
    np.testing.assert_allclose(quat_to_matrix(quat), expected[:3, :3], atol=1e-12)
    np.testing.assert_allclose(position, ZERO_PALM_POSITION, atol=5e-5)
    # the plain sum of the origin translations misses by about 1.1 cm: the shoulder pitch and
    # roll origins carry -0.279 and +0.279 rad rolls that tilt the roll joint's own offset
    summed = np.sum([j["origin_xyz"] for j in hand_chain], axis=0)
    assert 0.010 < np.linalg.norm(position - summed) < 0.013


@pytest.mark.parametrize("angle", [0.5, -1.0])
def test_one_joint_moves_the_palm_as_a_rigid_rotation_about_its_axis(chain, hand_chain, angle):
    frames = _zero_transforms(hand_chain)
    palm0 = frames[-1][:3, 3]
    order = list(chain.joint_names)
    for joint, frame in zip(hand_chain, frames, strict=True):
        if joint["type"] == "fixed":
            continue
        centre = frame[:3, 3]
        axis = frame[:3, :3] @ np.asarray(joint["axis"], dtype=float)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rot = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)
        expected = centre + rot @ (palm0 - centre)
        q = np.zeros(10)
        q[order.index(joint["name"])] = angle
        position, _ = chain.fk(q[3:], held=q[:3])
        np.testing.assert_allclose(position, expected, atol=1e-12, err_msg=joint["name"])
        # distance to the axis and the position along it are unchanged
        along = np.dot(position - centre, axis)
        assert along == pytest.approx(np.dot(palm0 - centre, axis), abs=1e-12)


def test_fk_and_fk_batch_agree_and_held_waist_defaults_to_zero(chain):
    rng = np.random.default_rng(48)
    q = rng.uniform(chain.lower, chain.upper, size=(20, 7))
    positions, rotations = chain.fk_batch(q)
    zero_waist, _ = chain.fk_batch(q, held=(0.0, 0.0, 0.0))
    np.testing.assert_array_equal(positions, zero_waist)
    for row, pos, rot in zip(q, positions, rotations, strict=True):
        p, quat = chain.fk(row)
        np.testing.assert_allclose(p, pos, atol=1e-12)
        np.testing.assert_allclose(quat_to_matrix(quat), rot, atol=1e-12)
        assert quat[0] >= 0.0
    # waist yaw turns the whole arm about the pelvis z axis (its origin is the pelvis origin)
    turned, _ = chain.fk_batch(q, held=(0.3, 0.0, 0.0))
    rz = quat_to_matrix(quat_from_yaw(0.3))
    np.testing.assert_allclose(turned, positions @ rz.T, atol=1e-12)


def test_chain_rejects_bad_input(joints, chain):
    with pytest.raises(ValueError, match="no chain"):
        UrdfChain.from_joint_dump(joints, "pelvis", "no_such_link")
    with pytest.raises(ValueError, match="no chain"):
        UrdfChain.from_joint_dump(joints, "left_wrist_yaw_link", jm.PALM_LINKS["right"])
    with pytest.raises(ValueError, match="active"):
        UrdfChain.from_joint_dump(joints, "pelvis", "right_hand_palm_link", active=["x_joint"])
    without_origins = [{k: v for k, v in j.items() if not k.startswith("origin_")} for j in joints]
    with pytest.raises(ValueError, match="no origin"):
        right_arm_chain(without_origins)
    with pytest.raises(ValueError):
        chain.fk(np.zeros(6))
    with pytest.raises(ValueError):
        chain.fk(np.zeros(7), held=(0.0, 0.0))


def test_rpy_matches_the_fixed_axis_convention():
    roll, pitch, yaw = 0.3, -0.7, 1.9
    expected = _rz(yaw) @ _ry(pitch) @ _rx(roll)
    np.testing.assert_allclose(rpy_to_matrix((roll, pitch, yaw)), expected, atol=1e-15)
    q = quat_multiply(
        quat_from_axis_angle((0, 0, 1), yaw),
        quat_multiply(
            quat_from_axis_angle((0, 1, 0), pitch), quat_from_axis_angle((1, 0, 0), roll)
        ),
    )
    np.testing.assert_allclose(quat_to_matrix(q), expected, atol=1e-12)
