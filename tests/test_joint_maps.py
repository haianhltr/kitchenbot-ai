"""The provisional joint map (issue #28) against the URDF dump fixture. No simulator needed."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kitchenbot.robots import joint_maps as jm

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "g1_dex3_joints_urdf.json"
SCRIPT = REPO / "scripts" / "smoke" / "dump_urdf_joints.py"

EXPECTED_COUNTS = {
    "legs": 12,
    "waist": 3,
    "arm_left": 7,
    "arm_right": 7,
    "hand_left": 7,
    "hand_right": 7,
}


@pytest.fixture(scope="module")
def dump():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def movable(dump):
    return [j["name"] for j in dump["joints"] if j["movable"]]


def test_fixture_is_small_and_records_its_source(dump):
    assert FIXTURE.stat().st_size < 1_000_000
    source = dump["source"]
    prefix = "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/"
    assert source["url"].startswith(prefix)
    assert len(source["commit"]) == 40
    assert source["licence"] == "BSD-3-Clause"
    assert source["fetched"]


def test_movable_flag_matches_joint_type(dump):
    for joint in dump["joints"]:
        assert joint["movable"] == (joint["type"] in ("revolute", "prismatic", "continuous"))
        if not joint["movable"]:
            assert joint["axis"] is None


def test_counts_per_group():
    assert {name: len(names) for name, names in jm.GROUPS.items()} == EXPECTED_COUNTS
    assert len(jm.ALL_JOINTS) == 43


def test_regexes_partition_the_movable_joints(movable):
    assert len(movable) == 43
    assert len(set(movable)) == len(movable)
    owners = {n: [g for g, rx in jm.GROUP_REGEX.items() if rx.fullmatch(n)] for n in movable}
    overlap = {n: g for n, g in owners.items() if len(g) > 1}
    leftover = [n for n, g in owners.items() if not g]
    assert not overlap, f"matched by more than one regex: {overlap}"
    assert not leftover, f"matched by no regex: {leftover}"
    for group, regex in jm.GROUP_REGEX.items():
        matched = [n for n in movable if regex.fullmatch(n)]
        assert len(matched) == EXPECTED_COUNTS[group], group
        # names and regex agree, and the name list keeps the URDF file order
        assert matched == list(jm.GROUPS[group]), group


def test_name_lists_are_in_the_fixture_without_duplicates(movable):
    assert len(set(jm.ALL_JOINTS)) == len(jm.ALL_JOINTS)
    assert set(jm.ALL_JOINTS) == set(movable)
    for group, names in jm.GROUPS.items():
        assert all(jm.GROUP_REGEX[group].fullmatch(n) for n in names), group


def test_fixed_joints_are_in_no_group(dump):
    fixed = [j["name"] for j in dump["joints"] if not j["movable"]]
    assert "left_hand_palm_joint" in fixed
    assert all(jm.group_of(n) is None for n in fixed)


def test_group_of():
    assert jm.group_of("right_knee_joint") == "legs"
    assert jm.group_of("waist_pitch_joint") == "waist"
    assert jm.group_of("left_hand_index_1_joint") == "hand_left"
    assert jm.group_of("right_wrist_yaw_joint") == "arm_right"
    assert jm.group_of("not_a_joint") is None


def test_arm_and_hand_lists_mirror_each_other():
    assert [n.replace("left_", "right_", 1) for n in jm.ARM_LEFT] == list(jm.ARM_RIGHT)
    assert [n.replace("left_", "right_", 1) for n in jm.HAND_LEFT] == list(jm.HAND_RIGHT)


def test_palm_and_wrist_links_exist(dump):
    links = set(dump["links"])
    assert set(jm.PALM_LINKS) == set(jm.WRIST_LINKS) == {"left", "right"}
    for side in ("left", "right"):
        assert jm.PALM_LINKS[side] in links
        assert jm.WRIST_LINKS[side] in links
        # the palm is fixed to the wrist link, which is the child of the arm's last joint
        palm_joint = next(j for j in dump["joints"] if j["child"] == jm.PALM_LINKS[side])
        assert palm_joint["type"] == "fixed"
        assert palm_joint["parent"] == jm.WRIST_LINKS[side]
        arm_tip = next(j for j in dump["joints"] if j["name"] == f"{side}_wrist_yaw_joint")
        assert arm_tip["child"] == jm.WRIST_LINKS[side]


def test_dex3_hand_joints_have_limits(dump):
    by_name = {j["name"]: j for j in dump["joints"]}
    for name in (*jm.HAND_LEFT, *jm.HAND_RIGHT):
        joint = by_name[name]
        assert joint["type"] == "revolute"
        assert joint["lower"] < joint["upper"]
        assert joint["effort"] > 0
        assert joint["velocity"] > 0


MINI_URDF = """<robot name="mini">
  <!-- <joint name="floating_base_joint" type="floating"><parent link="world"/>
       <child link="base"/></joint> -->
  <link name="base"/>
  <link name="a"/>
  <link name="b"/>
  <link name="c"/>
  <joint name="j_fixed" type="fixed"><parent link="base"/><child link="a"/></joint>
  <joint name="j_rev" type="revolute">
    <parent link="a"/><child link="b"/><axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="2.5" effort="10" velocity="3"/>
  </joint>
  <joint name="j_cont" type="continuous"><parent link="b"/><child link="c"/></joint>
</robot>
"""


def test_dump_script_parses_a_small_urdf(tmp_path):
    urdf = tmp_path / "mini.urdf"
    out = tmp_path / "mini.json"
    urdf.write_text(MINI_URDF, encoding="utf-8")
    subprocess.run([sys.executable, str(SCRIPT), str(urdf), str(out)], check=True)
    dump = json.loads(out.read_text(encoding="utf-8"))
    assert dump["links"] == ["base", "a", "b", "c"]
    assert [j["name"] for j in dump["joints"]] == ["j_fixed", "j_rev", "j_cont"]  # file order
    assert dump["counts"] == {"joints": 3, "movable": 2, "fixed": 1, "links": 4}
    fixed, rev, cont = dump["joints"]
    assert fixed["movable"] is False and fixed["axis"] is None and fixed["lower"] is None
    assert rev["axis"] == [0.0, 0.0, 1.0]
    assert (rev["lower"], rev["upper"], rev["effort"], rev["velocity"]) == (-1.5, 2.5, 10.0, 3.0)
    assert (rev["parent"], rev["child"]) == ("a", "b")
    assert cont["movable"] is True and cont["axis"] == [1.0, 0.0, 0.0]  # URDF default axis
    assert cont["lower"] is None and cont["upper"] is None
    assert "source" not in dump
