"""The balance-policy observation layout (#46): section 6 of docs/architecture.md as data."""

import pytest

from kitchenbot.locomotion import obs_layout
from kitchenbot.robots import joint_maps


def test_frame_terms_match_section_6():
    dims = [(t.name, t.dim) for t in obs_layout.FRAME_TERMS]
    assert dims == [
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("velocity_commands", 3),
        ("waist_lean_command", 1),
        ("lower_joint_pos", 15),
        ("lower_joint_vel", 15),
        ("last_action", 15),
        ("upper_joint_pos", 28),
        ("upper_joint_vel", 28),
        ("upper_targets", 28),
    ]


def test_totals_follow_from_the_terms_and_the_joint_groups():
    lower, upper = len(obs_layout.LOWER_BODY_JOINTS), len(obs_layout.UPPER_BODY_JOINTS)
    commands = 3 + 1  # planar velocity, waist lean (D012)
    base = 3 + 3  # angular velocity, projected gravity
    per_joint_terms = 3 * lower + 3 * upper  # position, velocity, last action / target
    assert obs_layout.FRAME_DIM == base + commands + per_joint_terms
    assert obs_layout.HISTORY_LENGTH == 5
    assert obs_layout.FLAT_DIM == obs_layout.HISTORY_LENGTH * obs_layout.FRAME_DIM


def test_the_waist_lean_command_sits_after_the_velocity_command():
    slices = obs_layout.term_slices()
    assert slices["waist_lean_command"] == slice(
        slices["velocity_commands"].stop, slices["velocity_commands"].stop + 1
    )
    assert slices["lower_joint_pos"].start == slices["waist_lean_command"].stop
    term = next(t for t in obs_layout.FRAME_TERMS if t.name == "waist_lean_command")
    assert term.dim == 1 and "waist_lean" in term.source


def test_joint_groups_partition_the_robot():
    lower, upper = obs_layout.LOWER_BODY_JOINTS, obs_layout.UPPER_BODY_JOINTS
    assert lower == joint_maps.LEGS + joint_maps.WAIST
    assert len(lower) == 15 and len(upper) == 28
    assert set(lower) | set(upper) == set(joint_maps.ALL_JOINTS)
    assert not set(lower) & set(upper)


def test_slices_tile_the_frame_and_the_history():
    slices = obs_layout.term_slices()
    covered = [i for s in slices.values() for i in range(s.start, s.stop)]
    assert covered == list(range(obs_layout.FRAME_DIM))
    assert obs_layout.flat_slice(0, "base_ang_vel") == slice(0, 3)
    last = slices["upper_targets"]
    assert last.stop == obs_layout.FRAME_DIM
    newest = obs_layout.HISTORY_LENGTH - 1
    assert obs_layout.flat_slice(newest, "upper_targets") == slice(
        newest * obs_layout.FRAME_DIM + last.start, newest * obs_layout.FRAME_DIM + last.stop
    )
    assert obs_layout.flat_slice(newest, "upper_targets").stop == obs_layout.FLAT_DIM
    lean = slices["waist_lean_command"]
    assert obs_layout.flat_slice(1, "waist_lean_command") == slice(
        obs_layout.FRAME_DIM + lean.start, obs_layout.FRAME_DIM + lean.stop
    )
    with pytest.raises(IndexError):
        obs_layout.flat_slice(obs_layout.HISTORY_LENGTH, "base_ang_vel")


def test_rows_are_json_ready():
    rows = obs_layout.as_rows()
    assert sum(row["dim"] for row in rows) == obs_layout.FRAME_DIM
    lower_pos = next(row for row in rows if row["name"] == "lower_joint_pos")
    assert lower_pos["joints"] == list(joint_maps.LEGS + joint_maps.WAIST)


def test_joint_count_must_match_dimension():
    with pytest.raises(ValueError):
        obs_layout.ObsTerm("bad", 2, "x", ("a",))
