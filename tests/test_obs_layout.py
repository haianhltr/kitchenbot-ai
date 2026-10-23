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
        ("lower_joint_pos", 15),
        ("lower_joint_vel", 15),
        ("last_action", 15),
        ("upper_joint_pos", 28),
        ("upper_joint_vel", 28),
        ("upper_targets", 28),
    ]


def test_totals():
    assert obs_layout.FRAME_DIM == 138
    assert obs_layout.HISTORY_LENGTH == 5
    assert obs_layout.FLAT_DIM == 690


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
    assert obs_layout.flat_slice(4, "upper_targets") == slice(4 * 138 + 110, 4 * 138 + 138)
    assert obs_layout.flat_slice(4, "upper_targets").stop == obs_layout.FLAT_DIM
    with pytest.raises(IndexError):
        obs_layout.flat_slice(5, "base_ang_vel")


def test_rows_are_json_ready():
    rows = obs_layout.as_rows()
    assert sum(row["dim"] for row in rows) == 138
    assert rows[3]["joints"] == list(joint_maps.LEGS + joint_maps.WAIST)


def test_joint_count_must_match_dimension():
    with pytest.raises(ValueError):
        obs_layout.ObsTerm("bad", 2, "x", ("a",))
