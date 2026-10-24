"""Plate-to-rack planner against a fake arm, hand and plate (docs/architecture.md section 9)."""

import dataclasses
import itertools
import math
import subprocess
import sys

import numpy as np
import pytest
from planner_helpers import (
    COUNTER_EDGE,
    PALM_START,
    PLATE_GRASP,
    SLOT_ENTRY,
    FakeArm,
    FakeHand,
    FakeRack,
    Fakes,
    run_episode,
)

from kitchenbot.evaluation.failure_tags import FailureTag
from kitchenbot.planning.geometry import Pose, plate_vertical_quat
from kitchenbot.planning.plate_to_rack import (
    HandAction,
    PlannerCommand,
    PlannerConfig,
    PlannerObservation,
    PlannerState,
    PlateToRackPlanner,
)

S = PlannerState
NOMINAL_ORDER = [
    S.IDLE, S.PRE_GRASP, S.GRASP, S.LIFT, S.TRANSPORT, S.ALIGN, S.INSERT, S.RELEASE, S.DONE
]


# ----------------------------------------------------------------------------- nominal run


def test_nominal_run_visits_every_state_in_order_and_finishes_under_30_s():
    log = run_episode(hz=50)
    assert log.states == NOMINAL_ORDER
    assert log.final.state is S.DONE and log.final.failure is None
    assert log.planner.failure is None and log.planner.failed_in is None
    assert log.entered_at(S.DONE) < 30.0
    times = [t for t, _ in log.visited]
    assert times == sorted(times)
    # every state finished inside its own provisional timeout
    timeouts = PlannerConfig().state_timeouts()
    for (t0, state), (t1, _) in itertools.pairwise(log.visited):
        assert t1 - t0 <= timeouts[state], state


def test_nominal_hand_verbs_and_arm_targets():
    log = run_episode()
    verbs = [k for k, _ in itertools.groupby(c.hand for c in log.commands)]
    # idle ends on the first step, so the first command issued is already pre_grasp
    assert verbs == [HandAction.PRE_GRASP, HandAction.CLOSE, HandAction.OPEN]
    by_state = {}
    for c in log.commands:
        by_state.setdefault(c.state, []).append(c)
    # pre_grasp shapes the hand above the rim; grasp closes only once the palm is at the rim
    assert {c.hand for c in by_state[S.PRE_GRASP]} == {HandAction.PRE_GRASP}
    assert by_state[S.GRASP][0].hand is HandAction.PRE_GRASP
    assert by_state[S.GRASP][-1].hand is HandAction.CLOSE
    # release opens first and retreats 10 cm up once the hand has let go
    wp = log.planner.waypoints
    assert by_state[S.RELEASE][0].arm_target == wp["release"]
    assert by_state[S.RELEASE][-1].arm_target == wp["retreat"]
    np.testing.assert_allclose(wp["retreat"].position - wp["release"].position, [0, 0, 0.10])
    assert log.final.arm_target == wp["retreat"] and log.final.hand is HandAction.OPEN


def test_waypoints_come_from_the_plate_slot_and_counter_edge_poses():
    planner = PlateToRackPlanner()
    planner.reset(Fakes().observe())
    wp = planner.waypoints
    up = np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(wp["pre_grasp"].position, PLATE_GRASP.position + 0.08 * up)
    assert wp["grasp"] == PLATE_GRASP
    # 15 cm plus one 2 cm position tolerance: the lift state exits within the tolerance (#60)
    np.testing.assert_allclose(wp["lift"].position, PLATE_GRASP.position + (0.15 + 0.02) * up)
    # counter edge: its x, y; high enough for both the lift and the edge clearance
    np.testing.assert_allclose(wp["counter_edge"].position[:2], COUNTER_EDGE.position[:2])
    assert wp["counter_edge"].position[2] == pytest.approx(
        max(PLATE_GRASP.position[2] + 0.15 + 0.02, COUNTER_EDGE.position[2] + 0.15)
    )
    np.testing.assert_allclose(wp["above_slot"].position, SLOT_ENTRY.position + 0.25 * up)
    np.testing.assert_allclose(wp["aligned"].position, wp["above_slot"].position)
    np.testing.assert_allclose(wp["insertion"].position, SLOT_ENTRY.position - 0.05 * up)
    # carried flat until align, vertical from the slot yaw afterwards
    for name in ("pre_grasp", "lift", "counter_edge", "above_slot"):
        np.testing.assert_allclose(wp[name].quat_wxyz, PLATE_GRASP.quat_wxyz)
    vertical = plate_vertical_quat(math.radians(90.0))
    for name in ("aligned", "insertion"):
        np.testing.assert_allclose(wp[name].quat_wxyz, vertical, atol=1e-12)


def test_insert_ends_on_slot_contact_before_the_full_depth():
    fakes = Fakes(rack=FakeRack(floor_depth_m=0.01))  # slot floor 1 cm below the entry
    log = run_episode(fakes=fakes)
    assert log.states == NOMINAL_ORDER
    wp, cfg = log.planner.waypoints, log.planner.config
    gap = wp["release"].position[2] - wp["insertion"].position[2]
    assert gap > cfg.position_tolerance_m  # left insert on contact, not on depth
    assert fakes.rack.in_slot(fakes.plate.pose.position)


# ----------------------------------------------------------------------------- failure tags


FAILURE_CASES = {
    # slot out of the arm's reach: the above-slot target is 0.64 m from the shoulder
    "ik_infeasible": (lambda: Fakes(arm=FakeArm(PALM_START, reach_radius=0.6)), S.TRANSPORT),
    # the plate is 5 cm from where the planner sees it, so the hand closes on air
    "grasp_miss": (lambda: Fakes(plate_observation_offset=(0.05, 0.0, 0.0)), S.GRASP),
    # the plate slips out 1 s after the grasp, during transport
    "drop": (lambda: Fakes(hand=FakeHand(slip_after_s=1.0)), S.TRANSPORT),
    # the observed slot is 5 cm to the side of the real one: the plate lands on a tine (50 N)
    "rack_collision": (lambda: Fakes(slot_observation_offset=(0.05, 0.0, 0.0)), S.INSERT),
    # the observed slot is 6 cm too high: the plate is released above the slot volume
    "misplace": (lambda: Fakes(slot_observation_offset=(0.0, 0.0, 0.06)), S.RELEASE),
    # a slow wrist cannot turn the plate vertical within the align timeout
    "timeout": (lambda: Fakes(arm=FakeArm(PALM_START, angular_speed=0.2)), S.ALIGN),
}


def test_every_failure_tag_except_success_and_fall_has_a_case():
    expected = {t.value for t in FailureTag} - {"success", "fall"}
    assert set(FAILURE_CASES) == expected


@pytest.mark.parametrize("tag", sorted(FAILURE_CASES))
def test_each_failure_triggers_exactly_its_tag(tag):
    make_fakes, failing_state = FAILURE_CASES[tag]
    log = run_episode(fakes=make_fakes())
    assert log.final.state is S.FAILED
    assert log.final.failure is FailureTag(tag)
    assert log.planner.failure is FailureTag(tag)
    assert log.planner.failed_in is failing_state
    assert log.states[-2:] == [failing_state, S.FAILED]
    assert log.states[:-1] == NOMINAL_ORDER[: NOMINAL_ORDER.index(failing_state) + 1]
    # only the very last command carries a failure
    assert all(c.failure is None for c in log.commands[:-1])


def test_total_budget_triggers_timeout_before_any_state_timeout():
    config = dataclasses.replace(PlannerConfig(), total_budget_s=3.0)
    log = run_episode(config=config)
    assert log.final.failure is FailureTag.TIMEOUT
    failed_in, failed_at = log.planner.failed_in, log.entered_at(S.FAILED)
    assert failed_at == pytest.approx(3.0 + 1 / 50)
    assert failed_at - log.entered_at(failed_in) < config.timeout_s(failed_in)


def test_same_step_causes_use_the_section_7_precedence():
    planner = PlateToRackPlanner()
    obs = Fakes().observe()
    planner.reset(obs)
    planner.step(obs, 0.02)  # idle -> pre_grasp
    both = dataclasses.replace(obs, arm_infeasible=True, rack_contact_force=50.0)
    assert planner.step(both, 0.02).failure is FailureTag.IK_INFEASIBLE


def test_terminal_command_is_returned_forever_and_the_state_never_changes():
    make_fakes, _ = FAILURE_CASES["drop"]
    log = run_episode(fakes=make_fakes())
    planner, final = log.planner, log.final
    other = dataclasses.replace(
        Fakes().observe(), hand_holding=True, plate_in_slot=True, arm_infeasible=True
    )
    for _ in range(100):
        assert planner.step(other, 0.5) is final
    assert planner.state is S.FAILED and planner.failure is FailureTag.DROP
    assert planner.time_in_state == 0.0
    # the failed command holds the palm where it was and keeps the hand closed
    assert final.hand is HandAction.CLOSE

    done = run_episode()
    for _ in range(10):
        assert done.planner.step(other, 1.0) is done.final
    assert done.planner.state is S.DONE


# ----------------------------------------------------------------------------- interface


def test_state_one_hot_has_one_hot_bit_and_the_right_length():
    planner = PlateToRackPlanner()
    fakes = Fakes()
    planner.reset(fakes.observe())
    seen = set()
    for _ in range(2000):
        hot = planner.state_one_hot()
        assert len(hot) == len(PlannerState) == 10
        assert all(isinstance(v, float) for v in hot)
        assert sorted(hot) == [0.0] * 9 + [1.0]
        assert hot.index(1.0) == list(PlannerState).index(planner.state)
        assert isinstance(planner.time_in_state, float)
        seen.add(planner.state)
        if planner.state is S.DONE:
            break
        fakes.apply(planner.step(fakes.observe(), 0.02), 0.02)
    assert seen == set(NOMINAL_ORDER)


def test_two_runs_with_the_same_inputs_give_identical_commands():
    a, b = run_episode(), run_episode()
    assert len(a.commands) == len(b.commands) > 100
    assert a.commands == b.commands
    assert a.visited == b.visited
    c, d = (run_episode(fakes=FAILURE_CASES["misplace"][0]()) for _ in "cd")
    assert c.commands == d.commands


def test_hand_action_values_are_lowercase_identifiers():
    assert [a.value for a in HandAction] == ["open", "pre_grasp", "close"]
    for action in HandAction:
        assert action.value.isidentifier() and action.value == action.value.lower()


def test_contract_types():
    assert [s.value for s in PlannerState] == [
        "idle", "pre_grasp", "grasp", "lift", "transport", "align", "insert", "release", "done",
        "failed",
    ]
    assert [f.name for f in dataclasses.fields(PlannerObservation)] == [
        "palm_pose", "plate_pose", "slot_pose", "counter_edge_pose", "hand_holding",
        "rack_contact_force", "arm_infeasible", "plate_in_slot",
    ]
    assert [f.name for f in dataclasses.fields(PlannerCommand)] == [
        "arm_target", "hand", "state", "failure", "waist_pitch_rad",
    ]
    assert [f.name for f in dataclasses.fields(Pose)] == ["position", "quat_wxyz"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        PlannerCommand(Pose((0, 0, 0)), HandAction.OPEN, S.IDLE).hand = HandAction.CLOSE


def test_step_needs_reset_and_a_positive_dt():
    planner = PlateToRackPlanner()
    obs = Fakes().observe()
    with pytest.raises(RuntimeError):
        planner.step(obs, 0.02)
    planner.reset(obs)
    with pytest.raises(ValueError):
        planner.step(obs, 0.0)


# ----------------------------------------------------------------------------- config


def test_config_numbers_equal_the_documented_values():
    cfg = PlannerConfig()
    assert cfg.pre_grasp_height_m == 0.08  # docs/architecture.md section 9
    assert cfg.lift_height_m == 0.15  # section 9, docs/evaluation.md section 3
    assert cfg.above_slot_height_m == 0.25  # section 9
    assert cfg.retreat_height_m == 0.10  # section 9
    assert cfg.rack_collision_force_n == 20.0  # docs/evaluation.md section 3
    assert cfg.vertical_tolerance_deg == 15.0  # docs/evaluation.md section 3
    assert cfg.hold_check_s == 2.0  # docs/evaluation.md section 3
    assert cfg.total_budget_s == 30.0  # docs/evaluation.md section 3


def test_every_config_number_cites_a_source_or_is_provisional():
    for f in dataclasses.fields(PlannerConfig):
        source = f.metadata.get("source", "")
        assert source == "provisional" or source.startswith("docs/"), f.name
        assert f.metadata.get("doc"), f.name
        if f.name.startswith("timeout_"):
            assert source == "provisional", f.name


def test_state_timeouts_cover_the_active_states_and_fit_the_budget():
    cfg = PlannerConfig()
    timeouts = cfg.state_timeouts()
    assert list(timeouts) == NOMINAL_ORDER[:-1]
    assert sum(timeouts.values()) <= cfg.total_budget_s
    with pytest.raises(ValueError):
        cfg.timeout_s(S.DONE)


@pytest.mark.parametrize(
    "changes",
    [{"lift_height_m": 0.0}, {"seat_contact_force_n": 25.0}, {"timeout_release_s": 1.0}],
)
def test_config_rejects_inconsistent_values(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(PlannerConfig(), **changes)


def test_importing_the_planner_pulls_in_no_simulator():
    code = (
        "import sys\n"
        "import kitchenbot.planning.plate_to_rack, kitchenbot.planning.geometry\n"
        "banned = {'isaaclab', 'isaacsim', 'omni', 'torch', 'carb', 'warp'}\n"
        "hits = sorted(m for m in sys.modules if m.split('.')[0] in banned)\n"
        "assert not hits, hits\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
