"""The planner's D012 waist lean (#67): a target per state, recorded by the fakes and the dry run.

The lean is ``PlannerConfig.waist_lean_rad`` in ``pre_grasp``, ``grasp`` and ``lift``. It is zero
from ``transport`` (whose first waypoint is the counter edge) on, and in ``idle``, ``failed`` and
``done``. No simulation. The fakes record the planner's target as sent. The 1 rad/s slew before the
balance policy is the deployment layer's job (#21, #68), not the planner's.
"""

import dataclasses

import pytest
from planner_helpers import FakeHand, Fakes, run_episode

from kitchenbot.controllers.reachability import ReachabilityEnvelope
from kitchenbot.evaluation.dry_run import simulate_episode
from kitchenbot.evaluation.failure_tags import FailureTag
from kitchenbot.locomotion import waist_lean_targets as wlt
from kitchenbot.planning.plate_to_rack import (
    LEANING_STATES,
    PlannerConfig,
    PlannerState,
    PlateToRackPlanner,
)
from kitchenbot.robots.joint_limits import WAIST_LIMITS
from kitchenbot.scene import kitchen_dims as kd
from kitchenbot.scene.randomization import sample_episode

S = PlannerState
LAYOUT = kd.RackSlotLayout()
LEAN = 0.52
#: The per-state table of the issue: the commanded waist pitch in every state, radians.
EXPECTED_LEAN = {
    S.IDLE: 0.0,
    S.PRE_GRASP: LEAN,
    S.GRASP: LEAN,
    S.LIFT: LEAN,
    S.TRANSPORT: 0.0,
    S.ALIGN: 0.0,
    S.INSERT: 0.0,
    S.RELEASE: 0.0,
    S.DONE: 0.0,
    S.FAILED: 0.0,
}


def test_the_lean_is_sourced_to_d012_and_inside_the_waist_limit():
    cfg = PlannerConfig()
    assert cfg.waist_lean_rad == LEAN
    field = {f.name: f for f in dataclasses.fields(PlannerConfig)}["waist_lean_rad"]
    assert field.metadata["source"] == "docs/decisions.md D012"
    assert cfg.waist_lean_rad <= WAIST_LIMITS["waist_pitch_joint"].upper  # 0.52 rad, the URDF
    # inside the range the balance policy is trained on (#68), which ramps at the 1 rad/s the
    # deployment layer has to slew the planner's target at
    assert cfg.waist_lean_rad <= wlt.MAX_LEAN_RAD
    assert wlt.MAX_RATE_RAD_S == 1.0
    assert LEANING_STATES == {S.PRE_GRASP, S.GRASP, S.LIFT}
    assert set(EXPECTED_LEAN) == set(PlannerState)


@pytest.mark.parametrize(
    "spec",
    [sample_episode(0, "seen", "S4"), sample_episode(7, "seen", "S4"),
     sample_episode(1000, "unseen", "S4")],
    ids=lambda s: f"{s.split}-{s.seed}",
)
def test_the_waist_command_per_state_across_a_full_dry_episode(spec):
    fakes = Fakes.from_episode(spec, LAYOUT, gate=False)
    log = run_episode(fakes=fakes)
    assert log.final.state is S.DONE
    seen_states = set()
    for command in log.commands:
        assert command.waist_pitch_rad == EXPECTED_LEAN[command.state], command.state
        seen_states.add(command.state)
    assert seen_states == set(EXPECTED_LEAN) - {S.IDLE, S.FAILED}  # idle exits on the first step
    # the fakes record the target of every command they were sent, nothing smoothed
    sent = [(c.state, c.waist_pitch_rad) for c in log.commands[:-1]]  # the last one is not applied
    assert fakes.waist_trace == sent
    # the lean is on from the first command to the last lift command, then off for good
    leaning = [c.waist_pitch_rad > 0.0 for c in log.commands]
    first_off = leaning.index(False)
    assert all(leaning[:first_off]) and not any(leaning[first_off:])
    assert log.commands[first_off].state is S.TRANSPORT
    assert log.commands[first_off - 1].state is S.LIFT


def test_the_lean_steps_at_the_state_changes_and_is_not_ramped_by_the_planner():
    log = run_episode()
    pitches = [c.waist_pitch_rad for c in log.commands]
    assert set(pitches) == {0.0, LEAN}  # two values only: the deployment layer slews between them


def test_a_failure_while_leaning_commands_the_waist_back_to_zero():
    log = run_episode(fakes=Fakes(plate_observation_offset=(0.05, 0.0, 0.0)))  # grasp miss
    assert log.final.failure is FailureTag.GRASP_MISS and log.planner.failed_in is S.GRASP
    assert log.commands[-2].waist_pitch_rad == LEAN
    assert log.final.waist_pitch_rad == 0.0
    for _ in range(3):  # the terminal command is returned forever, still upright
        assert log.planner.step(Fakes().observe(), 0.02).waist_pitch_rad == 0.0


def test_a_drop_during_transport_was_already_upright():
    log = run_episode(fakes=Fakes(hand=FakeHand(slip_after_s=1.0)))
    assert log.final.failure is FailureTag.DROP and log.planner.failed_in is S.TRANSPORT
    assert log.commands[-2].waist_pitch_rad == 0.0 and log.final.waist_pitch_rad == 0.0


def test_a_configured_lean_is_what_the_planner_commands():
    cfg = dataclasses.replace(PlannerConfig(), waist_lean_rad=0.30)
    log = run_episode(planner=PlateToRackPlanner(cfg))
    assert {c.waist_pitch_rad for c in log.commands if c.state in LEANING_STATES} == {0.30}
    with pytest.raises(ValueError):
        dataclasses.replace(PlannerConfig(), waist_lean_rad=0.0)


def test_the_dry_run_records_the_waist_trace_and_its_maximum():
    spec = sample_episode(0, "seen", "S4")
    off = simulate_episode(spec, LAYOUT, gate=False)
    assert off.state is S.DONE and off.max_waist_pitch_rad == LEAN
    assert {pitch for state, pitch in off.waist_trace if state in LEANING_STATES} == {LEAN}
    assert {pitch for state, pitch in off.waist_trace if state not in LEANING_STATES} == {0.0}
    # with the gate on the first reach is refused in pre_grasp, after one leaning command
    on = simulate_episode(spec, LAYOUT, gate=True, envelope=ReachabilityEnvelope.load())
    assert on.failed_in is S.PRE_GRASP and on.max_waist_pitch_rad == LEAN
    assert all(state is S.PRE_GRASP for state, _ in on.waist_trace)
    # nothing about the waist goes into the result record (the schema allows no extra metric)
    assert "waist_pitch_rad" not in str(off.result)
