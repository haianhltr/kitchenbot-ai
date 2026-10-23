"""Dex3SynergyHand logic with fake sensing callables (#42). No simulator needed."""

import itertools

import pytest

from kitchenbot.controllers.hand.base import HandPosture
from kitchenbot.controllers.hand.dex3_postures import ROLES, joint_name, posture_targets
from kitchenbot.controllers.hand.dex3_synergy import Dex3SynergyHand
from kitchenbot.robots.joint_limits import DEX3_LIMITS

DT = 0.02  # 50 Hz control step, docs/architecture.md section 5
VERB = {HandPosture.OPEN: "open", HandPosture.PRE_GRASP: "pre_grasp", HandPosture.CLOSED: "close"}


class Fakes:
    """Sensing values the test sets, and every target vector the hand wrote."""

    def __init__(self):
        self.contacts = 0
        self.tracks = False
        self.written: list[tuple[float, ...]] = []

    def hand(self, side="right", **kwargs) -> Dex3SynergyHand:
        return Dex3SynergyHand(
            side,
            fingertip_contacts=lambda: self.contacts,
            plate_tracks_palm=lambda: self.tracks,
            write_targets=lambda targets: self.written.append(tuple(targets)),
            **kwargs,
        )


def assert_inside_limits(side, targets):
    for role, angle in zip(ROLES, targets, strict=True):
        assert DEX3_LIMITS[joint_name(side, role)].contains(angle), (role, angle)


def test_starts_at_the_open_posture_and_writes_every_update():
    fakes = Fakes()
    hand = fakes.hand()
    assert hand.posture is HandPosture.OPEN
    assert hand.targets == posture_targets("right", "open")
    assert hand.motion_finished
    assert fakes.written == []  # nothing is written before the first update
    hand.update(DT)
    hand.update(DT)
    assert fakes.written == [hand.targets, hand.targets]
    assert len(fakes.written[0]) == 7


@pytest.mark.parametrize("side", ["left", "right"])
def test_close_reaches_the_closed_posture_in_exactly_close_duration(side):
    fakes = Fakes()
    hand = fakes.hand(side)
    hand.close()
    assert hand.posture is HandPosture.CLOSED
    assert hand.goal == posture_targets(side, "closed")
    for _ in range(24):  # 0.48 s
        hand.update(DT)
        assert not hand.motion_finished
        assert hand.targets != hand.goal
    hand.update(DT)  # 0.50 s
    assert hand.motion_finished
    assert hand.targets == posture_targets(side, "closed")


def test_interpolation_is_linear():
    fakes = Fakes()
    hand = fakes.hand("left")
    start, goal = posture_targets("left", "open"), posture_targets("left", "closed")
    hand.close()
    pairs = list(zip(start, goal, strict=True))
    hand.update(0.125)
    assert hand.targets == pytest.approx([a + 0.25 * (b - a) for a, b in pairs])
    hand.update(0.125)
    assert hand.targets == pytest.approx([(a + b) / 2 for a, b in pairs])


@pytest.mark.parametrize("duration", [0.2, 0.5, 1.0])
def test_close_duration_sets_the_time_for_every_posture_change(duration):
    fakes = Fakes()
    hand = fakes.hand(close_duration=duration)
    steps = round(duration / DT)
    for command, posture in (("pre_grasp", "pre_grasp"), ("close", "closed"), ("open", "open")):
        getattr(hand, command)()
        for _ in range(steps - 1):
            hand.update(DT)
        assert not hand.motion_finished, command
        hand.update(DT)
        assert hand.motion_finished, command
        assert hand.targets == posture_targets("right", posture)


def test_a_large_step_does_not_overshoot():
    fakes = Fakes()
    hand = fakes.hand()
    hand.close()
    hand.update(10.0)
    assert hand.targets == posture_targets("right", "closed")


def test_repeating_the_command_in_force_does_not_restart():
    fakes = Fakes()
    hand = fakes.hand()
    for _ in range(25):
        hand.close()  # a caller may send its verb every step
        hand.update(DT)
    assert hand.motion_finished


@pytest.mark.parametrize("side", ["left", "right"])
def test_interrupted_motions_never_jump_or_leave_the_limits(side):
    fakes = Fakes()
    hand = fakes.hand(side)
    script = ["close"] * 10 + ["open"] * 7 + ["pre_grasp"] * 30 + ["close"] * 13 + ["open"] * 40
    previous = hand.targets
    for i, verb in enumerate(script):
        if i == 0 or verb != script[i - 1]:
            getattr(hand, verb)()
            assert hand.targets == previous  # a command alone never moves the targets
        hand.update(DT)
        assert_inside_limits(side, hand.targets)
        # one 0.02 s step of a 0.5 s motion moves each joint at most 4 % of its full range
        for role, a, b in zip(ROLES, previous, hand.targets, strict=True):
            limit = DEX3_LIMITS[joint_name(side, role)]
            assert abs(b - a) <= DT / hand.close_duration * (limit.upper - limit.lower) + 1e-12
        previous = hand.targets
    assert hand.targets == posture_targets(side, "open")


TRUTH_TABLE = list(
    itertools.product(
        [HandPosture.OPEN, HandPosture.PRE_GRASP, HandPosture.CLOSED],  # commanded posture
        [False, True],  # interpolation finished
        [0, 1, 2, 3],  # fingertip contacts
        [False, True],  # plate tracks the palm
    )
)


@pytest.mark.parametrize(("posture", "finished", "contacts", "tracks"), TRUTH_TABLE)
def test_is_holding_truth_table(posture, finished, contacts, tracks):
    fakes = Fakes()
    hand = fakes.hand()
    if posture is HandPosture.OPEN:
        hand.close()
        hand.update(0.5)
    getattr(hand, VERB[posture])()
    hand.update(0.5 if finished else 0.3)
    assert hand.motion_finished == finished
    fakes.contacts, fakes.tracks = contacts, tracks
    expected = posture is HandPosture.CLOSED and finished and contacts >= 2 and tracks
    assert hand.is_holding() is expected


def test_min_contacts_is_configurable():
    fakes = Fakes()
    hand = fakes.hand(min_contacts=3)
    hand.close()
    hand.update(0.5)
    fakes.contacts, fakes.tracks = 2, True
    assert not hand.is_holding()
    fakes.contacts = 3
    assert hand.is_holding()


def test_holding_drops_when_contact_is_lost_or_the_hand_opens():
    fakes = Fakes()
    hand = fakes.hand()
    hand.close()
    hand.update(0.5)
    fakes.contacts, fakes.tracks = 3, True
    assert hand.is_holding()
    fakes.tracks = False
    assert not hand.is_holding()
    fakes.tracks = True
    hand.open()
    assert not hand.is_holding()


def test_bad_arguments_are_rejected():
    fakes = Fakes()
    with pytest.raises(ValueError):
        fakes.hand(close_duration=0.0)
    with pytest.raises(ValueError):
        fakes.hand(min_contacts=0)
    with pytest.raises(ValueError):
        fakes.hand("middle")
    with pytest.raises(ValueError):
        fakes.hand().update(-DT)


def test_initial_posture_can_be_given():
    fakes = Fakes()
    hand = fakes.hand("left", initial_posture="pre_grasp")
    assert hand.posture is HandPosture.PRE_GRASP
    assert hand.targets == posture_targets("left", "pre_grasp")
    assert hand.motion_finished
