"""The HandController contract (#42) and MockHand. No simulator needed."""

import subprocess
import sys

import pytest

from kitchenbot.controllers.hand.base import HandController, HandPosture
from kitchenbot.controllers.hand.dex3_synergy import Dex3SynergyHand
from kitchenbot.controllers.hand.mock import MockHand

VERBS = {"open": HandPosture.OPEN, "pre_grasp": HandPosture.PRE_GRASP, "close": HandPosture.CLOSED}


def make_synergy() -> Dex3SynergyHand:
    return Dex3SynergyHand(
        "right",
        fingertip_contacts=lambda: 0,
        plate_tracks_palm=lambda: False,
        write_targets=lambda targets: None,
    )


IMPLEMENTATIONS = {"dex3_synergy": make_synergy, "mock": MockHand}


def test_posture_values_are_the_config_strings():
    assert [p.value for p in HandPosture] == ["open", "pre_grasp", "closed"]
    assert HandPosture("pre_grasp") is HandPosture.PRE_GRASP
    assert HandPosture.CLOSED == "closed"
    with pytest.raises(ValueError):
        HandPosture("close")


def test_hand_controller_cannot_be_instantiated():
    with pytest.raises(TypeError):
        HandController()


def test_subclass_missing_a_method_cannot_be_instantiated():
    class NoUpdate(HandController):
        posture = HandPosture.OPEN

        def open(self):
            pass

        def pre_grasp(self):
            pass

        def close(self):
            pass

        def is_holding(self):
            return False

    with pytest.raises(TypeError, match="update"):
        NoUpdate()


@pytest.mark.parametrize("make", IMPLEMENTATIONS.values(), ids=IMPLEMENTATIONS.keys())
def test_implementations_expose_the_contract(make):
    hand = make()
    assert isinstance(hand, HandController)
    for name in ("open", "pre_grasp", "close", "is_holding", "update"):
        assert callable(getattr(hand, name)), name
    assert hand.posture is HandPosture.OPEN
    assert hand.is_holding() is False
    hand.update(0.02)
    with pytest.raises(AttributeError):
        hand.posture = HandPosture.CLOSED


@pytest.mark.parametrize("make", IMPLEMENTATIONS.values(), ids=IMPLEMENTATIONS.keys())
def test_planner_verbs_map_to_methods_and_postures(make):
    hand = make()
    for verb, posture in VERBS.items():
        getattr(hand, verb)()  # the composite env calls the method named by the planner's verb
        assert hand.posture is posture


@pytest.mark.parametrize("make", IMPLEMENTATIONS.values(), ids=IMPLEMENTATIONS.keys())
def test_negative_dt_is_rejected(make):
    with pytest.raises(ValueError):
        make().update(-0.01)


def test_new_modules_import_without_simulator_packages():
    code = (
        "import sys\n"
        "import kitchenbot.controllers.hand.base, kitchenbot.controllers.hand.dex3_postures\n"
        "import kitchenbot.controllers.hand.dex3_synergy, kitchenbot.controllers.hand.mock\n"
        "import kitchenbot.robots.joint_limits\n"
        "bad = ('isaaclab', 'isaacsim', 'omni', 'torch', 'pxr')\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert out.stdout.strip() == "[]"


# --- MockHand ---------------------------------------------------------------------------------


def run(hand: MockHand, seconds: float, dt: float = 0.02) -> None:
    for _ in range(round(seconds / dt)):
        hand.update(dt)


def test_mock_holds_exactly_after_close_duration_when_plate_in_reach():
    hand = MockHand(plate_in_reach=True, close_duration=0.5)
    hand.close()
    assert not hand.is_holding()
    run(hand, 0.48)  # 24 steps of 0.02 s
    assert not hand.is_holding()
    hand.update(0.02)  # 25th step: 0.5 s
    assert hand.is_holding()


def test_mock_never_holds_without_plate_in_reach():
    hand = MockHand(plate_in_reach=False)
    hand.close()
    run(hand, 5.0)
    assert not hand.is_holding()


def test_mock_open_and_pre_grasp_clear_holding_at_once():
    for release in ("open", "pre_grasp"):
        hand = MockHand()
        hand.close()
        run(hand, 0.5)
        assert hand.is_holding()
        getattr(hand, release)()
        assert not hand.is_holding()
        hand.close()  # re-closing restarts the timer
        assert not hand.is_holding()
        run(hand, 0.5)
        assert hand.is_holding()


def test_mock_repeated_close_does_not_restart_the_timer():
    hand = MockHand()
    for _ in range(25):
        hand.close()  # a caller may send its verb every step
        hand.update(0.02)
    assert hand.is_holding()
    assert hand.commands == ["close"] * 25


def test_mock_time_before_close_does_not_count():
    hand = MockHand()
    run(hand, 2.0)
    hand.close()
    assert not hand.is_holding()


def test_mock_drop_by_clearing_plate_in_reach():
    hand = MockHand()
    hand.close()
    run(hand, 0.5)
    assert hand.is_holding()
    hand.plate_in_reach = False
    assert not hand.is_holding()


def test_mock_records_verbs():
    hand = MockHand()
    hand.pre_grasp()
    hand.close()
    hand.open()
    assert hand.commands == ["pre_grasp", "close", "open"]
