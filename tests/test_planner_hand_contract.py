"""The planner's hand verbs drive the #42 ``HandController`` (architecture sections 8 and 9)."""

from planner_helpers import Fakes, run_episode

from kitchenbot.controllers.hand.base import HandController
from kitchenbot.controllers.hand.mock import MockHand
from kitchenbot.evaluation.failure_tags import FailureTag
from kitchenbot.planning.plate_to_rack import HandAction, PlannerConfig, PlannerState


def test_every_hand_action_is_an_abstract_hand_controller_method():
    for action in HandAction:
        assert action.value in HandController.__abstractmethods__, action
        assert callable(getattr(MockHand(), action.value)), action


def _mock_hand_episode(plate_in_reach: bool):
    hand = MockHand(plate_in_reach=plate_in_reach, close_duration=0.5)
    log = run_episode(fakes=Fakes(hand=hand, hand_reach_from_geometry=False))
    return hand, log


def test_nominal_episode_with_the_mock_hand_reaches_done():
    hand, log = _mock_hand_episode(plate_in_reach=True)
    assert log.final.state is PlannerState.DONE and log.final.failure is None
    assert log.entered_at(PlannerState.DONE) <= PlannerConfig().total_budget_s
    first = {verb: hand.commands.index(verb) for verb in ("pre_grasp", "close", "open")}
    assert first["pre_grasp"] < first["close"] < first["open"]


def test_mock_hand_out_of_reach_ends_in_grasp_miss():
    _, log = _mock_hand_episode(plate_in_reach=False)
    assert log.final.state is PlannerState.FAILED
    assert log.final.failure is FailureTag.GRASP_MISS
    assert log.planner.failed_in is PlannerState.GRASP
