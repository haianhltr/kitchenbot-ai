"""A scripted hand for unit tests and for the task planner (#43): no joints, no physics.

``close()`` reports holding once ``close_duration`` seconds of ``update`` time have passed, if
``plate_in_reach`` is true; with it false the hand never holds. ``plate_in_reach`` is a plain
attribute read on every ``is_holding`` call, so a test can set it to false after the grasp to
simulate a drop. ``open()`` and ``pre_grasp()`` clear holding at once. Repeating the command in
force does not restart the timer. Every verb received (``"open"``, ``"pre_grasp"``,
``"close"``) is appended to ``commands``, repeats included.
"""

from __future__ import annotations

from kitchenbot.controllers.hand.base import HandController, HandPosture

#: Slack when comparing accumulated time with ``close_duration`` (floating-point sums of dt).
TIME_EPS = 1e-9


class MockHand(HandController):
    """Holds after ``close_duration`` if ``plate_in_reach``; starts open."""

    def __init__(self, *, plate_in_reach: bool = True, close_duration: float = 0.5) -> None:
        if close_duration < 0.0:
            raise ValueError(f"close_duration must be non-negative, got {close_duration!r}")
        self.plate_in_reach = plate_in_reach
        self.close_duration = float(close_duration)
        self.commands: list[str] = []
        self._posture = HandPosture.OPEN
        self._closed_for = 0.0

    @property
    def posture(self) -> HandPosture:
        return self._posture

    def open(self) -> None:
        self._command("open", HandPosture.OPEN)

    def pre_grasp(self) -> None:
        self._command("pre_grasp", HandPosture.PRE_GRASP)

    def close(self) -> None:
        self._command("close", HandPosture.CLOSED)

    def is_holding(self) -> bool:
        return (
            self._posture is HandPosture.CLOSED
            and self._closed_for >= self.close_duration - TIME_EPS
            and bool(self.plate_in_reach)
        )

    def update(self, dt: float) -> None:
        if dt < 0.0:
            raise ValueError(f"dt must be non-negative, got {dt!r}")
        if self._posture is HandPosture.CLOSED:
            self._closed_for += dt

    def _command(self, verb: str, posture: HandPosture) -> None:
        self.commands.append(verb)
        if posture is self._posture:
            return
        self._posture = posture
        self._closed_for = 0.0
