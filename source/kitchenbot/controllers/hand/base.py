"""The interface every hand controller implements (docs/architecture.md section 8).

The task planner (#43) emits the verbs ``open``, ``pre_grasp`` and ``close`` as strings; the
composite environment calls the method of that name, then ``update(dt)`` once per control step.
Implementations: ``dex3_synergy.Dex3SynergyHand`` (logic only until #18 wires it to the simulator),
``mock.MockHand`` for tests, and ``KinematicFollowHand`` from #17. This module imports only the
standard library.
"""

from __future__ import annotations

import abc
from enum import StrEnum


class HandPosture(StrEnum):
    """A commanded hand posture. The values are the strings used in configs and logs."""

    OPEN = "open"
    PRE_GRASP = "pre_grasp"
    CLOSED = "closed"


class HandController(abc.ABC):
    """Open, pre-grasp and close a hand, and say whether it holds the plate.

    The three commands only set the goal and return at once; motion happens in ``update(dt)``.
    Repeating the command already in force must not restart it, so a caller may send its verb on
    every step.
    """

    @property
    @abc.abstractmethod
    def posture(self) -> HandPosture:
        """The posture last commanded, whether or not the hand has reached it yet."""

    @abc.abstractmethod
    def open(self) -> None:
        """Command the open posture (release)."""

    @abc.abstractmethod
    def pre_grasp(self) -> None:
        """Command the pre-grasp posture (fingers shaped around the rim, not touching)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Command the closed posture (grasp)."""

    @abc.abstractmethod
    def is_holding(self) -> bool:
        """True if the hand holds the plate now."""

    @abc.abstractmethod
    def update(self, dt: float) -> None:
        """Advance time-based behaviour by ``dt`` seconds; called once per control step."""
