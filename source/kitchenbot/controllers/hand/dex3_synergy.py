"""Dex3-1 synergy hand: three stored postures, timed interpolation, contact-confirmed holding.

Pure logic, no simulator. Sensing and actuation are injected as callables so the Isaac Lab adapter
of #18 is a thin wrapper around this class:

- ``fingertip_contacts() -> int``: fingertips touching the plate (fingertip ``ContactSensorCfg``
  filtered on the plate, in #18).
- ``plate_tracks_palm() -> bool``: the plate's velocity follows the palm's.
- ``write_targets(targets)``: receives the 7 joint position targets in ``joint_maps.HAND_<SIDE>``
  order, in radians, on every ``update``; #18 routes them to the hand's action term.

Every posture change (``open``, ``pre_grasp``, ``close``) interpolates linearly from the current
targets to the new posture's targets (``dex3_postures.posture_targets``) over ``close_duration``
seconds of ``update`` time. A change while moving starts from where the targets are, so targets
never jump. ``is_holding`` is true only when the posture is closed, the interpolation has finished,
at least ``min_contacts`` fingertips touch the plate and the plate tracks the palm
(docs/architecture.md section 8).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from kitchenbot.controllers.hand.base import HandController, HandPosture
from kitchenbot.controllers.hand.dex3_postures import ROLES, joint_name, posture_targets
from kitchenbot.robots.joint_limits import DEX3_LIMITS

#: Slack when comparing accumulated time with ``close_duration``, so that 25 steps of 0.02 s count
#: as 0.5 s despite floating-point rounding.
TIME_EPS = 1e-9


class Dex3SynergyHand(HandController):
    """Synergy controller for one Dex3-1 hand. Starts at ``initial_posture``, already reached."""

    def __init__(
        self,
        side: str,
        *,
        fingertip_contacts: Callable[[], int],
        plate_tracks_palm: Callable[[], bool],
        write_targets: Callable[[Sequence[float]], None],
        close_duration: float = 0.5,
        min_contacts: int = 2,
        initial_posture: HandPosture | str = HandPosture.OPEN,
    ) -> None:
        if close_duration <= 0.0:
            raise ValueError(f"close_duration must be positive, got {close_duration!r}")
        if min_contacts < 1:
            raise ValueError(f"min_contacts must be at least 1, got {min_contacts!r}")
        self._side = side
        self._fingertip_contacts = fingertip_contacts
        self._plate_tracks_palm = plate_tracks_palm
        self._write_targets = write_targets
        self._close_duration = float(close_duration)
        self._min_contacts = min_contacts
        self._limits = tuple(DEX3_LIMITS[joint_name(side, role)] for role in ROLES)
        self._posture = HandPosture(initial_posture)
        self._goal = posture_targets(side, self._posture)
        self._start = self._goal
        self._targets = self._goal
        self._elapsed = self._close_duration

    @property
    def side(self) -> str:
        return self._side

    @property
    def close_duration(self) -> float:
        return self._close_duration

    @property
    def posture(self) -> HandPosture:
        return self._posture

    @property
    def targets(self) -> tuple[float, ...]:
        """The joint targets last computed, in radians, ``joint_maps.HAND_<SIDE>`` order."""
        return self._targets

    @property
    def goal(self) -> tuple[float, ...]:
        """The targets of the commanded posture, reached when the motion finishes."""
        return self._goal

    @property
    def motion_finished(self) -> bool:
        """True once ``close_duration`` of ``update`` time has passed since the last change."""
        return self._elapsed >= self._close_duration - TIME_EPS

    def open(self) -> None:
        self._command(HandPosture.OPEN)

    def pre_grasp(self) -> None:
        self._command(HandPosture.PRE_GRASP)

    def close(self) -> None:
        self._command(HandPosture.CLOSED)

    def is_holding(self) -> bool:
        return (
            self._posture is HandPosture.CLOSED
            and self.motion_finished
            and self._fingertip_contacts() >= self._min_contacts
            and bool(self._plate_tracks_palm())
        )

    def update(self, dt: float) -> None:
        if dt < 0.0:
            raise ValueError(f"dt must be non-negative, got {dt!r}")
        self._elapsed = min(self._elapsed + dt, self._close_duration)
        if self.motion_finished:
            self._elapsed = self._close_duration
            self._targets = self._goal
        else:
            alpha = self._elapsed / self._close_duration
            self._targets = tuple(
                limit.clamp(a + alpha * (b - a))
                for a, b, limit in zip(self._start, self._goal, self._limits, strict=True)
            )
        self._write_targets(self._targets)

    def _command(self, posture: HandPosture) -> None:
        if posture is self._posture:
            return  # repeating the command in force does not restart it
        self._posture = posture
        self._start = self._targets
        self._goal = posture_targets(self._side, posture)
        self._elapsed = 0.0
