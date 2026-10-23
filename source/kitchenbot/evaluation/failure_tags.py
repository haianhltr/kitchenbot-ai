"""Failure taxonomy for evaluation trials (docs/evaluation.md section 7).

Every failed trial is tagged with exactly one primary cause. A trial that succeeded carries
``FailureTag.SUCCESS``. This module has no dependencies outside the standard library.

Tagging precedence
------------------
docs/evaluation.md names the causes but not what to do when several are observed in one trial
(a fall usually also ends in a timeout, a dropped plate may have scraped the rack first). The
precedence used here is the order the section 7 list gives, which is also the order the task
unfolds, with the cause that makes every later stage moot first:

1. ``fall``: balance was lost; nothing after it is meaningful.
2. ``ik_infeasible``: no IK solution within limits, so the reach never happened.
3. ``grasp_miss``: fingers closed and the plate was not held.
4. ``drop``: the plate was lost after the lift.
5. ``rack_collision``: a contact force threshold was hit on the way in.
6. ``misplace``: the plate was released outside the slot volume.
7. ``timeout``: none of the above, the clock simply ran out. This is the catch-all, so it is last.

The enum members are declared in exactly this order and :func:`primary_failure_tag` picks the
member that is declared first among the causes observed.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class FailureTag(StrEnum):
    """Primary cause of a failed trial, or ``SUCCESS``. Values are the strings in the JSON."""

    SUCCESS = "success"
    FALL = "fall"
    IK_INFEASIBLE = "ik_infeasible"
    GRASP_MISS = "grasp_miss"
    DROP = "drop"
    RACK_COLLISION = "rack_collision"
    MISPLACE = "misplace"
    TIMEOUT = "timeout"


#: Failure causes from highest to lowest precedence (everything except ``SUCCESS``).
FAILURE_PRECEDENCE: tuple[FailureTag, ...] = tuple(
    t for t in FailureTag if t is not FailureTag.SUCCESS
)

#: All tag values as they appear in the JSON, ``success`` first.
ALL_TAG_VALUES: tuple[str, ...] = tuple(t.value for t in FailureTag)

#: Failure tag values only, in precedence order.
FAILURE_TAG_VALUES: tuple[str, ...] = tuple(t.value for t in FAILURE_PRECEDENCE)


def primary_failure_tag(observed: Iterable[FailureTag | str]) -> FailureTag:
    """Return the single primary cause among the causes observed in one failed trial.

    ``observed`` may hold ``FailureTag`` members or their string values. ``SUCCESS`` entries are
    ignored. Raises ``ValueError`` if no failure cause is given: a failed trial with no observed
    cause is a bug in the caller, which should pass ``FailureTag.TIMEOUT`` when the episode
    simply ran out of time.
    """
    seen = {FailureTag(t) for t in observed} - {FailureTag.SUCCESS}
    for tag in FAILURE_PRECEDENCE:
        if tag in seen:
            return tag
    raise ValueError("no failure cause observed; pass FailureTag.TIMEOUT if the episode timed out")
