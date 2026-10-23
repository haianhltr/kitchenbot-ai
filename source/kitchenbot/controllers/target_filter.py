"""Right-arm target slew filter and the world-to-pelvis target pipeline (#48). numpy only.

docs/architecture.md section 7 puts two things between the planner's palm pose target and the
DifferentialIK controller: a slew limit of 0.3 m/s and 1 rad/s, so the IK step is small and the
balance policy sees smooth upper-body motion, and a reachability check, so a target outside the
precomputed envelope is reported as ``ik_infeasible`` instead of showing up as a controller failure.
Section 5 adds the frame: world targets are expressed in the pelvis frame every step, so the arm,
not the legs, compensates body sway.

:class:`TargetSlewFilter` is the slew limit on a :class:`~kitchenbot.geometry.Pose`: each step the
position moves at most ``max_lin_speed * dt`` along the straight line to the target, and the
orientation turns at most ``max_ang_speed * dt`` along the short arc (``quat_slerp`` by the capped
fraction of ``quat_angle``). The two caps are independent, so position and orientation can arrive
at different times.

:class:`ArmTargetPipeline` is what #14 and #16 call once per control step::

    world target --to_frame(pelvis)--> reachability gate --> slew filter --> pelvis-frame target

It returns an :class:`ArmTarget`: the filtered pelvis-frame pose for the IK and ``infeasible``,
which is the planner's ``arm_infeasible`` observation (#43) and the evaluator's ``ik_infeasible``
tag. When the commanded target is outside the envelope the pipeline keeps steering to the last
feasible world target, re-expressed in the current pelvis frame. The gate checks position only (the
envelope has no orientation), with an optional ``margin`` inside the envelope boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kitchenbot.controllers.reachability import ReachabilityEnvelope
from kitchenbot.geometry import Pose, pose_compose, quat_angle, quat_slerp, to_frame

__all__ = [
    "MAX_ANG_SPEED_RPS",
    "MAX_LIN_SPEED_MPS",
    "ArmTarget",
    "ArmTargetPipeline",
    "TargetSlewFilter",
]

#: Slew limits of the arm target, docs/architecture.md section 7.
MAX_LIN_SPEED_MPS = 0.3
MAX_ANG_SPEED_RPS = 1.0

#: A remaining distance (m) or angle (rad) this close to the step cap counts as within it, so
#: float round-off never adds an extra step of 1e-17 m at the end of a move.
_SNAP = 1e-9


class TargetSlewFilter:
    """Rate-limits a pose target: at most ``max_lin_speed`` m/s and ``max_ang_speed`` rad/s."""

    def __init__(
        self, max_lin_speed: float = MAX_LIN_SPEED_MPS, max_ang_speed: float = MAX_ANG_SPEED_RPS
    ) -> None:
        if not (max_lin_speed > 0.0 and max_ang_speed > 0.0):
            raise ValueError("slew limits must be positive")
        self.max_lin_speed = float(max_lin_speed)
        self.max_ang_speed = float(max_ang_speed)
        self._pose: Pose | None = None

    @property
    def pose(self) -> Pose | None:
        """The last output, ``None`` before :meth:`reset`."""
        return self._pose

    def reset(self, pose: Pose) -> None:
        """Start from ``pose``, normally the current palm pose."""
        self._pose = pose

    def step(self, target: Pose, dt: float) -> Pose:
        """Move the output toward ``target`` by at most one step's worth of each cap."""
        if self._pose is None:
            raise RuntimeError("call reset(pose) before step()")
        if not dt >= 0.0:
            raise ValueError(f"dt must be non-negative, got {dt}")
        current = self._pose

        delta = target.position - current.position
        distance = float(np.linalg.norm(delta))
        max_step = self.max_lin_speed * dt
        position_arrives = distance <= max_step + _SNAP

        angle = quat_angle(current.quat_wxyz, target.quat_wxyz)
        max_turn = self.max_ang_speed * dt
        quat_arrives = angle <= max_turn + _SNAP

        if position_arrives and quat_arrives:
            self._pose = target
        else:
            position = (
                target.position
                if position_arrives
                else current.position + delta * (max_step / distance)
            )
            quat = (
                target.quat_wxyz
                if quat_arrives
                else quat_slerp(current.quat_wxyz, target.quat_wxyz, max_turn / angle)
            )
            self._pose = Pose(position, quat)
        return self._pose


@dataclass(frozen=True)
class ArmTarget:
    """One step's output of :class:`ArmTargetPipeline`."""

    #: Filtered palm target in the pelvis frame: what the IK tracks this step.
    pose: Pose
    #: The commanded world target is outside the envelope (``arm_infeasible`` / ``ik_infeasible``).
    infeasible: bool


class ArmTargetPipeline:
    """World target to filtered pelvis-frame target with a reachability gate (section 7).

    ``envelope`` defaults to the committed right-palm envelope
    (:data:`kitchenbot.controllers.reachability.DEFAULT_PATH`); ``margin`` (m) asks the target to
    lie that far inside it.
    """

    def __init__(
        self,
        envelope: ReachabilityEnvelope | None = None,
        max_lin_speed: float = MAX_LIN_SPEED_MPS,
        max_ang_speed: float = MAX_ANG_SPEED_RPS,
        margin: float = 0.0,
    ) -> None:
        if margin < 0.0 or not math.isfinite(margin):
            raise ValueError("margin must be finite and non-negative")
        self.envelope = envelope if envelope is not None else ReachabilityEnvelope.load()
        self.filter = TargetSlewFilter(max_lin_speed, max_ang_speed)
        self.margin = float(margin)
        self._held_world: Pose | None = None

    @property
    def held_target(self) -> Pose | None:
        """The world target being steered to: the last feasible one, or the reset pose."""
        return self._held_world

    def reachable(self, position_in_pelvis) -> bool:
        """The reachability gate on a pelvis-frame position."""
        return self.envelope.margin_contains(position_in_pelvis, self.margin)

    def reset(self, palm_in_pelvis: Pose, pelvis_in_world: Pose) -> None:
        """Start from the current palm pose; until a feasible target arrives, hold it."""
        self.filter.reset(palm_in_pelvis)
        self._held_world = pose_compose(pelvis_in_world, palm_in_pelvis)

    def step(self, world_target: Pose, pelvis_in_world: Pose, dt: float) -> ArmTarget:
        """One control step: transform, gate, slew."""
        if self._held_world is None:
            raise RuntimeError("call reset(palm_in_pelvis, pelvis_in_world) before step()")
        local = to_frame(world_target, pelvis_in_world)
        infeasible = not self.reachable(local.position)
        if infeasible:
            local = to_frame(self._held_world, pelvis_in_world)
        else:
            self._held_world = world_target
        return ArmTarget(self.filter.step(local, dt), infeasible)
