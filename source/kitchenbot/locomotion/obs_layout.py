"""Observation layout of the balance policy, as data (docs/architecture.md section 6; #46).

One frame is the ten terms below, concatenated in this order; the policy sees ``HISTORY_LENGTH``
frames. The environment draft (``stand_env_cfg.py``) builds its observation group from this table,
and #6 writes the same table into the export manifest so the deployment wrapper and a future
real-robot bridge read the same layout.

History is **time-major**: the policy group sets ``history_order="time"`` (Isaac Lab 3.0
``ObservationGroupCfg``, see docs/isaaclab_3_api_notes.md section 5), so the flat vector is frame
0 (oldest) through frame ``HISTORY_LENGTH - 1`` (newest), each frame laid out as ``FRAME_TERMS``.
Joint order inside a term is the order of ``joints``, which is ``robots.joint_maps`` order.

This module imports only the standard library and ``joint_maps``.
"""

from __future__ import annotations

from dataclasses import dataclass

from kitchenbot.robots.joint_maps import ARM_LEFT, ARM_RIGHT, HAND_LEFT, HAND_RIGHT, LEGS, WAIST

#: The 15 joints the balance policy acts on, in action order.
LOWER_BODY_JOINTS: tuple[str, ...] = LEGS + WAIST

#: The 28 arm and hand joints the upper-body command drives, in command order.
UPPER_BODY_JOINTS: tuple[str, ...] = ARM_LEFT + ARM_RIGHT + HAND_LEFT + HAND_RIGHT

#: Frames of history the policy observes.
HISTORY_LENGTH = 5


@dataclass(frozen=True)
class ObsTerm:
    """One observation term: its name in the env config, its width and where it comes from."""

    name: str
    dim: int
    source: str
    joints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.joints and len(self.joints) != self.dim:
            raise ValueError(f"{self.name}: {len(self.joints)} joints for dimension {self.dim}")


#: One frame of the policy observation, in concatenation order.
FRAME_TERMS: tuple[ObsTerm, ...] = (
    ObsTerm("base_ang_vel", 3, "mdp.base_ang_vel (base frame, rad/s)"),
    ObsTerm("projected_gravity", 3, "mdp.projected_gravity (base frame, unit vector)"),
    ObsTerm("velocity_commands", 3, "mdp.generated_commands('base_velocity'): vx, vy, wz"),
    ObsTerm(
        "waist_lean_command", 1, "mdp.generated_commands('waist_lean'): forward pitch, rad"
    ),
    ObsTerm("lower_joint_pos", 15, "mdp.joint_pos_rel (minus default pose)", LOWER_BODY_JOINTS),
    ObsTerm("lower_joint_vel", 15, "mdp.joint_vel_rel", LOWER_BODY_JOINTS),
    ObsTerm("last_action", 15, "mdp.last_action (raw policy action)", LOWER_BODY_JOINTS),
    ObsTerm("upper_joint_pos", 28, "mdp.joint_pos_rel (minus default pose)", UPPER_BODY_JOINTS),
    ObsTerm("upper_joint_vel", 28, "mdp.joint_vel_rel", UPPER_BODY_JOINTS),
    ObsTerm(
        "upper_targets", 28, "mdp.generated_commands('upper_body'), absolute rad", UPPER_BODY_JOINTS
    ),
)

#: Values in one frame: 3 + 3 + 3 + 1 + 15 + 15 + 15 + 28 + 28 + 28.
FRAME_DIM: int = sum(term.dim for term in FRAME_TERMS)

#: Values in the flattened history the actor and critic receive.
FLAT_DIM: int = FRAME_DIM * HISTORY_LENGTH


def term_slices() -> dict[str, slice]:
    """Slice of each term inside one frame, keyed by term name."""
    slices: dict[str, slice] = {}
    start = 0
    for term in FRAME_TERMS:
        slices[term.name] = slice(start, start + term.dim)
        start += term.dim
    return slices


def flat_slice(frame: int, term_name: str) -> slice:
    """Slice of ``term_name`` in frame ``frame`` (0 = oldest) of the flattened history."""
    if not 0 <= frame < HISTORY_LENGTH:
        raise IndexError(f"frame must be in [0, {HISTORY_LENGTH}), got {frame}")
    inner = term_slices()[term_name]
    offset = frame * FRAME_DIM
    return slice(offset + inner.start, offset + inner.stop)


def as_rows() -> list[dict[str, object]]:
    """The frame table as plain rows for a JSON manifest (#6)."""
    return [
        {"name": t.name, "dim": t.dim, "source": t.source, "joints": list(t.joints)}
        for t in FRAME_TERMS
    ]
