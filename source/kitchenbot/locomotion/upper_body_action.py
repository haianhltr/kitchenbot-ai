# UNTESTED DRAFT (#46): written against Isaac Lab v3.0.0 source without a simulator; #5 runs and fixes it.  # noqa: E501
"""``UpperBodyFollowAction``: an action term with zero agent dimensions that writes joint targets.

The agent's action vector has no entries for this term (``action_dim == 0``). Instead, once per
environment step ``process_actions`` reads a ``(num_envs, len(joint_names))`` target tensor from a
*source*, and ``apply_actions`` writes it every physics step through the 3.0 target-command path,
``asset.actuators.target_command.set_position_index`` (``actuators/actuator_collection.py:715``).
``data.joint_pos_target`` is deprecated in 3.0 and ``target_command.position`` is a read-only
view; see docs/isaaclab_3_api_notes.md section 3.

The default source is the command term named ``cfg.command_name`` (``UpperBodyCommand`` during
training). Deployment replaces it with the arm and hand controllers by calling ``set_source``, so
the policy sees the same mechanism in both cases (docs/architecture.md section 5).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import MISSING

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from kitchenbot.locomotion.obs_layout import UPPER_BODY_JOINTS


class UpperBodyFollowAction(ActionTerm):
    """Writes position targets for ``cfg.joint_names`` from a source; no agent action."""

    cfg: UpperBodyFollowActionCfg

    def __init__(self, cfg: UpperBodyFollowActionCfg, env):
        super().__init__(cfg, env)
        joint_ids, self._joint_names = self._asset.find_joints(
            list(cfg.joint_names), preserve_order=True, as_proxy=True
        )
        self._joint_ids = joint_ids.torch
        if len(self._joint_names) != len(cfg.joint_names):
            raise ValueError(f"resolved {self._joint_names}, expected {list(cfg.joint_names)}")
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)
        self._targets = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        self._source: Callable[[], torch.Tensor] | None = None
        if cfg.command_name is not None:
            # The command manager is built before the action manager (manager_based_rl_env.py:112).
            command_manager = env.command_manager
            name = cfg.command_name
            self._source = lambda: command_manager.get_command(name)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._targets

    def set_source(self, source: Callable[[], torch.Tensor] | None) -> None:
        """Read targets from ``source()`` from now on; ``None`` holds the last targets."""
        self._source = source

    def process_actions(self, actions: torch.Tensor):
        # ``actions`` is the (num_envs, 0) slice the action manager hands every term.
        if self._source is not None:
            self._targets[:] = self._source()

    def apply_actions(self):
        self._asset.actuators.target_command.set_position_index(
            value=self._targets, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._targets[env_ids] = self._asset.data.default_joint_pos.torch[env_ids][
            :, self._joint_ids
        ]


@configclass
class UpperBodyFollowActionCfg(ActionTermCfg):
    """Configuration of ``UpperBodyFollowAction``."""

    class_type: type = UpperBodyFollowAction

    asset_name: str = MISSING
    """Scene name of the robot articulation."""

    joint_names: list[str] = list(UPPER_BODY_JOINTS)
    """Exact joint names, in the source's column order (``preserve_order=True``)."""

    command_name: str | None = "upper_body"
    """Command term to follow; ``None`` waits for ``set_source``."""
