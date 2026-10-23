# UNTESTED DRAFT (#46): written against Isaac Lab v3.0.0 source without a simulator; #5 runs and fixes it.  # noqa: E501
"""Curriculum term that widens the upper-body sampling range as the fall rate drops.

Isaac Lab calls curriculum terms at the top of ``_reset_idx`` with the ids of the environments
being reset, before the termination buffers and ``episode_length_buf`` are cleared
(``envs/manager_based_rl_env.py:386`` and ``:423``). So ``termination_manager.terminated[env_ids]``
says which of those episodes ended by falling (every non-time-out termination of the stand env is
a fall) and ``time_outs[env_ids]`` which ran to 20 s. Environments with ``episode_length_buf == 0``
(the very first reset) are skipped. Counts accumulate until a batch of finished episodes (by
default one per environment) is complete; then the smoothed fall rate and the range factor take
one step. The smoothing and the step rule are the tested functions in ``upper_body_targets``.
Isaac Lab references: docs/isaaclab_3_api_notes.md section 6.
"""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.managers import ManagerTermBase

from kitchenbot.locomotion import upper_body_targets as ubt


class UpperBodyRangeCurriculum(ManagerTermBase):
    """Moves ``UpperBodyCommand.range_factor`` from 0.2 toward 1.0; returns it for logging."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.fall_rate = 0.5  # neutral start: two batches with few falls before the first step
        self._ended = 0
        self._fell = 0

    def __call__(
        self,
        env,
        env_ids: Sequence[int] | slice,
        command_name: str = "upper_body",
        episodes_per_update: int | None = None,
        smoothing: float = ubt.FALL_RATE_SMOOTHING,
    ) -> float:
        term = env.command_manager.get_term(command_name)
        ended = env.episode_length_buf[env_ids] > 0
        fell = env.termination_manager.terminated[env_ids] & ended
        self._ended += int(ended.sum())
        self._fell += int(fell.sum())
        # One curriculum step per batch of finished episodes (default: one per environment),
        # not per reset call, which happens almost every step with many environments.
        batch = env.num_envs if episodes_per_update is None else episodes_per_update
        if self._ended >= batch:
            self.fall_rate = ubt.update_fall_rate(
                self.fall_rate, self._fell, self._ended, smoothing
            )
            term.range_factor = ubt.next_range_factor(term.range_factor, self.fall_rate)
            self._ended = self._fell = 0
        return term.range_factor
