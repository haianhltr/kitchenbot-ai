# UNTESTED DRAFT (#46): written against Isaac Lab v3.0.0 source without a simulator; #5 runs and fixes it.  # noqa: E501
"""``UpperBodyCommand``: random arm and hand targets while the balance policy trains.

A ``CommandTerm`` (v3.0.0 ``managers/command_manager.py:26``) over the 28 arm and hand joints. The
sampling rules live in ``upper_body_targets`` (numpy, tested on CI); this term converts tensors at
the boundary. Isaac Lab references are in docs/isaaclab_3_api_notes.md section 4.

Per environment the term keeps the previous target, the next target, the time spent in the
current segment and the segment length. On every resample it starts a new segment from the current
command to a fresh target; ``_update_command`` interpolates linearly. Segment length is the
``time_left`` the base class has just drawn from ``resampling_time_range`` (``_resample`` sets it
before calling ``_resample_command``), so the command arrives at each target exactly when the next
one is drawn. Reach episodes (``is_reach``) draw one right-arm target at the episode start and
ignore later resamples; the interpolation fraction clips at 1, so the arm then holds.

Limits are the articulation's soft joint limits (``ArticulationCfg.soft_joint_pos_limit_factor``,
0.9 for the G1), the default pose is ``default_joint_pos``; both are read from environment 0.
``range_factor`` is set by the curriculum term in ``curriculums.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from kitchenbot.locomotion import upper_body_targets as ubt


class UpperBodyCommand(CommandTerm):
    """Upper-body joint targets, shape ``(num_envs, 28)``, in ``cfg.joint_names`` order."""

    cfg: UpperBodyCommandCfg

    def __init__(self, cfg: UpperBodyCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        joint_ids, self.joint_names = self.robot.find_joints(
            list(cfg.joint_names), preserve_order=True, as_proxy=True
        )
        self.joint_ids = joint_ids.torch
        if len(self.joint_names) != len(cfg.joint_names):
            raise ValueError(f"resolved {self.joint_names}, expected {list(cfg.joint_names)}")

        limits = self.robot.data.soft_joint_pos_limits.torch[0, self.joint_ids]
        self._lower = limits[:, 0].cpu().numpy().astype(float)
        self._upper = limits[:, 1].cpu().numpy().astype(float)
        self._default = self.robot.data.default_joint_pos.torch[0, self.joint_ids].cpu().numpy()
        self._rng = np.random.default_rng(env.cfg.seed if cfg.seed is None else cfg.seed)

        self.range_factor: float = cfg.initial_range_factor
        num = self.num_envs
        default = torch.as_tensor(self._default, dtype=torch.float, device=self.device)
        self._command = default.repeat(num, 1)
        self._prev = self._command.clone()
        self._next = self._command.clone()
        self._elapsed = torch.zeros(num, device=self.device)
        self._duration = torch.ones(num, device=self.device)
        self.is_reach = torch.zeros(num, dtype=torch.bool, device=self.device)
        self.metrics["target_error"] = torch.zeros(num, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _to_device(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float, device=self.device)

    def _resample_command(self, env_ids: Sequence[int] | slice):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if ids.numel() == 0:
            return
        starting = self.command_counter[ids] == 0  # reset() zeroes the counter before resampling

        # -- episode start: choose reach or random, start from the current joint positions
        start_ids = ids[starting]
        if start_ids.numel() > 0:
            n = int(start_ids.numel())
            reach = ubt.sample_reach_episodes(self._rng, n, self.cfg.reach_fraction)
            self.is_reach[start_ids] = self._to_device(reach).bool()
            joint_pos = self.robot.data.joint_pos.torch[start_ids][:, self.joint_ids]
            self._command[start_ids] = joint_pos
            reach_ids = start_ids[self.is_reach[start_ids]]
            if reach_ids.numel() > 0:
                k = int(reach_ids.numel())
                target = ubt.sample_reach_targets(
                    self._lower, self._upper, self._default, self._rng, k
                )
                self._prev[reach_ids] = self._command[reach_ids]
                self._next[reach_ids] = self._to_device(target)
                durations = ubt.sample_durations(self._rng, k, self.cfg.reach_duration_range_s)
                self._duration[reach_ids] = self._to_device(durations)
                self._elapsed[reach_ids] = 0.0

        # -- random segments: every resampled env that is not running a reach
        rand_ids = ids[~self.is_reach[ids]]
        if rand_ids.numel() > 0:
            k = int(rand_ids.numel())
            target = ubt.sample_targets(
                self._lower, self._upper, self._default, self.range_factor, self._rng, k
            )
            self._prev[rand_ids] = self._command[rand_ids]
            self._next[rand_ids] = self._to_device(target)
            self._duration[rand_ids] = self.time_left[rand_ids].clamp_min(1e-3)
            self._elapsed[rand_ids] = 0.0

    def _update_command(self):
        self._elapsed += self._env.step_dt
        self._command[:] = ubt.interpolate(self._prev, self._next, self._elapsed, self._duration)

    def _update_metrics(self):
        joint_pos = self.robot.data.joint_pos.torch[:, self.joint_ids]
        self.metrics["target_error"] = torch.linalg.norm(joint_pos - self._command, dim=-1)


@configclass
class UpperBodyCommandCfg(CommandTermCfg):
    """Configuration of ``UpperBodyCommand``."""

    class_type: type = UpperBodyCommand

    asset_name: str = MISSING
    """Scene name of the robot articulation."""

    joint_names: list[str] = list(ubt.UPPER_BODY_JOINTS)
    """Exact joint names, in command order (``preserve_order=True``)."""

    resampling_time_range: tuple[float, float] = ubt.RESAMPLE_TIME_RANGE_S
    """Seconds between random targets."""

    initial_range_factor: float = ubt.RANGE_FACTOR_START
    """Range factor before the curriculum moves it."""

    reach_fraction: float = ubt.REACH_EPISODE_FRACTION
    """Fraction of episodes that run the slow right-arm reach."""

    reach_duration_range_s: tuple[float, float] = ubt.REACH_DURATION_RANGE_S
    """Seconds the reach takes to arrive."""

    seed: int | None = None
    """Seed of the numpy generator; ``None`` uses the environment's seed."""
