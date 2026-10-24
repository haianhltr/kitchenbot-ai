# UNTESTED DRAFT (#46 and #68): written against Isaac Lab v3.0.0 source without a simulator; #5 runs and fixes it.  # noqa: E501
"""Command terms of the balance-policy training: ``UpperBodyCommand`` and ``WaistLeanCommand``.

``UpperBodyCommand`` gives random arm and hand targets while the balance policy trains. It is a
``CommandTerm`` (v3.0.0 ``managers/command_manager.py:26``) over the 28 arm and hand joints. The
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

``WaistLeanCommand`` (#68, D012) is the forward waist pitch the planner will ask for during
pre-grasp, grasp and lift: one value in radians per environment, observed by the policy and tracked
by ``rewards.waist_tracking_l1``. The rules (targets in ``[0, 0.468]`` rad, up to the joint's soft
limit ``robots.joint_limits.WAIST_LEAN_MAX_RAD`` as D012 was amended, exactly zero half of the
time, a new target every 1 to 3 s, a 1 rad/s ramp instead of a step) and the reason for each are in
``waist_lean_targets`` (numpy, tested on CI); this term converts tensors at the boundary. The
metric follows ``UniformVelocityCommand`` (v3.0.0 ``commands/velocity_command.py:123-156``):
per-step errors are summed and the episode mean is written in ``reset`` before the base class logs
and zeroes the metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass, index_fill_

from kitchenbot.locomotion import upper_body_targets as ubt
from kitchenbot.locomotion import waist_lean_targets as wlt


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


class WaistLeanCommand(CommandTerm):
    """Forward waist pitch command in rad, shape ``(num_envs, 1)``."""

    cfg: WaistLeanCommandCfg

    def __init__(self, cfg: WaistLeanCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        joint_ids, joint_names = self.robot.find_joints(
            [cfg.joint_name], preserve_order=True, as_proxy=True
        )
        if len(joint_names) != 1:
            raise ValueError(f"resolved {joint_names}, expected [{cfg.joint_name!r}]")
        self.joint_ids = joint_ids.torch
        seed = env.cfg.seed if cfg.seed is None else cfg.seed
        self._rng = np.random.default_rng(None if seed is None else [seed, wlt.RNG_STREAM])

        self._command = torch.zeros(self.num_envs, 1, device=self.device)
        self._target = torch.zeros(self.num_envs, 1, device=self.device)
        # -- metrics: finalized per-episode means, written at reset() (see the module docstring)
        self.metrics["pitch_error"] = torch.zeros(self.num_envs, device=self.device)
        self._error_sum = torch.zeros(self.num_envs, device=self.device)
        self._step_count = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        self.metrics["pitch_error"][env_ids] = self._error_sum[env_ids] / self._step_count[
            env_ids
        ].clamp_min(1.0)
        extras = super().reset(env_ids)
        index_fill_(self._error_sum, env_ids, 0.0)
        index_fill_(self._step_count, env_ids, 0.0)
        return extras

    def _resample_command(self, env_ids: Sequence[int] | slice):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if ids.numel() == 0:
            return
        # reset() zeroes the counter before resampling: the episode starts upright, as the reset
        # puts the waist at its default pose
        starting = ids[self.command_counter[ids] == 0]
        self._command[starting] = 0.0
        targets = wlt.sample_lean_targets(
            self._rng, int(ids.numel()), self.cfg.max_lean_rad, self.cfg.zero_share
        )
        self._target[ids] = torch.as_tensor(targets, dtype=torch.float, device=self.device)[:, None]
        if self.cfg.max_rate_rad_s is None:
            self._command[ids] = self._target[ids]

    def _update_command(self):
        if self.cfg.max_rate_rad_s is None:
            self._command[:] = self._target
        else:
            step = self.cfg.max_rate_rad_s * self._env.step_dt
            self._command[:] = wlt.slew_toward(self._command, self._target, step)

    def _update_metrics(self):
        pitch = self.robot.data.joint_pos.torch[:, self.joint_ids]
        self._error_sum += (pitch - self._command).abs()[:, 0]
        self._step_count += 1.0


@configclass
class WaistLeanCommandCfg(CommandTermCfg):
    """Configuration of ``WaistLeanCommand``."""

    class_type: type = WaistLeanCommand

    asset_name: str = MISSING
    """Scene name of the robot articulation."""

    joint_name: str = wlt.WAIST_PITCH_JOINT
    """The joint whose position the metric compares with the command."""

    resampling_time_range: tuple[float, float] = wlt.RESAMPLE_TIME_RANGE_S
    """Seconds between targets."""

    max_lean_rad: float = wlt.MAX_LEAN_RAD
    """Largest target, rad; targets are uniform in ``[0, max_lean_rad]``."""

    zero_share: float = wlt.ZERO_SHARE
    """Probability that a resampled target is exactly zero."""

    max_rate_rad_s: float | None = wlt.MAX_RATE_RAD_S
    """Rate at which the command moves toward its target; ``None`` makes it a step."""

    seed: int | None = None
    """Seed of the numpy generator; ``None`` uses the environment's seed."""
