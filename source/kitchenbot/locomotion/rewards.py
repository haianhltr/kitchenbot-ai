# UNTESTED DRAFT (#46 and #68): written against Isaac Lab v3.0.0 source without a simulator; #5 runs and fixes it.  # noqa: E501
"""Reward terms specific to standing with a moving upper body (docs/architecture.md section 6).

Every other reward of the stand environment is reused from ``isaaclab.envs.mdp`` or the velocity
task's ``mdp`` (docs/isaaclab_3_api_notes.md section 6). Data buffers are 3.0 ``ProxyArray``s,
read with ``.torch``.
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

from kitchenbot.locomotion import waist_lean_targets as wlt
from kitchenbot.robots.joint_maps import WAIST

#: Default selection for ``waist_tracking_l1``: the three waist joints in ``joint_maps.WAIST``
#: order (yaw, roll, pitch). ``preserve_order=True`` matters: the reward reads the columns by
#: position, and a regex selection would follow the articulation's own order instead.
WAIST_JOINTS = SceneEntityCfg("robot", joint_names=list(WAIST), preserve_order=True)


def feet_planted_when_standing(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    command_threshold: float = 0.06,
) -> torch.Tensor:
    """1 where the velocity command is (near) zero and every selected foot is in contact, else 0.

    Contact is the maximum normal-force norm over the sensor history above ``force_threshold``,
    the same test ``feet_slide`` uses. The command threshold matches Isaac Lab's
    ``stand_still_joint_deviation_l1`` (0.06 m/s on the planar command).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_normal_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > force_threshold
    command = env.command_manager.get_command(command_name)
    standing = torch.linalg.norm(command[:, :2], dim=1) < command_threshold
    standing &= command[:, 2].abs() < command_threshold
    return (in_contact.all(dim=1) & standing).float()


def waist_tracking_l1(
    env, command_name: str, asset_cfg: SceneEntityCfg = WAIST_JOINTS
) -> torch.Tensor:
    """``|pitch - command| + |yaw| + |roll|`` of the waist, in rad (D012, #68).

    Replaces the #46 reward that pulled the whole waist to zero: the pitch now tracks the lean
    command (``WaistLeanCommand``, shape ``(num_envs, 1)``), yaw and roll stay at zero. Angles are
    measured from zero, not from the default pose, which are the same while the waist default is
    zero as in ``G1_29DOF_CFG``. ``asset_cfg`` must select the three waist joints in
    ``joint_maps.WAIST`` order, as ``WAIST_JOINTS`` does. Use as a penalty (negative weight).
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    command = env.command_manager.get_command(command_name)[:, 0]
    return wlt.tracking_error(
        joint_pos[:, wlt.WAIST_PITCH_INDEX],
        joint_pos[:, wlt.WAIST_YAW_INDEX],
        joint_pos[:, wlt.WAIST_ROLL_INDEX],
        command,
    )
