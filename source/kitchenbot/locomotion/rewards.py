# UNTESTED DRAFT (#46): written against Isaac Lab v3.0.0 source without a simulator; #5 runs and fixes it.  # noqa: E501
"""Reward terms specific to standing with a moving upper body (docs/architecture.md section 6).

Every other reward of the stand environment is reused from ``isaaclab.envs.mdp`` or the velocity
task's ``mdp`` (docs/isaaclab_3_api_notes.md section 6). Data buffers are 3.0 ``ProxyArray``s,
read with ``.torch``.
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

#: Default selection for ``waist_deviation_l1``: the three waist joints.
WAIST_JOINTS = SceneEntityCfg("robot", joint_names="waist_.*_joint")


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


def waist_deviation_l1(env, asset_cfg: SceneEntityCfg = WAIST_JOINTS) -> torch.Tensor:
    """Sum of absolute waist joint angles, measured from zero (not from the default pose).

    Equal to ``mdp.joint_deviation_l1`` while the waist default is zero, as it is in
    ``G1_29DOF_CFG``; kept separate so a non-zero waist default does not change the meaning.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos.torch[:, asset_cfg.joint_ids].abs().sum(dim=1)
