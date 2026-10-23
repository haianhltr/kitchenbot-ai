# UNTESTED DRAFT (#59): written against Isaac Lab v3.0.0 source without a simulator; #10 and #11 run and fix it.  # noqa: E501
"""ArtVIP ``dishwasher_2`` as an Isaac Lab articulation, tines on its lower rack, slot poses.

Every Isaac Lab symbol below is pinned to a v3.0.0 file and line in docs/isaaclab_3_api_notes.md
section 9, which also holds the checklist #10 and #11 run this against. The numbers come from
``kitchen_dims`` and ``dishwasher_frames`` (CI-tested); this file only wires them into configs.

- ``make_dishwasher_cfg()``: an ``ArticulationCfg`` spawning ``dishwasher_2_overrides.usda``
  (``UsdFileCfg``) at ``scale = RACK_SCALE`` (1.0, uniform) and ``DISHWASHER_POSE_KITCHEN``. The
  overrides layer puts the articulation root link ``E_body_5`` at the layer origin, so
  ``init_state.pos`` is both where the spawner puts the prim and where a reset writes the root link.
  ``init_state``: door at 90 deg (pi / 2 rad, its upper limit), lower rack at -0.2 m (the limit this
  issue keeps; #11 raises both to ``DISHWASHER_RACK_TRAVEL_M``), upper rack at 0. Three
  ``ImplicitActuatorCfg`` groups keyed by the link names #29 and #59 use, ``E_door_4``,
  ``E_shelf_03`` and ``E_shelf_1_04``. Their regexes are the joint prim names
  (``RevoluteJoint_dishwasher_2_middle``...), because Isaac Lab matches joint names, which are
  the joint prims' names, not the links' (#29 named the joints after their links). The gains are
  #29's: door k = 20 and d = 10 per degree in USD, so 1145.9 and 573.0 per radian here; racks
  k = 10 N/m, d = 10 N s/m. ``activate_contact_sensors=True`` so #11 can read rack contact.
- ``make_rack_tines_cfg()``: the tines. **Mechanism:** a nested ``AssetBaseCfg`` whose prim path is
  under the lower-rack link, ``{ENV_REGEX_NS}/Dishwasher/ArtVIP/E_shelf_1_04/RackTines``, spawning
  ``rack_tines.usda`` with ``UsdFileCfg`` at ``RACK_TINES_POSE_IN_LINK`` under the compensating
  scale (``dishwasher_frames.rack_tines_scale_in_link``: the link scales its x by 0.9191). Isaac
  Lab 3.0.0 supports this: the scene spawns entities parents first, sorted by prim-path depth
  (``interactive_scene.py:765-773``), ``Asset`` spawns at ``init_state`` relative to the parent
  (``asset.py:45-51``), ``spawn_from_usd_file`` adds a reference at the path (``from_files.py:650``,
  ``create_prim`` ``prims.py:46``), and the USD cloner copies parents before declared descendants
  and skips descendants a parent copy already covers (``cloner/usd.py:49``, ``:109-123``). The tine
  colliders carry no ``RigidBodyAPI``, so by the UsdPhysics rules they become shapes of the rack
  link's rigid body and move with it. Fallback if #11 finds otherwise:
  ``scripts/make_dishwasher_overrides.py --with-tines`` writes the same reference into the layer.
- ``slot_poses_w()``: the planner's slot poses in the world frame, from ``kitchen_dims`` through
  ``dishwasher_frames.slot_poses_kitchen`` plus each environment's origin; ``slot_poses_from_sim()``
  composes them from the simulated rack-link pose instead, for #11's 1 cm comparison.

Quaternions here are Isaac Lab's (x, y, z, w); ``planning/geometry.py`` is (w, x, y, z), so slot
poses are returned as (x, y, z, yaw) and #21 builds its own quaternion.
"""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils.math import quat_apply

from kitchenbot.scene import dishwasher_frames as df
from kitchenbot.scene import kitchen_dims as kd

ASSETS = Path(__file__).resolve().parent / "assets"
DISHWASHER_USD = ASSETS / "dishwasher" / "dishwasher_2_overrides.usda"
RACK_TINES_USD = ASSETS / "dishwasher" / "rack_tines.usda"

DISHWASHER_PRIM = "{ENV_REGEX_NS}/Dishwasher"
ASSET_CHILD = "ArtVIP"  # the referencing prim inside dishwasher_2_overrides.usda
TINES_CHILD = "RackTines"

#: Joint positions at reset: door open (on its upper limit), lower rack at the -0.2 m limit kept by
#: #59 (#11 raises the limit and this to -kd.DISHWASHER_RACK_TRAVEL_M), upper rack in.
DOOR_INIT_RAD = math.radians(kd.DISHWASHER_DOOR_OPEN_DEG)
LOWER_RACK_INIT_M = kd.DISHWASHER_RACK_LIMITS_M[0]
UPPER_RACK_INIT_M = 0.0

#: Actuator groups, keyed by the link each joint moves (the names of the issue).
ACTUATOR_JOINTS = {
    df.DOOR_LINK: df.DOOR_JOINT,
    df.UPPER_RACK_LINK: df.UPPER_RACK_JOINT,
    df.LOWER_RACK_LINK: df.LOWER_RACK_JOINT,
}


def lower_rack_link_path(prim_path: str = DISHWASHER_PRIM) -> str:
    """Prim path (expression) of the lower-rack link under a dishwasher spawned at ``prim_path``."""
    return f"{prim_path}/{ASSET_CHILD}/{df.LOWER_RACK_LINK}"


def make_dishwasher_cfg(
    prim_path: str = DISHWASHER_PRIM,
    scale: float = kd.RACK_SCALE,
    pose: tuple[float, float, float, float] = kd.DISHWASHER_POSE_KITCHEN,
    lower_rack_init: float = LOWER_RACK_INIT_M,
) -> ArticulationCfg:
    """The dishwasher articulation, fixed to the world by ArtVIP's own fixed joint."""
    x, y, z, yaw = pose
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(DISHWASHER_USD),
            scale=(scale, scale, scale),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(x, y, z),
            rot=df.yaw_quat_xyzw(yaw),
            joint_pos={
                df.DOOR_JOINT: DOOR_INIT_RAD,
                df.UPPER_RACK_JOINT: UPPER_RACK_INIT_M,
                df.LOWER_RACK_JOINT: lower_rack_init,
            },
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            df.DOOR_LINK: ImplicitActuatorCfg(
                joint_names_expr=[df.DOOR_JOINT],
                stiffness=df.DOOR_STIFFNESS_PER_RAD,
                damping=df.DOOR_DAMPING_PER_RAD,
            ),
            df.UPPER_RACK_LINK: ImplicitActuatorCfg(
                joint_names_expr=[df.UPPER_RACK_JOINT],
                stiffness=df.RACK_STIFFNESS,
                damping=df.RACK_DAMPING,
            ),
            df.LOWER_RACK_LINK: ImplicitActuatorCfg(
                joint_names_expr=[df.LOWER_RACK_JOINT],
                stiffness=df.RACK_STIFFNESS,
                damping=df.RACK_DAMPING,
            ),
        },
    )


def make_rack_tines_cfg(
    dishwasher_prim_path: str = DISHWASHER_PRIM, scale: float = kd.RACK_SCALE
) -> AssetBaseCfg:
    """The authored tines under the lower-rack link (see the module docstring for the mechanism).

    ``rack_tines.usda`` is generated for the rack outline at scale 1.0; at another ``scale``
    regenerate it with ``--rack-width`` and ``--rack-depth`` multiplied by the scale (pitch and
    tine height unchanged) so the base frame still covers the rack.
    """
    x, y, z, yaw = kd.RACK_TINES_POSE_IN_LINK
    return AssetBaseCfg(
        prim_path=f"{lower_rack_link_path(dishwasher_prim_path)}/{TINES_CHILD}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(RACK_TINES_USD), scale=df.rack_tines_scale_in_link(scale)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, z), rot=df.yaw_quat_xyzw(yaw)),
    )


#: Default instances for the scene config.
DISHWASHER_CFG: ArticulationCfg = make_dishwasher_cfg()
RACK_TINES_CFG: AssetBaseCfg = make_rack_tines_cfg()


def slot_poses_w(
    env_origins: torch.Tensor,
    rack_pos: float = LOWER_RACK_INIT_M,
    scale: float = kd.RACK_SCALE,
    pose: tuple[float, float, float, float] = kd.DISHWASHER_POSE_KITCHEN,
) -> torch.Tensor:
    """Slots 1 to 5 in the world frame, ``(num_envs, 5, 4)`` as (x, y, z, yaw).

    ``env_origins`` is ``scene.env_origins`` (``interactive_scene.py:336``), ``(num_envs, 3)``; the
    kitchen frame of every environment is its origin, unrotated. ``rack_pos`` is the lower-rack
    joint position the poses are for (metres, negative is out).
    """
    local = torch.tensor(
        df.slot_poses_kitchen(scale, rack_pos, pose),
        dtype=env_origins.dtype,
        device=env_origins.device,
    )
    out = local.unsqueeze(0).repeat(env_origins.shape[0], 1, 1)
    out[..., :3] += env_origins.unsqueeze(1)
    return out


def slot_poses_from_sim(dishwasher, scale: float = kd.RACK_SCALE) -> torch.Tensor:
    """Slots 1 to 5 composed from the simulated lower-rack link pose, ``(num_envs, 5, 3)``.

    ``dishwasher`` is the scene's ``Articulation``. PhysX reports link poses without scale, so the
    tine offset is applied in metres: the link-local ``RACK_TINES_POSE_IN_LINK`` times the link's
    x scale and the asset scale. #11 compares this with :func:`slot_poses_w` at the measured joint
    position (acceptance: 1 cm).
    """
    body_ids, _ = dishwasher.find_bodies(df.LOWER_RACK_LINK)
    pose = dishwasher.data.body_link_pose_w.torch[:, body_ids[0]]  # (num_envs, 7), quat x, y, z, w
    tx, ty, tz, _ = kd.RACK_TINES_POSE_IN_LINK
    origin = (scale * tx * kd.DISHWASHER_LOWER_RACK_LINK_SCALE_X, scale * ty, scale * tz)
    offsets = [
        (origin[0] + sx, origin[1] + sy, origin[2] + sz)
        for sx, sy, sz, _yaw in df.slot_layout(scale).slot_poses()
    ]
    local = torch.tensor(offsets, dtype=pose.dtype, device=pose.device)  # (5, 3) in the link
    quat = pose[:, None, 3:7].expand(-1, local.shape[0], -1)
    return pose[:, None, 0:3] + quat_apply(quat, local.unsqueeze(0).expand(pose.shape[0], -1, -1))
