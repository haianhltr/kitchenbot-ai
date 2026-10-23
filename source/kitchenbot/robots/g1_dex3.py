# UNTESTED DRAFT (#46): written against Isaac Lab v3.0.0 source without a simulator; #5 runs and fixes it.  # noqa: E501
"""G1 (29-DoF body) with Dex3-1 hands: Isaac Lab's ``G1_29DOF_CFG`` with our fixes.

Changes against ``isaaclab_assets.robots.unitree.G1_29DOF_CFG`` (v3.0.0 ``unitree.py:413``; every
reference below is in docs/isaaclab_3_api_notes.md section 2):

- Hip roll actuator limits set to the rev 1.0 URDF values, 139 N·m and 20 rad/s (upstream has
  88 N·m and 32 rad/s, ``unitree.py:457`` and ``:463``; IsaacLab issue #8530). The knee is already
  139 N·m and 20 rad/s upstream (``:459``, ``:465``); the draft states it again so the intent is
  explicit, without changing a value.
- ``activate_contact_sensors=True``: upstream has ``False`` (``:416``) and the stand env's contact
  sensor needs contact reporting.
- ``joint_ordering`` is ``joint_maps.ALL_JOINTS``, so every joint-indexed tensor in Isaac Lab uses
  our order. A name that does not exist in the USD makes articulation init fail, which is the
  check #3 wants.
- ``make_g1_dex3_cfg(fix_root_link=...)`` gives the floating-base (default) or fixed-pelvis
  variant.

#3 verifies the joint names, the limits and the default pose against the Isaac Sim USD.
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import clone
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from kitchenbot.robots.joint_maps import ALL_JOINTS, LEGS

#: Rev 1.0 URDF actuator limits (tests/fixtures/g1_dex3_joints_urdf.json): N·m and rad/s.
REV_1_0_LIMITS: dict[str, tuple[float, float]] = {
    "hip_roll": (139.0, 20.0),
    "knee": (139.0, 20.0),
}

#: Upstream regex keys in the ``legs`` actuator group that the fix replaces with explicit names.
_UPSTREAM_KEYS = {"hip_roll": ".*_hip_roll_joint", "knee": ".*_knee_joint"}


def _joints(kind: str) -> list[str]:
    """Leg joint names containing ``_<kind>_joint``, from ``joint_maps.LEGS``."""
    return [name for name in LEGS if name.endswith(f"_{kind}_joint")]


def make_g1_dex3_cfg(fix_root_link: bool = False) -> ArticulationCfg:
    """A fresh G1 + Dex3 articulation config; ``fix_root_link=True`` pins the pelvis."""
    cfg = clone(G1_29DOF_CFG)
    cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    cfg.spawn.fix_root_link = fix_root_link
    cfg.spawn.activate_contact_sensors = True
    cfg.joint_ordering = ALL_JOINTS

    legs = cfg.actuators["legs"]
    effort = dict(legs.actuator_effort_limit)
    velocity = dict(legs.actuator_velocity_limit)
    for kind, (effort_limit, velocity_limit) in REV_1_0_LIMITS.items():
        effort.pop(_UPSTREAM_KEYS[kind], None)
        velocity.pop(_UPSTREAM_KEYS[kind], None)
        for name in _joints(kind):
            effort[name] = effort_limit
            velocity[name] = velocity_limit
    legs.actuator_effort_limit = effort
    legs.actuator_velocity_limit = velocity
    return cfg


#: Floating-base G1 + Dex3, the configuration the stand environment trains on.
G1_DEX3_CFG: ArticulationCfg = make_g1_dex3_cfg(fix_root_link=False)

#: Fixed-pelvis variant for arm and hand development.
G1_DEX3_FIXED_CFG: ArticulationCfg = make_g1_dex3_cfg(fix_root_link=True)
