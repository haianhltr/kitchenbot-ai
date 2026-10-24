# UNTESTED DRAFT (#59): written against Isaac Lab v3.0.0 source without a simulator; #10 and #11 run and fix it.  # noqa: E501
"""The kitchen ``InteractiveSceneCfg``: floor, counter, dishwasher with tines, plates, G1, light.

Every Isaac Lab symbol is pinned in docs/isaaclab_3_api_notes.md section 9. Each environment's
origin is the kitchen frame of ``kitchen_dims`` (the counter's front-edge middle, on the floor), so
``counter.usda`` and ``floor.usda`` spawn at the origin unrotated and every pose below is a
``kitchen_dims`` or ``dishwasher_frames`` number.

- **Layers.** ``LAYERS`` = ``counter``, ``dishwasher``, ``plates``; the floor, the robot and the
  light always load. :func:`make_kitchen_scene_cfg` sets the fields of unselected layers to None,
  which ``InteractiveScene`` skips (``interactive_scene.py:767``, ``cfg is not None``).
- **Floor.** ``floor.usda`` per environment (friction 1.0, top at z = 0); ``ENV_SPACING_M`` equals
  the 6 m floor so neighbouring floors tile. ``GroundPlaneCfg`` (``from_files_cfg.py:373``) is the
  alternative #10 may pick.
- **Dishwasher.** ``dishwasher_cfg.DISHWASHER_CFG`` at ``DISHWASHER_POSE_KITCHEN``, its tines as the
  nested ``RACK_TINES_CFG``, and the static cabinet under the body, part of the same layer: the
  plinth dressed as a low cabinet (carcass, door front, handle; D013, #70).
  ``dishwasher_cabinet.usda`` is authored in the kitchen frame, like ``counter.usda``, so it spawns
  at the origin unrotated; its carcass is ``dishwasher_frames.plinth_box_kitchen`` shortened by
  ``kitchen_dims.PLINTH_GAP_M``.
- **Plates: one diameter per episode.** Three ``RigidObjectCfg`` (22, 25 and 27 cm, #30), one at the
  nominal spawn point on the counter (``randomization.PLATE_NOMINAL_XY_M`` on the counter top) and
  two parked flat on the floor behind the counter. #13's reset event moves the drawn diameter to
  its sampled pose and the other two to their parking spots. Chosen over one asset switched per
  episode because the 3.0 cloner fixes each environment's spawn variant at clone time
  (``MultiUsdFileCfg``, ``wrappers_cfg.py:56``, picks once per environment), so a per-episode draw
  needs all three bodies present; it costs two idle rigid bodies per environment.
- **Robot.** ``robots/g1_dex3.make_g1_dex3_cfg(fix_root_link=...)``: floating base by default,
  ``fixed_base=True`` pins the pelvis for arm and hand work. It stands at
  ``(0, -ROBOT_STANDOFF_M)`` with Isaac Lab's own height and heading (``unitree.py:436-445``:
  z = 0.75 m and a +90 deg yaw, so it faces +y, the counter).
- **Light.** One dome light, global.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace

from kitchenbot.robots.g1_dex3 import make_g1_dex3_cfg
from kitchenbot.scene import dishwasher_cfg as dwc
from kitchenbot.scene import kitchen_dims as kd
from kitchenbot.scene.randomization import PLATE_NOMINAL_XY_M

ASSETS = Path(__file__).resolve().parent / "assets"

#: Selectable layers, in load order; the floor, the robot and the light are always present.
LAYERS = ("counter", "dishwasher", "plates")
#: The scene fields each layer owns.
LAYER_FIELDS = {
    "counter": ("counter",),
    "dishwasher": ("dishwasher_cabinet", "dishwasher", "rack_tines"),
    "plates": ("plate_22", "plate_25", "plate_27"),
}
ENV_SPACING_M = kd.FLOOR_SIZE_M[0]

PLATE_FILES = {0.22: "plate_22.usda", 0.25: "plate_25.usda", 0.27: "plate_27.usda"}
ACTIVE_PLATE_M = 0.25  # the plate on the counter at spawn; #13 draws one per episode
#: Parking spots of the inactive plates: flat on the floor behind the counter, out of the robot's
#: way and inside the 6 m floor (provisional, #13).
PLATE_PARK_XY_M = {0.22: (-1.5, 1.5), 0.25: (-1.0, 1.5), 0.27: (-0.5, 1.5)}


def plate_cfg(diameter_m: float, active: bool) -> RigidObjectCfg:
    """One plate (``RigidBodyAPI``, mass and colliders are in the USD, #30)."""
    if active:
        pos = (PLATE_NOMINAL_XY_M[0], PLATE_NOMINAL_XY_M[1], kd.COUNTER_HEIGHT_M)
    else:
        pos = (*PLATE_PARK_XY_M[diameter_m], 0.0)
    name = PLATE_FILES[diameter_m].removesuffix(".usda").replace("plate_", "Plate_")
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS / "plates" / PLATE_FILES[diameter_m]),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def robot_cfg(fixed_base: bool = False) -> ArticulationCfg:
    """G1 + Dex3 at the stand-off, facing the counter."""
    cfg = make_g1_dex3_cfg(fix_root_link=fixed_base)
    pos = (0.0, -kd.ROBOT_STANDOFF_M, cfg.init_state.pos[2])
    return replace(cfg, init_state=replace(cfg.init_state, pos=pos))


def cabinet_cfg() -> AssetBaseCfg:
    """Static low cabinet from the floor to just under the dishwasher body (#70, D013).

    The asset is generated at ``kitchen_dims.RACK_SCALE`` and ``DISHWASHER_POSE_KITCHEN``
    (``scripts/make_cabinet_usd.py``); regenerate it if either changes.
    """
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DishwasherCabinet",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / "kitchen" / "dishwasher_cabinet.usda")),
    )


@configclass
class KitchenSceneCfg(InteractiveSceneCfg):
    """Every layer on; use :func:`make_kitchen_scene_cfg` to select layers and the base mode."""

    floor: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Floor",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / "kitchen" / "floor.usda")),
    )
    counter: AssetBaseCfg | None = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Counter",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / "kitchen" / "counter.usda")),
    )
    dishwasher_cabinet: AssetBaseCfg | None = cabinet_cfg()
    dishwasher: ArticulationCfg | None = dwc.DISHWASHER_CFG
    rack_tines: AssetBaseCfg | None = dwc.RACK_TINES_CFG
    plate_22: RigidObjectCfg | None = plate_cfg(0.22, active=ACTIVE_PLATE_M == 0.22)
    plate_25: RigidObjectCfg | None = plate_cfg(0.25, active=ACTIVE_PLATE_M == 0.25)
    plate_27: RigidObjectCfg | None = plate_cfg(0.27, active=ACTIVE_PLATE_M == 0.27)
    robot: ArticulationCfg = robot_cfg(fixed_base=False)
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )


def parse_layers(text: str) -> tuple[str, ...]:
    """``"counter,dishwasher"`` -> ``("counter", "dishwasher")``; unknown names raise."""
    names = tuple(name.strip() for name in text.split(",") if name.strip())
    unknown = sorted(set(names) - set(LAYERS))
    if unknown:
        raise ValueError(f"unknown layers {unknown}; choose from {', '.join(LAYERS)}")
    return names


def make_kitchen_scene_cfg(
    num_envs: int = 1,
    env_spacing: float = ENV_SPACING_M,
    layers: Iterable[str] = LAYERS,
    fixed_base: bool = False,
) -> KitchenSceneCfg:
    """The kitchen scene with only ``layers`` loaded and the robot's base fixed or floating."""
    layers = tuple(layers)
    parse_layers(",".join(layers))
    cfg = KitchenSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    cfg.robot = robot_cfg(fixed_base=fixed_base)
    for layer, fields in LAYER_FIELDS.items():
        if layer not in layers:
            for name in fields:
                setattr(cfg, name, None)
    return cfg
