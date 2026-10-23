"""Kitchen scene: USD layers, scene configuration and randomization.

Files (docs/architecture.md section 10):

- ``kitchen_dims.py``: every authored dimension of the counter, sink, floor and rack tines in one
  standard-library module, each marked with its source or ``provisional``, plus ``RackSlotLayout``
  with ``slot_pose(i)`` in the rack frame (x across the tines, y along the tine rows, z up). The
  USD generators, their ``--check`` modes, the tests, the plate spawner and the planner's slot poses
  read it. Created by #47.
- ``usd_prims.py``: ``pxr`` helpers shared by the generators below (primitive colliders, physics
  material, checks). Created by #47.
- ``assets/``: authored text USD layers. Third-party assets are fetched into
  ``assets/third_party/`` (git-ignored) by ``scripts/fetch_assets.py``. Committed so far:

  - ``plates/plate_22.usda``, ``plate_25.usda``, ``plate_27.usda`` by ``scripts/make_plate_usd.py``
    (#30);
  - ``kitchen/counter.usda`` and ``kitchen/floor.usda`` by ``scripts/make_counter_usd.py`` (#47);
  - ``dishwasher/rack_tines.usda`` by ``scripts/make_rack_tines_usd.py`` (#47); #11 attaches it to
    the rack link of the ArtVIP dishwasher.

Planned files:

- ``kitchen_scene_cfg.py``: the ``InteractiveSceneCfg`` with selectable layers. Created by P10.
- ``plate_cfg.py``: ``RigidObjectCfg`` factory for the rimmed plates. Created by P09.
- ``dishwasher_cfg.py``: ``ArticulationCfg`` and slot pose constants. Created by P11.
- ``randomization.py``: event terms implementing docs/evaluation.md section 2. Created by P13.
"""
