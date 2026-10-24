"""Kitchen scene: USD layers, scene configuration and randomization.

Files (docs/architecture.md section 10):

- ``kitchen_dims.py``: every authored dimension of the counter, sink, floor and rack tines in one
  standard-library module, each marked with its source or ``provisional``, plus ``RackSlotLayout``
  with ``slot_pose(i)`` in the rack frame (x across the tines, y along the tine rows, z up). The
  USD generators, their ``--check`` modes, the tests, the plate spawner and the planner's slot poses
  read it. Created by #47.
- ``usd_prims.py``: ``pxr`` helpers shared by the generators below (primitive colliders, physics
  material, checks). Created by #47.
- ``randomization.py``: docs/evaluation.md section 2 as code, numpy and the standard library only.
  ``RandomizationRanges.for_split`` holds every row for the seen and the unseen split;
  ``sample_episode(seed, split, scenario)`` returns the plate pose, diameter, rack slot, pelvis
  noise, frictions and push schedule of one episode as a plain ``EpisodeSpec``, the same every time
  for the same arguments (the sub-seed rule is in its docstring); ``sample_split`` does it for a
  scenario's evaluation seeds. ``scripts/make_eval_configs.py`` writes the ranges into
  ``configs/eval/*.yaml``. Created by #52; the Isaac Lab event terms of #13 wrap it.
- ``assets/``: authored text USD layers. Third-party assets are fetched into
  ``assets/third_party/`` (git-ignored) by ``scripts/fetch_assets.py``. Committed so far:

  - ``plates/plate_22.usda``, ``plate_25.usda``, ``plate_27.usda`` by ``scripts/make_plate_usd.py``
    (#30);
  - ``kitchen/counter.usda`` and ``kitchen/floor.usda`` by ``scripts/make_counter_usd.py`` (#47);
  - ``kitchen/dishwasher_cabinet.usda`` by ``scripts/make_cabinet_usd.py`` (#70): the dishwasher's
    plinth as a static low cabinet (carcass, toe kick, door front in the counter's colour, bar
    handle) in the kitchen frame, checked against ``dishwasher_frames.plinth_box_kitchen`` and the
    open dishwasher door and the right leg;
  - ``dishwasher/rack_tines.usda`` by ``scripts/make_rack_tines_usd.py`` (#47); #11 attaches it to
    the rack link of the ArtVIP dishwasher;
  - ``dishwasher/dishwasher_2_overrides.usda`` by ``scripts/make_dishwasher_overrides.py`` (#59):
    references the git-ignored ArtVIP ``dishwasher_2`` and overrides its colliders, limits, default
    targets and root-link offset; ``--check`` composes it with usd-core.
- ``dishwasher_frames.py``: the dishwasher's frames as numbers, standard library only (#59): slot
  poses in the kitchen frame, the slot row from the pelvis, the tine attachment scale, footprints
  and the plinth box (dressed as a cabinet by #70). Every number is a ``kitchen_dims`` constant
  (``DISHWASHER_*``).
- ``dishwasher_cfg.py`` and ``kitchen_scene_cfg.py``: UNTESTED DRAFTS written against the Isaac Lab
  v3.0.0 source (#59), parsed on CI and never imported there. The ``ArticulationCfg`` of the
  dishwasher with its tines and a slot-pose helper, and the ``InteractiveSceneCfg`` with selectable
  layers (``counter``, ``dishwasher``, ``plates``) and a fixed or floating G1. #10 and #11 run them.

Planned files:

- ``plate_cfg.py``: ``RigidObjectCfg`` factory for the rimmed plates. Created by P09.
- The Isaac Lab event terms (reset events and the push) that apply an ``EpisodeSpec`` to the scene,
  in a file beside ``randomization.py`` whose name P13 chooses. Created by P13.
"""
