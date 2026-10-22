"""Kitchen scene: USD layers, scene configuration and randomization.

Planned files (docs/architecture.md section 10):

- ``assets/``: authored text USD layers (counter, floor, plates, dishwasher, rack tines).
  Third-party assets are fetched into ``assets/third_party/`` (git-ignored) by
  ``scripts/fetch_assets.py``.
- ``kitchen_scene_cfg.py``: the ``InteractiveSceneCfg`` with selectable layers. Created by P10.
- ``plate_cfg.py``: ``RigidObjectCfg`` factory for the rimmed plates. Created by P09.
- ``dishwasher_cfg.py``: ``ArticulationCfg`` and slot pose constants. Created by P11.
- ``randomization.py``: event terms implementing docs/evaluation.md section 2. Created by P13.
"""
