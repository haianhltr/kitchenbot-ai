"""Composite task environments and gym registration.

Planned files (docs/architecture.md sections 5 and 11):

- ``stand_only_env_cfg.py``: balance policy plus a constant upper-body posture, for S0. P07.
- ``reach_env_cfg.py``: fixed-pelvis reaching with the arm IK, for S1 fixed-base. P14.
- ``kitchen_env_cfg.py``: floating base, balance policy, arm IK and hand term in the kitchen, with
  the planner-facing API. P15.

Gym IDs are registered lazily so that importing this package never imports Isaac Sim.
"""
