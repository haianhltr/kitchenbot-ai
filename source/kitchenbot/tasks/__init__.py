"""Composite task environments and gym registration.

Planned files (docs/architecture.md sections 5 and 11):

- ``stand_only_env_cfg.py``: balance policy plus a constant upper-body posture, for S0. P07.
- ``reach_env_cfg.py``: fixed-pelvis reaching with the arm IK, for S1 fixed-base. P14.
- ``kitchen_env_cfg.py``: floating base, balance policy, arm IK and hand term in the kitchen, with
  the planner-facing API. P15.

Gym IDs are registered lazily so that importing this package never imports Isaac Sim: nothing here
imports ``registry.py`` (an untested draft from #46, see docs/isaaclab_3_api_notes.md section 7).
Importing ``kitchenbot.tasks.registry`` registers every KitchenBot task with ``gymnasium`` using
string entry points, so it loads no environment code until a task is made. Two ways to reach it:

- **Our scripts** (with Isaac Lab installed)::

      from kitchenbot.tasks.registry import register_tasks
      register_tasks()  # idempotent; returns the ids it added
      env = gym.make("KitchenBot-Stand-G1-Dex3", cfg=env_cfg)

- **The ``isaaclab`` CLI** (``train``, ``play``, ``zero_agent``, ``random_agent``, ``list_envs``)
  imports the modules listed in the ``isaaclab.tasks`` entry-point group before it looks up a task.
  P05 adds that hook to ``pyproject.toml`` and checks it with ``isaaclab list_envs``::

      [project.entry-points."isaaclab.tasks"]
      kitchenbot = "kitchenbot.tasks.registry"
"""
