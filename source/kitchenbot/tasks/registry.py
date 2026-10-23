# UNTESTED DRAFT (#46): written against Isaac Lab v3.0.0 source without a simulator; #5 runs and fixes it.  # noqa: E501
"""Gym registration of the KitchenBot tasks.

Importing this module registers every task (the Isaac Lab CLI discovers tasks by importing the
modules in the ``isaaclab.tasks`` entry-point group, ``isaaclab/cli/__init__.py:48``), and
``register_tasks()`` does the same for our own scripts; both are idempotent. The entry points are
strings, as in Isaac Lab's G1 registration (``core/velocity/config/g1/__init__.py:16``), so this
module imports only ``gymnasium``; the environment drafts load when a command resolves the config.

Nothing imports this module at package import: ``import kitchenbot.tasks`` stays free of
simulator packages. How scripts and the CLI reach it is in ``kitchenbot/tasks/__init__.py``.
"""

import gymnasium as gym

#: Task id -> registration kwargs.
TASKS: dict[str, dict[str, str]] = {
    "KitchenBot-Stand-G1-Dex3": {
        "env_cfg_entry_point": "kitchenbot.locomotion.stand_env_cfg:StandEnvCfg",
        "rsl_rl_cfg_entry_point": "kitchenbot.locomotion.agents.rsl_rl_ppo_cfg:StandPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
}


def register_tasks() -> list[str]:
    """Register every task in ``TASKS`` that gym does not know yet; return the ids registered."""
    added = []
    for task_id, kwargs in TASKS.items():
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs=dict(kwargs),
        )
        added.append(task_id)
    return added


register_tasks()
