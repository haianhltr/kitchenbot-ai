"""Lower-body balance policy: training environment and deployment action terms.

Planned files (docs/architecture.md sections 5 and 6):

- ``stand_env_cfg.py``: the ``KitchenBot-Stand-G1-Dex3`` manager-based environment. Created by P05.
- ``commands.py``: ``UpperBodyCommand``, the command term that samples random arm and hand targets
  during training. Created by P05.
- ``upper_body_action.py``: ``UpperBodyFollowAction``, an action term with agent action dimension
  zero that writes joint targets from a source object. Created by P05.
- ``rewards.py``: reward terms specific to standing with a moving upper body. Created by P05.
- ``lower_body_action.py``: ``LowerBodyPolicyAction``, wrapping the frozen TorchScript actor and its
  manifest. Created by P07.
- ``agile_baseline_action.py``: wrapper for NVIDIA's Agile checkpoint, baseline only. Created
  by P07.
- ``agents/``: RSL-RL agent configs. Created by P05.
"""
