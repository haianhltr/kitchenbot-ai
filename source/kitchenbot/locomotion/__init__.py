"""Lower-body balance policy: training environment and deployment action terms.

Importing this package imports nothing else: the environment files below import Isaac Lab and
torch at module level, so they are loaded only by the Isaac Lab CLI or by a script that runs with
the simulator (docs/architecture.md sections 5 and 6).

Simulator-free and tested on CI (#46):

- ``upper_body_targets.py``: the math of the upper-body command (numpy): target sampling inside the
  joint limits scaled by a range factor, 1 to 3 s resample times, linear interpolation, the
  20 percent slow right-arm reach, and the range-factor curriculum rule.
- ``waist_lean_targets.py``: the math of the waist-lean command (D012, #68): targets in
  ``[0, 0.52]`` rad, exactly zero half of the time, a new target every 1 to 3 s, a 1 rad/s ramp,
  and the waist tracking error the reward uses.
- ``obs_layout.py``: the policy observation layout as data (139 values per frame, 5 frames,
  time-major); #6 writes it into the export manifest.

Untested drafts written against the Isaac Lab v3.0.0 source without a simulator (#46); P05 runs
and fixes them. Each starts with an ``UNTESTED DRAFT`` header line, CI only parses them, and every
Isaac Lab symbol they use is in docs/isaaclab_3_api_notes.md:

- ``stand_env_cfg.py``: ``StandEnvCfg`` for ``KitchenBot-Stand-G1-Dex3`` and ``StandEnvCfg_PLAY``.
- ``commands.py``: ``UpperBodyCommand``, the command term that samples arm and hand targets, and
  ``WaistLeanCommand``, the forward waist pitch command (#68).
- ``upper_body_action.py``: ``UpperBodyFollowAction``, an action term with agent action dimension
  zero that writes joint targets from a source object.
- ``rewards.py``: reward terms specific to standing with a moving upper body, among them the
  waist tracking term (#68).
- ``curriculums.py``: the curriculum term that widens the upper-body range as falls become rare.
- ``agents/rsl_rl_ppo_cfg.py``: the RSL-RL PPO config.

Planned:

- ``lower_body_action.py``: ``LowerBodyPolicyAction``, wrapping the frozen TorchScript actor and its
  manifest. Created by P07.
- ``agile_baseline_action.py``: wrapper for NVIDIA's Agile checkpoint, baseline only. Created
  by P07.
"""
