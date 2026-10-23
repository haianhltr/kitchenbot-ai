"""Arm and hand controllers.

Files, existing and planned (docs/architecture.md sections 7 and 8):

- ``arm_ik.py``: DifferentialIK wrapper for the right arm with target slewing and a reachability
  check. Created by P14 and P16.
- ``hand/base.py``: the ``HandController`` interface (open, pre_grasp, close, is_holding, plus
  ``update(dt)`` and the commanded ``posture``) and ``HandPosture``. Exists since #42.
- ``hand/dex3_postures.py`` and ``hand/dex3_synergy.py``: the Dex3-1 posture table built on the
  URDF joint limits and the ``Dex3SynergyHand`` logic with injected sensing. Exist since #42,
  without a simulator; P18 (#18) wires them to Isaac Lab and tunes the postures.
- ``hand/mock.py``: ``MockHand`` for tests and the task planner. Exists since #42.
- ``hand/kinematic_follow.py``: labelled baseline that attaches the plate kinematically. P17.
- ``hand/dex1.py``: conditional fallback for Unitree's two-jaw gripper. P19, only if needed.
"""
