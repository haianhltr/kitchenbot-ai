"""Arm and hand controllers.

Planned files (docs/architecture.md sections 7 and 8):

- ``arm_ik.py``: DifferentialIK wrapper for the right arm with target slewing and a reachability
  check. Created by P14 and P16.
- ``hand/base.py``: the ``HandController`` interface: open, pre_grasp, close, is_holding. Created
  by P17.
- ``hand/kinematic_follow.py``: labelled baseline that attaches the plate kinematically. P17.
- ``hand/dex3_synergy.py``: three-posture Dex3-1 controller with fingertip contact
  confirmation. P18.
- ``hand/dex1.py``: conditional fallback for Unitree's two-jaw gripper. P19, only if needed.
"""
