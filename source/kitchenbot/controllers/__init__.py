"""Arm and hand controllers.

Files, existing and planned (docs/architecture.md sections 7 and 8):

- ``arm_ik.py``: DifferentialIK wrapper for the right arm, fed by ``ArmTargetPipeline``. Created
  by P14 and P16.
- ``target_filter.py``: ``TargetSlewFilter`` (0.3 m/s, 1 rad/s on a ``Pose``) and
  ``ArmTargetPipeline`` (world target to pelvis frame, reachability gate, slew; returns the
  pelvis-frame target and ``infeasible``, the planner's ``arm_infeasible``). Exists since #48.
- ``reachability.py`` and ``data/right_arm_envelope.npz``: ``ReachabilityEnvelope``, the right
  palm's reachable 2 cm voxels in the pelvis frame (waist at zero), sampled with URDF forward
  kinematics by ``scripts/make_reach_envelope.py``. Exist since #48; P16 checks the envelope
  against the real IK.
- ``hand/base.py``: the ``HandController`` interface (open, pre_grasp, close, is_holding, plus
  ``update(dt)`` and the commanded ``posture``) and ``HandPosture``. Exists since #42.
- ``hand/dex3_postures.py`` and ``hand/dex3_synergy.py``: the Dex3-1 posture table built on the
  URDF joint limits and the ``Dex3SynergyHand`` logic with injected sensing. Exist since #42,
  without a simulator; P18 (#18) wires them to Isaac Lab and tunes the postures.
- ``hand/mock.py``: ``MockHand`` for tests and the task planner. Exists since #42.
- ``hand/kinematic_follow.py``: labelled baseline that attaches the plate kinematically. P17.
- ``hand/dex1.py``: conditional fallback for Unitree's two-jaw gripper. P19, only if needed.
"""
