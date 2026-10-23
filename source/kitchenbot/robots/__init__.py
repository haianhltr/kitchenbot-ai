"""Robot configurations and joint maps.

Planned files (docs/architecture.md section 4):

- ``joint_maps.py``: joint and link names of the G1 + Dex3 asset as dumped from the USD, grouped
  into legs, waist, arms and hands, with the regexes every action term uses. A provisional version
  derived from the Unitree URDF exists since #28; P03 verifies it against the USD.
- ``joint_limits.py``: ``JointLimit`` and the Dex3-1 hand joint limits, copied from the same URDF.
  Exists since #42; the arm and waist limits were added in #48.
- ``kinematics.py``: numpy forward kinematics of a serial chain built from a URDF joint dump
  (``UrdfChain``), and ``right_arm_chain`` from the pelvis to the right palm link with the waist
  held at zero by default. Exists since #48; P03 checks the chain against the USD.
- ``g1_dex3.py``: a copy of Isaac Lab's ``G1_29DOF_CFG`` with the rev 1.0 hip-roll limits
  (139 N·m, 20 rad/s) and switchable ``fix_root_link``. An untested draft exists since #46 (it
  imports Isaac Lab, so nothing imports it on CI); P03 verifies it against the USD.
"""
