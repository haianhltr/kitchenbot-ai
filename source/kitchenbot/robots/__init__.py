"""Robot configurations and joint maps.

Planned files (docs/architecture.md section 4):

- ``joint_maps.py``: joint and link names of the G1 + Dex3 asset as dumped from the USD, grouped
  into legs, waist, arms and hands, with the regexes every action term uses. Created by P03.
- ``g1_dex3.py``: a copy of Isaac Lab's ``G1_29DOF_CFG`` with the rev 1.0 hip-roll limits
  (139 N·m, 20 rad/s) and switchable ``fix_root_link``. Created by P03.
"""
