"""Hand controllers sharing one interface (docs/architecture.md section 8).

- ``base``: ``HandController`` and ``HandPosture`` (#42).
- ``dex3_postures``: Dex3-1 open, pre-grasp and closed postures as closing fractions of the URDF
  joint limits; the thumb closing directions are unverified until #18 runs them (#42).
- ``dex3_synergy``: ``Dex3SynergyHand`` logic without a simulator (#42). #18 adds the Isaac Lab
  wrapper: fingertip contact sensors, the plate-tracks-palm check and writing the targets to the
  hand action term, and tunes the postures.
- ``mock``: ``MockHand`` for unit tests and the task planner (#42).
- ``kinematic_follow``: the M4a baseline that attaches the plate kinematically, added by #17.
- ``dex1``: the Dex1 two-jaw fallback, #19, only if needed.

Nothing in this package imports Isaac Lab, Isaac Sim or torch yet.
"""
