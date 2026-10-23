"""Task planning (docs/architecture.md section 9).

- ``plate_to_rack.py``: the finite-state machine idle, pre_grasp, grasp, lift, transport, align,
  insert, release, done (plus failed), with provisional per-state timeouts, a 30 s budget and
  failure tags from :mod:`kitchenbot.evaluation.failure_tags`. ``PlateToRackPlanner.step(obs, dt)``
  reads a ``PlannerObservation`` and returns a ``PlannerCommand`` (palm pose target and a
  ``HandAction`` naming a ``HandController`` method); it never calls the arm or the hand. Built and
  tested against a fake arm and hand in #43; the slot constants, rack force sensor and S3 runs
  in simulation are #21's.
- ``geometry.py``: re-export of :mod:`kitchenbot.geometry` (``Pose`` and the wxyz quaternion
  helpers, including the plate-vertical orientation derived from the slot yaw), which moved to the
  package root in #48 so the arm controllers share it.

Single-environment pure Python with numpy; no simulator import. Whether the composite environment
vectorises it across ``num_envs`` is #22's decision.
"""
