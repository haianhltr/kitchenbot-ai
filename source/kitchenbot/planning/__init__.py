"""Task planning (docs/architecture.md section 9).

- ``plate_to_rack.py``: the finite-state machine idle, pre_grasp, grasp, lift, transport, align,
  insert, release, done (plus failed), with provisional per-state timeouts, a 30 s budget and
  failure tags from :mod:`kitchenbot.evaluation.failure_tags`. ``PlateToRackPlanner.step(obs, dt)``
  reads a ``PlannerObservation`` and returns a ``PlannerCommand`` (palm pose target, a
  ``HandAction`` naming a ``HandController`` method, and the D012 waist pitch target: the lean in
  ``pre_grasp``, ``grasp`` and ``lift``, zero otherwise, #67); it never calls the arm, the hand or
  the balance policy. The waist target steps at state changes. The deployment layer (#21) slews
  it at no more than 1 rad/s before the policy sees it, the ramp the policy is trained on (#68).
  Built and tested against a fake arm and hand in #43; the slot constants, rack force sensor and
  S3 runs in simulation are #21's.
- ``fakes.py``: the fake arm, hand, plate and rack the planner is tested against, moved there
  from ``tests/planner_helpers.py`` in #60 (which re-exports them). Test doubles: #21 and #22
  replace them with the simulated arm, hand, plate and rack. The fake arm can sit behind the #48
  reachability gate and slew filter, and ``Fakes.from_episode`` builds the plate, rack and counter
  edge from an ``EpisodeSpec`` and a ``RackSlotLayout`` for the CI dry run
  (``kitchenbot.evaluation.dry_run``); the frames it uses are in its docstring.
- ``geometry.py``: the plate grasp convention, decided in #67. The grasp frame the planner observes
  is the top rim point nearest the robot, ``(+D/2, 0, PLATE_HEIGHT_M)`` in the plate body frame
  turned so that its ``+x`` faces the robot, with ``x`` pointing back at the plate centre. The
  module also derives the slot entry from it (one diameter above the tine tops) and states what
  follows for the quarter turn, the kinematic-follow attach and the judges. It also re-exports
  :mod:`kitchenbot.geometry` (``Pose`` and the wxyz quaternion helpers, including the
  plate-vertical orientation derived from the slot yaw), which moved to the package root in #48 so
  the arm controllers share it.

Single-environment pure Python with numpy; no simulator import. Whether the composite environment
vectorises it across ``num_envs`` is #22's decision.
"""
