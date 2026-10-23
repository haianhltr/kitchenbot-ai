"""The ``dry`` scenario: the whole evaluation chain on CI, no simulation (#60).

``scripts/evaluate.py --scenario dry --config configs/eval/s4_seen.yaml`` runs, for every seed,

1. the sampler (:func:`kitchenbot.scene.randomization.sample_episode`) for the episode's plate pose,
   diameter, rack slot and pelvis noise;
2. the plate-to-rack planner (:class:`~kitchenbot.planning.plate_to_rack.PlateToRackPlanner`, #43)
   against the test doubles of :mod:`kitchenbot.planning.fakes` at 50 Hz: a fake arm behind the
   #48 reachability gate and slew filter (``reach_gate``, on by default), a fake hand, plate and
   rack built from the sampled episode and the ``kitchen_dims.RackSlotLayout``;
3. the judges (:mod:`kitchenbot.evaluation.judges`, #57), stepped every control step with the
   plate **body** pose converted from the fakes' grasp frame (the convention of #58's tests, see
   :mod:`kitchenbot.planning.fakes`), the hand's holding flag and the rack contact force;
4. :func:`~kitchenbot.evaluation.judges.episode_outcome` for ``success`` and the primary failure
   tag, and the S4 metric set for the result document.

It is the template of a real backend: #21 and #22 replace step 2's fakes with the simulation and
keep this file's shape (a loader that reads the :class:`EvalConfig`, a backend with
``resolved_config`` / ``run_batch`` / ``close``, an episode loop that feeds the same judges). It is
**synthetic**: ``run.synthetic`` is true, nothing in it is a measurement, and
``scripts/update_experiments.py`` and ``scripts/plot_results.py`` refuse it unless told
``--allow-synthetic``. What the numbers do say is whether the contracts between the sampler, the
planner, the arm pipeline and the judges hold together, and what the reach gap of docs/status.md
is worth when the committed envelope guards the arm: with the stand-off of ``kitchen_dims`` the
first target of ``pre_grasp`` is out of the palm's reach in every sampled episode, so the gate-on
run ends in ``ik_infeasible`` (``tests/test_dry_run.py`` asserts it for seeds 0 to 19 of both
splits), while the gate-off run is the check that the planner, the fakes and the judges agree.

Config
------
The evaluator hashes the ``--config`` YAML and does not read it; this loader does. The YAML names
the sampler scenario (``scenario``: S2, S3 or S4; the dry run always reports the S4 metric set and
judges a full pick-and-place, whichever it mirrors, so S2 and S3 only change the draws), the split
(it must be the run's split) and the ``randomization`` block (read with
:meth:`RandomizationRanges.from_dict`, so an edited range changes the episodes). Without
``--config`` it mirrors S4 with the split's default ranges. No YAML field and no schema field
carries the mirrored scenario: ``run.config_path`` says which file was used and the scenario goes
into ``resolved_config`` and so into ``run.config_hash``.

``EvalConfig.extras`` (the command line has no flag for it): ``{"reach_gate": False}`` turns the
reachability gate off, which leaves the fake arm unconstrained. Both settings are testable and
differ in ``run.config_hash``.

What the dry run does not model
-------------------------------
Balance (``fell`` is always false and ``torso_orientation_error_deg`` always 0.0: there is no
torso), contact physics (the rack force is the fake rack's box model, the plate is held rigidly and
rests where it is released, the counter contact the grasp judge takes is always false), friction
(drawn by the sampler, unused here) and the hand's grip.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kitchenbot.controllers.reachability import ReachabilityEnvelope
from kitchenbot.evaluation.failure_tags import FailureTag
from kitchenbot.evaluation.judges import GraspJudge, PlacementJudge, episode_outcome
from kitchenbot.evaluation.runner import EpisodeResult, EvalConfig, ScenarioConfigError
from kitchenbot.planning.fakes import DRY_RUN_RACK_FRAME, PELVIS_HEIGHT_M, Fakes
from kitchenbot.planning.plate_to_rack import PlannerConfig, PlannerState, PlateToRackPlanner
from kitchenbot.scene import kitchen_dims as kd
from kitchenbot.scene.randomization import (
    EpisodeSpec,
    RandomizationRanges,
    SpawnDims,
    sample_episode,
)

__all__ = [
    "CONTROL_HZ",
    "EXTRA_KEYS",
    "GENERATOR_VERSION",
    "MIRRORABLE_SCENARIOS",
    "SETTLE_S",
    "DryEpisode",
    "DryRunBackend",
    "dry_run_planner_config",
    "read_eval_yaml",
    "run_dry_episode",
    "simulate_episode",
]

#: Version of the dry-run chain; part of ``resolved_config`` and so of ``run.config_hash``. Bump it
#: when the fakes, the loop or the mapping to metrics change what an episode produces.
GENERATOR_VERSION = "dry-v1"

#: Control rate of the loop, the rate of the arm and hand controllers (docs/architecture.md
#: section 5).
CONTROL_HZ = 50.0

#: How long the loop keeps running after the planner is ``done`` so that the placement judge's 2 s
#: stability window can close (the planner's own hold check ends about when the window opens).
SETTLE_S = 3.0

#: The sampler scenarios the dry run can mirror: the ones with a grasp and a placement phase.
MIRRORABLE_SCENARIOS: tuple[str, ...] = ("S2", "S3", "S4")

#: ``EvalConfig.extras`` keys this scenario reads.
EXTRA_KEYS: tuple[str, ...] = ("reach_gate",)

_DEFAULT_MIRROR = "S4"


def dry_run_planner_config() -> PlannerConfig:
    """The planner configuration of the dry run: the defaults with the lift aimed 2 cm higher.

    **Contract gap found by the dry run, not fixed here** (``plate_to_rack.py`` is not this
    issue's): the planner's ``lift`` state exits when the palm is within ``position_tolerance_m``
    (2 cm) of the waypoint ``lift_height_m`` (15 cm) above the grasp, and the next waypoint, the
    counter edge, is at the same height, so a plate carried on a straight slew never rises above
    about 14.9 cm; the grasp judge needs 15 cm (``GRASP_LIFT_HEIGHT_M``, the same number read from
    the same ``PlannerConfig.lift_height_m``). With the defaults every dry-run episode therefore
    ends in ``drop`` (the plate was held, never judged grasped, then released in the slot).
    Aiming the waypoint at ``lift_height_m + position_tolerance_m`` removes it. The judges keep
    reading the default 15 cm, since they take it from a default ``PlannerConfig``. #21 decides
    the real fix (waypoint margin, or an exit on height) and then this override goes.
    """
    base = PlannerConfig()
    return PlannerConfig(lift_height_m=base.lift_height_m + base.position_tolerance_m)


def read_eval_yaml(path: Path | str, split: str) -> tuple[str, RandomizationRanges]:
    """``(sampler scenario, ranges)`` from a ``configs/eval`` YAML, validated for a ``split`` run.

    Raises :class:`ScenarioConfigError` if the file is not a mapping, names a scenario the dry run
    cannot mirror, belongs to another split or has a ``randomization`` block the sampler rejects.
    """
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ScenarioConfigError(f"{path.as_posix()}: cannot read as YAML: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ScenarioConfigError(f"{path.as_posix()}: expected a mapping at the top level")
    scenario = data.get("scenario")
    if scenario not in MIRRORABLE_SCENARIOS:
        raise ScenarioConfigError(
            f"{path.as_posix()}: scenario {scenario!r}; the dry run judges a pick and place and "
            f"mirrors one of {MIRRORABLE_SCENARIOS}"
        )
    if data.get("split") != split:
        raise ScenarioConfigError(
            f"{path.as_posix()}: split {data.get('split')!r} but the run is on split {split!r}"
        )
    block = data.get("randomization")
    try:
        ranges = (
            RandomizationRanges.for_split(split)
            if block is None
            else RandomizationRanges.from_dict(block)
        )
        if ranges.split != split:
            raise ValueError(f"randomization.split is {ranges.split!r}, not {split!r}")
    except (TypeError, ValueError) as exc:
        raise ScenarioConfigError(f"{path.as_posix()}: bad randomization block: {exc}") from exc
    return str(scenario), ranges


@dataclass(frozen=True)
class DryEpisode:
    """One simulated episode: the result record plus what a test wants to look at."""

    result: EpisodeResult
    #: The planner's final state (``done``, ``failed``, or whatever it was when the loop ended).
    state: PlannerState
    #: The planner's failure tag (``None`` unless it failed) and the state it failed in.
    planner_failure: FailureTag | None
    failed_in: PlannerState | None
    #: Steps on which the arm reported ``infeasible`` (the gate refused).
    gate_refusals: int
    #: Control steps simulated.
    steps: int
    grasp_why: str
    placement_why: str


def simulate_episode(
    spec: EpisodeSpec,
    layout: kd.RackSlotLayout,
    *,
    gate: bool,
    envelope: ReachabilityEnvelope | None = None,
    stand_off_m: float = kd.ROBOT_STANDOFF_M,
    planner_config: PlannerConfig | None = None,
    hz: float = CONTROL_HZ,
    settle_s: float = SETTLE_S,
) -> DryEpisode:
    """Run the planner against the fakes of ``spec`` with the judges watching; one S4 episode.

    The loop: at control step ``k`` (time ``k / hz``) the judges see the fakes' state, then the
    planner reads the same observation and commands, then the fakes apply the command. It ends at
    the planner's ``failed``, at the placement judge's success (its 2 s window closed), or
    ``settle_s`` after the planner is ``done``. ``stand_off_m`` moves the pelvis (the reach-gap
    knob); ``envelope`` is only used with ``gate``. ``planner_config`` defaults to
    :func:`dry_run_planner_config`, not to ``PlannerConfig()`` (see there why).
    """
    cfg = planner_config if planner_config is not None else dry_run_planner_config()
    fakes = Fakes.from_episode(spec, layout, gate=gate, envelope=envelope, stand_off_m=stand_off_m)
    planner = PlateToRackPlanner(cfg)
    grasp = GraspJudge()
    placement = PlacementJudge(layout, spec.rack_slot, spec.plate_diameter_m)
    rack_frame = fakes.rack.frame
    dt = 1.0 / hz
    settle_steps = int(round(settle_s * hz))
    max_steps = int(round((cfg.total_budget_s + settle_s + 1.0) * hz))

    planner.reset(fakes.observe())
    grasp.reset(fakes.plate.body_pose())
    done_at: int | None = None
    ended_by_planner_failure = False
    k = 0
    while True:
        t = k * dt
        obs = fakes.observe()
        body = fakes.plate.body_pose()
        grasp.step(t, body, obs.hand_holding)
        placement.step(t, body, rack_frame, obs.rack_contact_force, released=not obs.hand_holding)
        if placement.placement_success:  # the verdict is final
            break
        if done_at is not None and k - done_at >= settle_steps:
            break
        if k >= max_steps:
            break
        command = planner.step(obs, dt)
        if command.state is PlannerState.FAILED:
            ended_by_planner_failure = True
            k += 1
            break
        if command.state is PlannerState.DONE and done_at is None:
            done_at = k
        fakes.apply(command, dt)
        k += 1
    end_s = k * dt

    failure = planner.failure
    completion = placement.completion_time_s
    # Within the 30 s timeout is part of task success (docs/evaluation.md section 3).
    late = completion is not None and completion > cfg.total_budget_s + 1e-9
    stuck = not ended_by_planner_failure and done_at is None and k >= max_steps
    timed_out = failure is FailureTag.TIMEOUT or late or stuck
    try:
        success, tag = episode_outcome(
            "S4",
            fell=False,  # no balance model
            ik_infeasible=fakes.arm.refusals > 0,
            grasp=grasp.grasp_success,
            placement=placement.placement_success,
            dropped=grasp.dropped or failure is FailureTag.DROP,
            rack_collision_before_release=placement.rack_collision,
            timed_out=timed_out,
            held=grasp.held,
            released=placement.released,
        )
    except ValueError as exc:  # a failed trial no judge or planner signal explains
        raise RuntimeError(
            f"dry run seed {spec.seed}: the planner ({planner.state.value}) and the judges "
            f"disagree: {exc}. grasp: {grasp.why()}. placement: {placement.why()}"
        ) from exc

    completion_s = round(float(completion), 4) if success and completion is not None else None
    metrics = {
        "fell": False,
        "torso_orientation_error_deg": 0.0,
        "grasp_success": bool(grasp.grasp_success),
        "placement_success": bool(placement.placement_success),
        "rack_collision": bool(placement.rack_collision),
        "completion_time_s": completion_s,
    }
    result = EpisodeResult(
        seed=spec.seed,
        success=success,
        failure_tag=tag,
        duration_s=completion_s if completion_s is not None else end_s,
        metrics=metrics,
    )
    return DryEpisode(
        result=result,
        state=planner.state,
        planner_failure=failure,
        failed_in=planner.failed_in,
        gate_refusals=fakes.arm.refusals,
        steps=k,
        grasp_why=grasp.why(),
        placement_why=placement.why(),
    )


def run_dry_episode(
    spec: EpisodeSpec,
    layout: kd.RackSlotLayout,
    *,
    gate: bool,
    envelope: ReachabilityEnvelope | None = None,
    **kwargs: Any,
) -> EpisodeResult:
    """:func:`simulate_episode`, keeping only the :class:`EpisodeResult`."""
    return simulate_episode(spec, layout, gate=gate, envelope=envelope, **kwargs).result


class DryRunBackend:
    """Backend of the ``dry`` scenario (:class:`~kitchenbot.evaluation.runner.ScenarioBackend`).

    Reads the :class:`EvalConfig`: the split, ``config_path`` (see the module docstring) and
    ``extras["reach_gate"]``. Holds no resources.
    """

    def __init__(self, config: EvalConfig, layout: kd.RackSlotLayout | None = None) -> None:
        unknown = sorted(set(config.extras) - set(EXTRA_KEYS))
        if unknown:
            raise ScenarioConfigError(
                f"unknown extras {unknown} for the dry scenario; it reads {list(EXTRA_KEYS)}"
            )
        gate = config.extras.get("reach_gate", True)
        if not isinstance(gate, bool):
            raise ScenarioConfigError(f"extras['reach_gate'] must be a bool, got {gate!r}")
        self.gate = gate
        self.split = config.split
        self.layout = layout if layout is not None else kd.RackSlotLayout()
        if config.config_path is not None:
            self.sampler_scenario, self.ranges = read_eval_yaml(config.config_path, config.split)
        else:
            self.sampler_scenario = _DEFAULT_MIRROR
            self.ranges = RandomizationRanges.for_split(config.split)
        self.planner_config = dry_run_planner_config()
        self.envelope = ReachabilityEnvelope.load() if self.gate else None

    def resolved_config(self) -> Mapping[str, Any]:
        envelope = None
        if self.envelope is not None:
            envelope = {
                "seed": self.envelope.seed,
                "samples": self.envelope.samples,
                "spacing_m": self.envelope.spacing,
                "occupied_voxels": self.envelope.occupied_count,
            }
        return {
            "generator": GENERATOR_VERSION,
            "sampler_scenario": self.sampler_scenario,
            "randomization": self.ranges.to_dict(),
            "reach_gate": self.gate,
            "envelope": envelope,
            "control_hz": CONTROL_HZ,
            "settle_s": SETTLE_S,
            "stand_off_m": kd.ROBOT_STANDOFF_M,
            "pelvis_height_m": PELVIS_HEIGHT_M,
            "rack_frame_xyz_m": [float(v) for v in DRY_RUN_RACK_FRAME.position],
            "rack_layout": dataclasses.asdict(self.layout),
            "planner": dataclasses.asdict(self.planner_config),
        }

    def run_batch(self, seeds: Sequence[int]) -> list[EpisodeResult]:
        dims = SpawnDims(rack=self.layout)
        results = []
        for seed in seeds:
            spec = sample_episode(seed, self.split, self.sampler_scenario, self.ranges, dims)
            results.append(
                run_dry_episode(
                    spec,
                    self.layout,
                    gate=self.gate,
                    envelope=self.envelope,
                    planner_config=self.planner_config,
                )
            )
        return results

    def close(self) -> None:
        return None
