"""The ``dry`` scenario (#60): sampler, planner against fakes, judges, result document, CLI, plots.

No simulation. The episode logic is exercised through ``simulate_episode`` (which also returns
the planner's final state), the document path through ``run_evaluation`` and the command line.
"""

import copy
import dataclasses
import functools
import json
from collections import Counter

import pytest
from eval_helpers import REPO, load_script
from jsonschema import Draft202012Validator

from kitchenbot.controllers.reachability import (
    ENVELOPE_PATHS,
    FINGER_BASE_OFFSET_M,
    ReachabilityEnvelope,
)
from kitchenbot.evaluation import dry_run
from kitchenbot.evaluation.cli import main as evaluate_main
from kitchenbot.evaluation.dry_run import (
    DryRunBackend,
    read_eval_yaml,
    simulate_episode,
)
from kitchenbot.evaluation.failure_tags import FAILURE_TAG_VALUES, FailureTag
from kitchenbot.evaluation.judges import GRASP_LIFT_HEIGHT_M
from kitchenbot.evaluation.metrics import metrics_for
from kitchenbot.evaluation.results import load_results
from kitchenbot.evaluation.runner import (
    EvalConfig,
    ScenarioConfigError,
    file_sha256,
    get_scenario,
    run_evaluation,
    write_result,
)
from kitchenbot.evaluation.schema import load_schema, validate_result
from kitchenbot.evaluation.seeds import default_seeds, parse_seeds
from kitchenbot.planning.plate_to_rack import PlannerConfig, PlannerState
from kitchenbot.scene import kitchen_dims as kd
from kitchenbot.scene.randomization import RandomizationRanges, sample_episode

CONFIG_DIR = REPO / "configs" / "eval"
LAYOUT = kd.RackSlotLayout()
_ENVELOPES = {name: ReachabilityEnvelope.load(name) for name in ("waist0", "waist")}
_ENVELOPE = _ENVELOPES["waist0"]
S = PlannerState

N = 20  # the "twenty seeds" of the issue
SEEDS = {"seen": tuple(range(N)), "unseen": tuple(range(1000, 1000 + N))}
SPLITS = ("seen", "unseen")
GATES = (True, False)


def _config(split, gate, *, seeds=None, yaml=True, envelope=None, **kwargs):
    extras = {"reach_gate": gate}
    if envelope is not None:
        extras["gate_envelope"] = envelope
    return EvalConfig(
        scenario="dry",
        split=split,
        seeds=SEEDS[split] if seeds is None else seeds,
        num_envs=50,
        config_path=CONFIG_DIR / f"s4_{split}.yaml" if yaml else None,
        extras=extras,
        **kwargs,
    )


@functools.cache
def _doc(split, gate, envelope=None):
    return run_evaluation(_config(split, gate, envelope=envelope))


def doc(split, gate, envelope=None):
    """A validated twenty-seed document; a fresh copy, so a test may change it."""
    return copy.deepcopy(_doc(split, gate, envelope))


@functools.cache
def _episode(split, gate, seed, envelope="waist0"):
    return simulate_episode(
        sample_episode(seed, split, "S4"),
        LAYOUT,
        gate=gate,
        envelope=_ENVELOPES[envelope] if gate else None,
    )


def _episodes(split, gate, count, envelope="waist0"):
    """The first ``count`` seeds of the split, simulated once per seed for the whole module."""
    return tuple(
        _episode(split, gate, seed, envelope) for seed in default_seeds("S4", split)[:count]
    )


# ----------------------------------------------------------------------------- the document


@pytest.mark.parametrize("gate", GATES, ids=["gate-on", "gate-off"])
@pytest.mark.parametrize("split", SPLITS)
def test_twenty_seeds_validate_against_the_schema(split, gate):
    d = doc(split, gate)
    validate_result(d)  # schema plus the cross-field checks
    Draft202012Validator(load_schema()).validate(d)
    assert d["run"]["scenario"] == "dry" and d["run"]["split"] == split
    assert d["run"]["seeds"] == list(SEEDS[split]) and len(d["episodes"]) == N
    assert d["aggregates"]["num_episodes"] == N
    for episode in d["episodes"]:
        assert set(episode["metrics"]) == set(metrics_for("dry")) == set(metrics_for("S4"))
        assert episode["failure_tag"] == (
            "success" if episode["success"] else episode["failure_tag"]
        )
        assert episode["success"] or episode["failure_tag"] in FAILURE_TAG_VALUES
        assert episode["success"] == (episode["failure_tag"] == "success")


@pytest.mark.parametrize("gate", GATES, ids=["gate-on", "gate-off"])
def test_the_same_seed_gives_the_same_document_twice(gate):
    seeds = (3, 4, 5, 6)
    first = run_evaluation(_config("seen", gate, seeds=seeds))
    second = run_evaluation(_config("seen", gate, seeds=seeds))
    assert first["episodes"] == second["episodes"]
    assert first["aggregates"] == second["aggregates"]
    assert first["run"]["config_hash"] == second["run"]["config_hash"]
    differing = {k for k in first["run"] if first["run"][k] != second["run"][k]}
    assert differing <= {"started_at", "wall_clock_s"}


def test_an_episode_does_not_depend_on_the_other_seeds_of_the_batch():
    backend = DryRunBackend(_config("seen", False))
    alone = backend.run_batch([5])[0]
    assert backend.run_batch([2, 5, 9])[1] == alone
    assert backend.run_batch([5, 5])[0] == alone  # nothing leaks from one episode into the next


def test_the_document_is_marked_synthetic_with_the_config_recorded():
    d = doc("seen", True)
    run = d["run"]
    assert run["synthetic"] is True and run["gpu_name"] == "none"
    path = CONFIG_DIR / "s4_seen.yaml"
    assert run["config_path"] == path.as_posix() and run["config_sha256"] == file_sha256(path)
    spec = get_scenario("dry")
    assert spec.synthetic and not spec.uses_gpu


def test_a_real_looking_dry_document_is_rejected_by_the_schema():
    d = doc("seen", False)
    d["run"]["synthetic"] = False
    with pytest.raises(Exception, match="True|const|synthetic"):
        validate_result(d)


def test_update_experiments_and_plot_results_refuse_the_dry_result_by_default(tmp_path):
    path = write_result(
        doc("seen", False), tmp_path / "eval" / "20261023T100000Z_abababa_dry_seen.json"
    )
    accepted, refused = load_results([path.parent])
    assert accepted == [] and [r.name for r in refused] == [path.name]
    accepted, refused = load_results([path.parent], allow_synthetic=True)
    assert [r.name for r in accepted] == [path.name] and refused == []

    experiments = REPO / "docs" / "experiments.md"
    scratch = tmp_path / "experiments.md"
    scratch.write_bytes(experiments.read_bytes())
    update = load_script("update_experiments")
    assert update.main([str(path), "--doc", str(scratch)]) == 1  # named explicitly: refused
    assert scratch.read_bytes() == experiments.read_bytes()
    plot = load_script("plot_results")
    assert plot.main([str(path), "--out", str(tmp_path / "none")]) == 1
    assert not (tmp_path / "none").exists()


def test_plot_results_draws_the_dry_result_when_allowed(tmp_path, capsys):
    results = tmp_path / "eval"
    write_result(doc("seen", True), results / "20261023T100000Z_abababa_dry_seen.json")
    write_result(doc("unseen", True), results / "20261023T100100Z_abababa_dry_unseen.json")
    plot = load_script("plot_results")
    out = tmp_path / "figures"
    assert plot.main([str(results), "--out", str(out), "--allow-synthetic"]) == 0
    assert sorted(p.name for p in out.glob("*.png")) == [
        "failure_tags.png",
        "success_rates.png",
        "torso_error.png",
    ]
    assert all(p.stat().st_size > 5_000 for p in out.glob("*.png"))
    assert "wrote" in capsys.readouterr().out


# ----------------------------------------------------------------------------- the reach gate


# The gate pins below come from this geometry (#66, re-pinned by #67 and #77): both committed
# envelopes of the finger-base point (D012), the waist envelope's forward pitch sampled up to the
# 0.468 rad soft limit (``joint_limits.WAIST_LEAN_MAX_RAD``, 26.81 deg, #77; it was the 0.52 rad
# URDF limit, 29.79 deg, before), the 0.30 m stand-off, the pelvis at 0.75 m; the fakes' grasp
# frame on the top rim point nearest the robot (planning.geometry, #67), the counter-edge waypoint
# at the grasp x, and the rack frame of the #59 raised dishwasher with the rack pulled out
# (``DRY_RUN_RACK_FRAME``). The ``waist`` gate checks every target against the leaning envelope,
# also the transport and slot targets the planner commands upright. A check of each waypoint
# against the envelope of its commanded posture (lean for pre_grasp, grasp, lift; waist0 after)
# passes for 48 of 50 seen and 50 of 50 unseen seeds (#77 PR; 50 of 50 on both with the 29.79 deg
# envelope, #67 PR). Before #67, with #66's envelopes, every gate-on episode ended
# ``ik_infeasible``.

#: ``(successes, ik_infeasible)`` of the gate-on run over ``SEEDS``, per envelope and split.
#: 50 seeds per split (#77): waist0 seen 13 / 37, unseen 43 / 7; waist seen 48 / 2 (seeds 19
#: and 28, the grasp target 1.0 and 0.4 cm outside the capped envelope), unseen 50 / 0. The capped
#: pitch range moves only the ("waist", "seen") cell: it was (20, 0), 50 / 0 over 50 seeds (#67).
GATE_ON = {
    ("waist0", "seen"): (5, 15),
    ("waist0", "unseen"): (18, 2),
    ("waist", "seen"): (19, 1),
    ("waist", "unseen"): (20, 0),
}


@pytest.mark.parametrize("envelope", [None, "waist"], ids=["waist0", "waist"])
@pytest.mark.parametrize("split", SPLITS)
def test_the_gate_on_and_off_results_at_the_d012_geometry(split, envelope):
    on, off = doc(split, True, envelope), doc(split, False)
    on_n = on["aggregates"]["failure_tag_counts"]["ik_infeasible"]
    off_n = off["aggregates"]["failure_tag_counts"]["ik_infeasible"]
    assert (on["aggregates"]["success"]["count"], on_n) == GATE_ON[(envelope or "waist0", split)]
    assert off_n == 0 and off["aggregates"]["success"]["count"] == N
    assert on["run"]["config_hash"] != off["run"]["config_hash"]  # the option is hashed


#: How each gate-on episode of ``SEEDS`` ends: the state the planner failed in, or ``done``.
#: 50 seeds per split (#77): waist0 seen 30 pre_grasp, 7 grasp, 13 done; unseen 2 pre_grasp,
#: 5 grasp, 43 done; waist seen 2 grasp, 48 done (seed 19 is the grasp refusal of the first
#: twenty; 50 done before the cap, #67), unseen 50 done. No episode is refused after the lift.
END_STATE = {
    ("waist0", "seen"): {S.PRE_GRASP: 13, S.GRASP: 2, S.DONE: 5},
    ("waist0", "unseen"): {S.PRE_GRASP: 1, S.GRASP: 1, S.DONE: 18},
    ("waist", "seen"): {S.GRASP: 1, S.DONE: 19},
    ("waist", "unseen"): {S.DONE: 20},
}


@pytest.mark.parametrize("envelope", ["waist0", "waist"])
@pytest.mark.parametrize("split", SPLITS)
def test_where_the_gate_refuses_at_the_nominal_stand_off(split, envelope):
    episodes = _episodes(split, True, N, envelope)
    ends = Counter(e.failed_in or e.state for e in episodes)
    assert dict(ends) == END_STATE[(envelope, split)]
    for episode in episodes:
        if episode.state is S.DONE:
            assert episode.gate_refusals == 0 and episode.result.success
            continue
        assert episode.state is S.FAILED
        assert episode.planner_failure is FailureTag.IK_INFEASIBLE
        assert episode.gate_refusals >= 1
        assert episode.result.failure_tag is FailureTag.IK_INFEASIBLE
        assert episode.result.metrics["completion_time_s"] is None
        assert episode.result.metrics["grasp_success"] is False
        if episode.failed_in is S.PRE_GRASP:  # the first target is refused, nothing moved
            assert episode.steps <= 2


def test_the_waist_envelope_lets_more_episodes_through_than_waist0():
    for split in SPLITS:
        done = {
            name: sum(e.state is S.DONE for e in _episodes(split, True, N, name))
            for name in ("waist0", "waist")
        }
        assert done["waist"] > done["waist0"]


def test_a_closer_stand_off_lets_the_planner_past_pre_grasp():
    def past(stand_off_m):
        return sum(
            simulate_episode(
                sample_episode(seed, "seen", "S4"),
                LAYOUT,
                gate=True,
                envelope=_ENVELOPE,
                stand_off_m=stand_off_m,
            ).failed_in
            is not S.PRE_GRASP
            for seed in range(N)
        )

    # finger-base waist0 envelope, seeds 0 to 19 (#67 geometry): 7 of 20 get past pre_grasp at
    # 0.30 m, 15 at 0.25 m, all 20 at 0.15 m (19 at 0.20 m)
    assert (past(0.30), past(0.25), past(0.15)) == (7, 15, N)


@pytest.mark.parametrize("name", ["waist0", "waist"])
def test_the_resolved_config_names_the_envelope_that_gated(name):
    backend = DryRunBackend(_config("seen", True, envelope=name))
    assert backend.gate_envelope == name and backend.envelope.same_as(_ENVELOPES[name])
    block = backend.resolved_config()["envelope"]
    assert block["name"] == name
    assert block["file"] == ENVELOPE_PATHS[name].name
    assert block["tip_offset_m"] == list(FINGER_BASE_OFFSET_M)
    assert block["waist_range_deg"] == [list(r) for r in _ENVELOPES[name].waist_range_deg]
    assert block["occupied_voxels"] == _ENVELOPES[name].occupied_count
    json.dumps(block)  # plain JSON, hashable into run.config_hash


def test_gate_envelope_defaults_to_waist0_and_is_hashed():
    default = DryRunBackend(_config("seen", True))
    assert dry_run.DEFAULT_GATE_ENVELOPE == default.gate_envelope == "waist0"
    explicit = DryRunBackend(_config("seen", True, envelope="waist0"))
    waist = DryRunBackend(_config("seen", True, envelope="waist"))
    assert default.resolved_config() == explicit.resolved_config()
    assert waist.resolved_config() != default.resolved_config()
    assert waist.resolved_config()["envelope"]["file"] == "right_arm_envelope_waist.npz"
    on0, onw = doc("seen", True), doc("seen", True, "waist")
    assert on0["run"]["config_hash"] != onw["run"]["config_hash"]
    assert set(onw["run"]) == set(on0["run"])  # the run block has no envelope key
    # with the gate off the envelope is not loaded, so its name changes nothing
    off = DryRunBackend(_config("seen", False))
    off_waist = DryRunBackend(_config("seen", False, envelope="waist"))
    assert off.envelope is None and off.resolved_config()["envelope"] is None
    assert off_waist.resolved_config() == off.resolved_config()


def test_the_gate_off_is_the_extras_option_and_not_a_run_field():
    off, on, default = _config("seen", False), _config("seen", True), _config("seen", True)
    assert off.extras == {"reach_gate": False}
    assert off.to_dict() == on.to_dict()  # not in the run block
    assert DryRunBackend(off).gate is False
    assert DryRunBackend(on).gate is True
    no_extras = EvalConfig(scenario="dry", seeds=(0,))
    assert DryRunBackend(no_extras).gate is True  # on by default
    assert DryRunBackend(default).resolved_config()["reach_gate"] is True


def test_bad_extras_are_a_scenario_config_error():
    with pytest.raises(ScenarioConfigError, match="unknown extras.*speed"):
        DryRunBackend(EvalConfig(scenario="dry", seeds=(0,), extras={"speed": 1}))
    with pytest.raises(ScenarioConfigError, match="reach_gate"):
        DryRunBackend(EvalConfig(scenario="dry", seeds=(0,), extras={"reach_gate": "off"}))
    for bad in ("waist1", "right_arm_envelope.npz", True, None):
        with pytest.raises(ScenarioConfigError, match="gate_envelope.*one of"):
            DryRunBackend(EvalConfig(scenario="dry", seeds=(0,), extras={"gate_envelope": bad}))
    with pytest.raises(ValueError, match="extras keys"):
        EvalConfig(scenario="dry", seeds=(0,), extras={1: True})


# ----------------------------------------------------------------------------- planner and judges


def test_without_the_gate_the_fake_arm_reaches_done_on_the_seen_split():
    episodes = _episodes("seen", False, 50)  # the S4 trial count of the seen split
    done = [e for e in episodes if e.state is S.DONE]
    assert len(done) / len(episodes) >= 0.9  # most seeds; measured: all of them
    assert all(e.gate_refusals == 0 for e in episodes)


@pytest.mark.parametrize("split", SPLITS)
def test_without_the_gate_the_judges_confirm_what_the_planner_did(split):
    episodes = _episodes(split, False, N)
    succeeded = [e for e in episodes if e.result.success]
    assert len(succeeded) / len(episodes) >= 0.9
    for e in succeeded:
        metrics = e.result.metrics
        assert e.state is S.DONE and e.result.failure_tag is FailureTag.SUCCESS
        assert metrics["grasp_success"] and metrics["placement_success"]
        assert metrics["rack_collision"] is False and metrics["fell"] is False
        # the placement judge's 2 s window closed after the planner's whole sequence, under 30 s
        assert 5.0 < metrics["completion_time_s"] <= PlannerConfig().total_budget_s
        assert e.result.duration_s == metrics["completion_time_s"]
    for e in episodes:  # every failure is explained by a tag with the metrics to match
        if not e.result.success:
            assert e.result.metrics["completion_time_s"] is None


def test_the_unseen_slot_and_plate_are_exercised():
    specs = [sample_episode(s, "unseen", "S4") for s in range(1000, 1000 + N)]
    assert {s.rack_slot for s in specs} == {5} and {s.plate_diameter_m for s in specs} == {0.27}
    assert all(e.state is S.DONE for e in _episodes("unseen", False, N))


def test_the_dry_run_has_no_torso_so_it_reports_none():
    for e in _episodes("seen", False, 5):
        assert e.result.metrics["fell"] is False
        assert e.result.metrics["torso_orientation_error_deg"] == 0.0
        assert e.result.traces is None


def test_with_the_default_planner_config_every_seen_episode_is_grasped_and_placed():
    # regression for the lift gap the first dry run found (the lift waypoint was aimed at exactly
    # 15 cm, the exit tolerance left the plate at 14.9 cm and every episode ended in `drop`)
    episodes = _episodes("seen", False, 50)
    assert len(episodes) == 50
    for e in episodes:
        assert e.result.success and e.state is S.DONE and e.planner_failure is None
        assert e.result.failure_tag is not FailureTag.DROP
        assert e.grasp_why.startswith("grasped"), e.grasp_why
        assert e.result.metrics["grasp_success"] is True
    assert DryRunBackend(_config("seen", False)).planner_config == PlannerConfig()
    assert PlannerConfig().lift_height_m == GRASP_LIFT_HEIGHT_M == 0.15  # the protocol number


def test_a_budget_the_run_cannot_meet_is_a_timeout():
    spec = sample_episode(0, "seen", "S4")
    short = dataclasses.replace(PlannerConfig(), total_budget_s=6.0)
    episode = simulate_episode(spec, LAYOUT, gate=False, planner_config=short)
    assert episode.state is S.FAILED and episode.planner_failure is FailureTag.TIMEOUT
    assert episode.result.failure_tag is FailureTag.TIMEOUT and not episode.result.success


# ----------------------------------------------------------------------------- the YAML


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("scenario", ["S2", "S3", "S4"])
def test_the_committed_yamls_are_read_for_the_sampler_scenario_and_ranges(scenario, split):
    path = CONFIG_DIR / f"{scenario.lower()}_{split}.yaml"
    assert read_eval_yaml(path, split) == (scenario, RandomizationRanges.for_split(split))


@pytest.mark.parametrize("scenario", ["S2", "S3"])
def test_an_s2_or_s3_yaml_changes_the_draws_not_the_metric_set(scenario):
    seeds = (0, 1, 2)
    s4 = run_evaluation(_config("seen", False, seeds=seeds))
    other = run_evaluation(
        EvalConfig(
            scenario="dry",
            seeds=seeds,
            config_path=CONFIG_DIR / f"{scenario.lower()}_seen.yaml",
            extras={"reach_gate": False},
        )
    )
    validate_result(other)
    assert other["run"]["scenario"] == "dry" and other["run"]["config_path"].endswith(
        f"{scenario.lower()}_seen.yaml"
    )
    assert other["run"]["config_hash"] != s4["run"]["config_hash"]
    assert set(other["episodes"][0]["metrics"]) == set(metrics_for("S4"))
    backend = DryRunBackend(
        EvalConfig(
            scenario="dry", seeds=seeds, config_path=CONFIG_DIR / f"{scenario.lower()}_seen.yaml"
        )
    )
    assert backend.resolved_config()["sampler_scenario"] == scenario


def test_without_a_config_the_dry_run_mirrors_s4_with_the_split_ranges():
    backend = DryRunBackend(EvalConfig(scenario="dry", split="unseen", seeds=(1000,)))
    assert backend.sampler_scenario == "S4"
    assert backend.ranges == RandomizationRanges.for_split("unseen")
    with_yaml = DryRunBackend(_config("unseen", True))
    assert (
        with_yaml.resolved_config() == backend.resolved_config()
    )  # the committed YAML is the default


def test_an_edited_randomization_block_changes_the_episodes(tmp_path):
    text = (CONFIG_DIR / "s4_seen.yaml").read_text(encoding="utf-8")
    assert "plate_yaw_half_range_deg: 30.0" in text
    edited = tmp_path / "s4_seen.yaml"
    edited.write_text(
        text.replace("plate_yaw_half_range_deg: 30.0", "plate_yaw_half_range_deg: 0.0")
    )
    backend = DryRunBackend(
        EvalConfig(scenario="dry", seeds=(0,), config_path=edited, extras={"reach_gate": False})
    )
    assert backend.ranges.plate_yaw_half_range_deg == 0.0
    base = DryRunBackend(_config("seen", False))
    assert backend.resolved_config() != base.resolved_config()
    assert (
        run_evaluation(_config("seen", False, seeds=(0,)))["run"]["config_hash"]
        != (
            run_evaluation(
                EvalConfig(
                    scenario="dry", seeds=(0,), config_path=edited, extras={"reach_gate": False}
                )
            )["run"]["config_hash"]
        )
    )


@pytest.mark.parametrize(
    ("name", "split", "message"),
    [
        ("s0_seen.yaml", "seen", "scenario 'S0'"),
        ("s1_unseen.yaml", "unseen", "scenario 'S1'"),
        ("s4_unseen.yaml", "seen", "split 'unseen' but the run is on split 'seen'"),
        ("s4_seen.yaml", "unseen", "split 'seen' but the run is on split 'unseen'"),
    ],
)
def test_a_yaml_the_dry_run_cannot_mirror_is_a_scenario_config_error(name, split, message):
    seeds = (0,) if split == "seen" else (1000,)
    with pytest.raises(ScenarioConfigError, match=message):
        DryRunBackend(
            EvalConfig(scenario="dry", split=split, seeds=seeds, config_path=CONFIG_DIR / name)
        )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("- just\n- a list\n", "mapping"),
        ("scenario: S4\nsplit: seen\nrandomization: {split: seen, bogus: 1}\n", "randomization"),
        ("scenario: S4\nsplit: seen\nrandomization: {split: unseen}\n", "randomization.split"),
        ("scenario: [unclosed\n", "cannot read as YAML"),
    ],
)
def test_a_broken_yaml_is_a_scenario_config_error(tmp_path, content, message):
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ScenarioConfigError, match=message):
        read_eval_yaml(path, "seen")


# ----------------------------------------------------------------------------- the command line


def test_the_command_line_runs_the_dry_scenario_end_to_end(tmp_path, capsys):
    argv = [
        "--scenario",
        "dry",
        "--config",
        str(CONFIG_DIR / "s4_seen.yaml"),
        "--seeds",
        "0-4",
        "--out",
        str(tmp_path),
    ]
    assert evaluate_main(argv) == 0
    out = capsys.readouterr().out
    assert "scenario dry | split seen" in out and "SYNTHETIC RUN" in out
    # the gate is on by default (waist0): seeds 1 and 3 succeed, 0, 2 and 4 are refused (GATE_ON)
    assert "success: 2/5" in out and "failure tags: fall 0, ik_infeasible 3" in out
    written = list(tmp_path.glob("*_dry_seen.json"))
    assert len(written) == 1
    from kitchenbot.evaluation.results import load_result

    assert load_result(written[0]).doc["run"]["scenario"] == "dry"


def test_the_command_line_gates_on_the_waist_envelope(tmp_path, capsys):
    argv = ["--scenario", "dry", "--config", str(CONFIG_DIR / "s4_seen.yaml"), "--seeds", "0-4",
            "--extra", "gate_envelope=waist", "--out", str(tmp_path)]
    assert evaluate_main(argv) == 0
    out = capsys.readouterr().out
    assert "extras gate_envelope=waist" in out
    # with the waist envelope seeds 0 to 4 all succeed at the #67 geometry (END_STATE)
    assert "success: 5/5" in out and "failure tags: fall 0, ik_infeasible 0" in out
    assert len(list(tmp_path.glob("*_dry_seen.json"))) == 1


def test_the_default_seeds_of_the_dry_run_are_the_s4_trial_counts():
    assert parse_seeds(None, "dry", "seen") == list(range(50))
    assert parse_seeds(None, "dry", "unseen") == list(range(1000, 1050))


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--config", "configs/eval/s0_seen.yaml"], "scenario 'S0'"),
        (
            ["--split", "unseen", "--seeds", "1000-1001", "--config", "configs/eval/s4_seen.yaml"],
            "split 'seen'",
        ),
    ],
)
def test_the_command_line_reports_a_rejected_config_as_a_usage_error(
    tmp_path, capsys, args, message
):
    args = [a if not a.startswith("configs/") else str(REPO / a) for a in args]
    assert evaluate_main(["--scenario", "dry", *args, "--out", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and message in err
    assert list(tmp_path.glob("*.json")) == []


def test_the_module_imports_no_simulator():
    import subprocess
    import sys

    code = (
        "import sys, kitchenbot.evaluation.dry_run, kitchenbot.planning.fakes\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "('isaaclab', 'isaacsim', 'omni', 'pxr', 'torch')]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


def test_the_module_surface():
    assert dry_run.GENERATOR_VERSION == "dry-v2" and dry_run.CONTROL_HZ == 50.0  # v2: #67
    assert set(dry_run.MIRRORABLE_SCENARIOS) == {"S2", "S3", "S4"}
    assert dry_run.EXTRA_KEYS == ("reach_gate", "gate_envelope")
