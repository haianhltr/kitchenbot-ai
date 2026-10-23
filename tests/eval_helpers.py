"""Helpers shared by the evaluation tests: result documents built through the fake scenario."""

import copy
import functools
import importlib.util
from pathlib import Path

from kitchenbot.evaluation.runner import EvalConfig, run_evaluation, write_result
from kitchenbot.evaluation.schema import validate_result
from kitchenbot.evaluation.stats import aggregate_episodes

REPO = Path(__file__).resolve().parents[1]


@functools.cache
def _base_doc(split: str, policy: str, seeds: tuple[int, ...]) -> dict:
    return run_evaluation(
        EvalConfig(scenario="fake", split=split, policy=policy, seeds=seeds, num_envs=50)
    )


def fake_doc(
    split: str = "seen",
    policy: str = "own",
    *,
    seeds: tuple[int, ...] = (),
    started_at: str | None = None,
    traces: bool = True,
) -> dict:
    """A validated document from the fake scenario, optionally without traces.

    Returns a fresh deep copy each time, so a test may modify it.
    """
    doc = copy.deepcopy(_base_doc(split, policy, tuple(seeds)))
    if started_at is not None:
        doc["run"]["started_at"] = started_at
    if not traces:
        for episode in doc["episodes"]:
            del episode["traces"]
    validate_result(doc)
    return doc


def as_real_s4(doc: dict, commit: str = "ab" * 20, dirty: bool = False) -> dict:
    """The same document relabelled as a real S4 run. The fake scenario mirrors S4's metrics."""
    doc = copy.deepcopy(doc)
    doc["run"].update(
        scenario="S4",
        synthetic=False,
        gpu_name="NVIDIA GeForce RTX 5080",
        git_commit=commit,
        git_dirty=dirty,
    )
    validate_result(doc)
    return doc


def scenario_doc(scenario: str, episodes: list[dict]) -> dict:
    """A real document of ``scenario`` holding exactly ``episodes`` (dicts without ``seed``)."""
    doc = as_real_s4(fake_doc("unseen", seeds=tuple(range(1000, 1000 + len(episodes)))))
    doc["run"]["scenario"] = scenario
    doc["episodes"] = [{"seed": 1000 + i, **e} for i, e in enumerate(episodes)]
    doc["aggregates"] = aggregate_episodes(scenario, doc["episodes"])
    validate_result(doc)
    return doc


def write_doc(directory: Path, name: str, doc: dict) -> Path:
    return write_result(doc, directory / name)


def load_script(name: str):
    """Import ``scripts/<name>.py`` as a module (scripts is not a package)."""
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
