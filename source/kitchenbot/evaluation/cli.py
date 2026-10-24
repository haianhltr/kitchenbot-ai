"""Command line of the evaluator. ``scripts/evaluate.py`` calls :func:`main`.

Example (docs/evaluation.md section 4)::

    python scripts/evaluate.py --scenario S4 --split unseen --seeds 1000-1049 \
        --num_envs 50 --headless --config configs/eval/s4_unseen.yaml

``--config`` names the committed ``configs/eval`` YAML the run was started from; its path and
SHA256 go into the result (``run.config_path``, ``run.config_sha256``). Of its contents the
command line reads only ``num_envs``, and only when ``--num_envs`` is not given (#78): the flag
overrides the file, a file without the key or no ``--config`` means
``runner.DEFAULT_NUM_ENVS``, and ``run.num_envs`` records the value used. A scenario's loader may
read more (``dry`` does, for the sampler scenario and ranges). ``--base`` and the other options
are not read from the file: the fixed-pelvis runs of S2 and S3 pass ``--base fixed`` with their
``configs/eval/s<N>_<split>_fixed.yaml`` (#78).

``--extra KEY=VALUE`` (repeatable, value a YAML scalar) fills ``EvalConfig.extras``, the options
one scenario reads: ``--scenario dry --extra reach_gate=false`` runs the dry scenario without the
reachability gate. A key the scenario does not read is a usage error. Extras are not in the
``run`` block; they reach ``run.config_hash`` through the scenario's resolved config.

Exit status: 0 on success, 2 for invalid arguments or a configuration the scenario rejects, 3 for
a scenario that is registered but not implemented yet.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from kitchenbot.evaluation.runner import (
    BASES,
    DEFAULT_NUM_ENVS,
    DEFAULT_OUTPUT_DIR,
    HANDS,
    POLICIES,
    EvalConfig,
    ScenarioConfigError,
    default_output_path,
    run_evaluation,
    scenario_names,
    write_result,
)
from kitchenbot.evaluation.seeds import SPLITS, parse_seeds

#: Printed before the summary of every run on the unseen split.
HELD_OUT_NOTICE = "held-out split: report only, do not tune on these seeds"

EXIT_USAGE = 2
EXIT_NOT_IMPLEMENTED = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description=(
            "Run one evaluation scenario once per seed and write a schema-checked JSON result "
            "(docs/evaluation.md). Every reported metric must come from this script."
        ),
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=scenario_names(),
        help=(
            "S0 to S4, or fake or dry (synthetic, no simulation, CI only; dry chains the sampler, "
            "the planner against fakes and the judges)"
        ),
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="seen",
        help=(
            "seed split: seen (0-99, for development; default) or unseen (1000-1049, held out: "
            "report only, do not tune on these seeds)"
        ),
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "a count N (first N seeds of the split), a comma list with ranges (0,5,10-12 or "
            "1000-1049), or omitted for the split's default trial count; a single explicit seed "
            "is written 7-7"
        ),
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help=(
            "parallel environments; seeds run in batches of this size. Default: the num_envs of "
            f"the --config YAML, else {DEFAULT_NUM_ENVS}; the flag overrides the YAML"
        ),
    )
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument(
        "--policy", choices=POLICIES, default="own", help="lower-body policy (default own)"
    )
    parser.add_argument(
        "--hand", choices=HANDS, default="dex3", help="hand controller (default dex3)"
    )
    parser.add_argument(
        "--base",
        choices=BASES,
        default="floating",
        help=(
            "fixed pelvis or floating base with the balance policy active (default floating). "
            "Not read from --config: start a fixed-pelvis S2 or S3 run with --base fixed and "
            "its configs/eval/s<N>_<split>_fixed.yaml"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "the committed configs/eval/*.yaml this run was started from (optional): its path "
            "and SHA256 are recorded in the result. The evaluator reads only its num_envs, "
            "when --num_envs is not given (the dry scenario also reads its scenario, split and "
            "ranges)"
        ),
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "an option only one scenario reads, repeatable; the value is a YAML scalar (false, "
            "0.3, text). dry reads reach_gate (default true): --extra reach_gate=false turns the "
            "reachability gate off; and gate_envelope (waist0, the default, or waist): the "
            "committed envelope the gate loads. The result's run.config_hash covers them"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "output file (ending in .json) or directory; default "
            f"{DEFAULT_OUTPUT_DIR.as_posix()}/<timestamp>_<commit>_<scenario>_<split>.json"
        ),
    )
    return parser


def parse_extras(items: Sequence[str]) -> dict[str, Any]:
    """``{key: value}`` from ``--extra KEY=VALUE`` items; the value is parsed as a YAML scalar.

    ``false``, ``0.3``, ``7`` and ``text`` become a bool, a float, an int and a string; an empty
    value is ``None``. Raises ``ValueError`` for an item without ``=`` or with an empty key, for a
    repeated key and for a value that is not valid YAML.
    """
    extras: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--extra expects KEY=VALUE, got {item!r}")
        if key in extras:
            raise ValueError(f"--extra {key} given twice")
        try:
            extras[key] = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"--extra {key}: cannot read {raw!r} as a YAML value: {exc}") from exc
    return extras


def yaml_num_envs(path: Path | str) -> int | None:
    """The ``num_envs`` of a ``configs/eval`` YAML, or ``None`` if the file has no such key.

    Raises ``ValueError`` if the file cannot be read as a YAML mapping or its ``num_envs`` is not
    a positive integer.
    """
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"--config {path.as_posix()}: cannot read as YAML: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"--config {path.as_posix()}: expected a mapping at the top level")
    if "num_envs" not in data:
        return None
    value = data["num_envs"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"--config {path.as_posix()}: num_envs must be a positive integer, got {value!r}"
        )
    return value


def resolve_num_envs(flag: int | None, config_path: Path | str | None) -> int:
    """The ``num_envs`` a run uses: the flag if given, else the YAML's, else the default.

    The YAML is read only when the flag is absent. A ``config_path`` that is not a file is left
    to :class:`EvalConfig` to reject, so the error names the missing file.
    """
    if flag is not None:
        return flag
    if config_path is not None and Path(config_path).is_file():
        from_yaml = yaml_num_envs(config_path)
        if from_yaml is not None:
            return from_yaml
    return DEFAULT_NUM_ENVS


def format_summary(doc: Mapping[str, Any]) -> str:
    """Human-readable lines for the numbers in ``doc``; this is what the run prints."""
    run, agg = doc["run"], doc["aggregates"]

    def rate_line(label: str, r: Mapping[str, Any]) -> str:
        return (
            f"  {label}: {r['count']}/{r['n']} = {100 * r['rate']:.1f}% "
            f"(95% Wilson {100 * r['ci_low']:.1f}% to {100 * r['ci_high']:.1f}%)"
        )

    lines = [
        f"scenario {run['scenario']} | split {run['split']} | policy {run['policy']} | "
        f"hand {run['hand']} | base {run['base']} | {agg['num_episodes']} trials | "
        f"num_envs {run['num_envs']}",
    ]
    if run["synthetic"]:
        lines.append("SYNTHETIC RUN: generated from seeds, not a measurement.")
    lines.append(rate_line("success", agg["success"]))
    for name, r in agg["rates"].items():
        lines.append(rate_line(name, r))
    for name, c in agg["continuous"].items():
        if c["n"] == 0:
            lines.append(f"  {name}: no values")
            continue
        std = "n/a" if c["std"] is None else f"{c['std']:.3f}"
        lines.append(
            f"  {name}: mean {c['mean']:.3f}, std {std}, median {c['median']:.3f} (n={c['n']})"
        )
    counts = ", ".join(f"{tag} {n}" for tag, n in agg["failure_tag_counts"].items())
    lines.append(f"  failure tags: {counts}")
    if "config_path" in run:
        lines.append(f"config {run['config_path']} | sha256 {run['config_sha256'][:12]}")
    lines.append(
        f"commit {run['git_commit'][:12]}{' (dirty)' if run['git_dirty'] else ''} | "
        f"gpu {run['gpu_name']} | wall-clock {run['wall_clock_s']:.2f} s"
    )
    return "\n".join(lines)


def _resolve_out(out: Path | None, doc: Mapping[str, Any]) -> Path:
    if out is None:
        return default_output_path(doc)
    if out.suffix.lower() == ".json":
        return out
    return default_output_path(doc, out)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluator. Returns the process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        seeds = parse_seeds(args.seeds, args.scenario, args.split)
        config = EvalConfig(
            scenario=args.scenario,
            split=args.split,
            seeds=tuple(seeds),
            num_envs=resolve_num_envs(args.num_envs, args.config),
            policy=args.policy,
            hand=args.hand,
            base=args.base,
            headless=args.headless,
            config_path=args.config,
            extras=parse_extras(args.extra),
        )
    except ValueError as exc:
        parser.error(str(exc))  # exits with status 2

    try:
        doc = run_evaluation(config)
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED
    except ScenarioConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    path = write_result(doc, _resolve_out(args.out, doc))
    if config.split == "unseen":
        print(HELD_OUT_NOTICE)
    print(format_summary(doc))
    if config.extras:
        print("extras " + ", ".join(f"{key}={value}" for key, value in config.extras.items()))
    print(f"wrote {path.as_posix()}")
    return 0
