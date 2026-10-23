"""Command line of the evaluator. ``scripts/evaluate.py`` calls :func:`main`.

Example (docs/evaluation.md section 4)::

    python scripts/evaluate.py --scenario S4 --split unseen --seeds 1000-1049 \
        --num_envs 50 --headless

Exit status: 0 on success, 2 for invalid arguments, 3 for a scenario that is registered but not
implemented yet.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kitchenbot.evaluation.runner import (
    BASES,
    DEFAULT_OUTPUT_DIR,
    HANDS,
    POLICIES,
    EvalConfig,
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
        help="S0 to S4, or fake (synthetic, no simulation, CI only)",
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
        default=50,
        help="parallel environments; seeds run in batches of this size (default 50)",
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
        help="fixed pelvis or floating base with the balance policy active (default floating)",
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
            num_envs=args.num_envs,
            policy=args.policy,
            hand=args.hand,
            base=args.base,
            headless=args.headless,
        )
    except ValueError as exc:
        parser.error(str(exc))  # exits with status 2

    try:
        doc = run_evaluation(config)
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED

    path = write_result(doc, _resolve_out(args.out, doc))
    if config.split == "unseen":
        print(HELD_OUT_NOTICE)
    print(format_summary(doc))
    print(f"wrote {path.as_posix()}")
    return 0
