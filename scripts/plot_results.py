"""Turn result JSONs from scripts/evaluate.py into figures (docs/evaluation.md section 4).

    python scripts/plot_results.py                          # everything in outputs/eval/
    python scripts/plot_results.py a.json b.json --out figs

Writes success_rates.png, failure_tags.png and torso_error.png (deterministic names) to
``outputs/figures/`` (git-ignored) unless --out is given. Run from the repository root.
Synthetic results (the fake scenario) are refused unless --allow-synthetic is given, and then
the figures say so. Exit status: 0 ok, 1 a named file was refused, 2 bad input.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from kitchenbot.evaluation.plotting import render_all, select_runs
from kitchenbot.evaluation.results import DEFAULT_RESULTS_DIR, ResultError, load_results

DEFAULT_FIGURES_DIR = Path("outputs") / "figures"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plot_results.py", description=__doc__.split("\n")[0])
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=f"result JSON files or directories of them (default {DEFAULT_RESULTS_DIR.as_posix()})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help=f"output directory (default {DEFAULT_FIGURES_DIR.as_posix()})",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="also plot synthetic (fake scenario) results; the figures are marked as synthetic",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        accepted, refused = load_results(
            args.paths or [DEFAULT_RESULTS_DIR], allow_synthetic=args.allow_synthetic
        )
    except ResultError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for result in refused:
        print(f"refused synthetic result {result.name} (use --allow-synthetic)", file=sys.stderr)
    if not accepted:
        print("error: nothing to plot", file=sys.stderr)
        return 1 if any(r.explicit for r in refused) else 2
    runs, superseded = select_runs(accepted)
    for name in superseded:
        print(f"note: {name} is superseded by a newer run of the same configuration")
    for path in render_all(runs, args.out):
        print(f"wrote {path.as_posix()}")
    return 1 if any(r.explicit for r in refused) else 0


if __name__ == "__main__":
    sys.exit(main())
