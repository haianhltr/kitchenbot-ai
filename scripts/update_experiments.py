"""Append measured rows to the "Evaluation results" table of docs/experiments.md.

    python scripts/update_experiments.py                        # everything in outputs/eval/
    python scripts/update_experiments.py outputs/eval/x.json    # one file
    python scripts/update_experiments.py --doc /tmp/copy.md     # another file, never the real one

One row per aggregate rate of each result document and one per continuous metric it carries
(mean with the median in parentheses, bootstrap interval of the mean, number of episodes with the
metric). A document already listed (keyed by its file name in the JSON column) is
skipped, so the script is idempotent. Synthetic results (the fake scenario) are refused unless
--allow-synthetic is given: never commit those rows. Run from the repository root.
Exit status: 0 ok, 1 a named file was refused, 2 bad input or table not found.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from kitchenbot.evaluation.experiments import TableError, add_documents
from kitchenbot.evaluation.results import DEFAULT_RESULTS_DIR, ResultError, load_results

DEFAULT_DOC = Path("docs") / "experiments.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="update_experiments.py", description=__doc__.split("\n")[0]
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=f"result JSON files or directories of them (default {DEFAULT_RESULTS_DIR.as_posix()})",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=DEFAULT_DOC,
        help=f"markdown file holding the table (default {DEFAULT_DOC.as_posix()})",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="also add synthetic (fake scenario) results; use only with --doc on a scratch copy",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        accepted, refused = load_results(
            args.paths or [DEFAULT_RESULTS_DIR], allow_synthetic=args.allow_synthetic
        )
        # newline="" keeps the file's own line endings through the read and the write.
        with open(args.doc, encoding="utf-8", newline="") as handle:
            text = handle.read()
        new_text, added, skipped = add_documents(text, [(r.name, r.doc) for r in accepted])
    except (ResultError, TableError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for result in refused:
        print(f"refused synthetic result {result.name} (use --allow-synthetic)", file=sys.stderr)
    for name in skipped:
        print(f"already listed: {name}")
    if added:
        with open(args.doc, "w", encoding="utf-8", newline="") as handle:
            handle.write(new_text)
    print(f"{len(added)} document(s) added to {args.doc.as_posix()}, {len(skipped)} skipped")
    return 1 if any(r.explicit for r in refused) else 0


if __name__ == "__main__":
    sys.exit(main())
