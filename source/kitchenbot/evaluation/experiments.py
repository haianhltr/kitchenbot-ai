"""Rows of the "Evaluation results" table in ``docs/experiments.md``.

``scripts/update_experiments.py`` appends one row per aggregate rate of a result document, plus
the median completion time where the document has one, to the table whose header is :data:`HEADER`.
A document is keyed by its file name (the last column): a name already in the table is skipped,
so running the script twice adds nothing the second time.

Only the table is touched; the rest of the file is written back byte for byte, with the file's
own line endings.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

SECTION = "## Evaluation results"

COLUMNS: tuple[str, ...] = (
    "Date",
    "Commit",
    "Scenario",
    "Split",
    "Policy",
    "Hand",
    "Base",
    "Metric",
    "Value",
    "95% CI",
    "Trials",
    "JSON",
)

HEADER = "| " + " | ".join(COLUMNS) + " |"


class TableError(Exception):
    """``docs/experiments.md`` does not contain the expected table."""


def _percent(x: float) -> str:
    return f"{100 * x:.1f}%"


def _commit_cell(run: Mapping[str, Any]) -> str:
    commit = run["git_commit"]
    short = commit[:7] if commit != "unknown" else "unknown"
    # A result from a modified tree cannot be reproduced from the commit alone; say so.
    return short + ("+dirty" if run["git_dirty"] else "")


def rows_for(doc: Mapping[str, Any], file_name: str) -> list[str]:
    """Table rows for one result document, in a fixed order.

    ``success_rate`` first, then the other rates in the document's order, then the median
    completion time if the document has successes with a completion time.
    """
    run, agg = doc["run"], doc["aggregates"]
    prefix = [
        run["started_at"][:10],
        _commit_cell(run),
        run["scenario"],
        run["split"],
        run["policy"],
        run["hand"],
        run["base"],
    ]
    suffix = f"`{file_name}`"

    def row(metric: str, value: str, ci: str, trials: str) -> str:
        cells = [*prefix, metric, value, ci, trials, suffix]
        return "| " + " | ".join(cells) + " |"

    rows = []
    rates = {"success_rate": agg["success"], **agg["rates"]}
    for name, rate in rates.items():
        rows.append(
            row(
                name,
                _percent(rate["rate"]),
                f"{_percent(rate['ci_low'])} to {_percent(rate['ci_high'])}",
                str(rate["n"]),
            )
        )
    completion = agg["continuous"].get("completion_time_s")
    if completion is not None and completion["n"] > 0:
        rows.append(
            row(
                "completion_time_s (median)",
                f"{completion['median']:.2f} s",
                "n/a",
                f"{completion['n']} of {agg['num_episodes']}",
            )
        )
    return rows


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def locate_table(lines: list[str]) -> tuple[int, int]:
    """Return ``(first_row, end)``: the index of the first data row and one past the last row.

    Finds the table under :data:`SECTION`, checks that its header has exactly :data:`COLUMNS`.
    """
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == SECTION)
    except StopIteration:
        raise TableError(f"no '{SECTION}' section") from None
    header = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("|")),
        None,
    )
    if header is None or header + 1 >= len(lines):
        raise TableError(f"no table under '{SECTION}'")
    if _cells(lines[header]) != list(COLUMNS):
        raise TableError(f"table header is not '{HEADER}'")
    if not set(lines[header + 1].replace("|", "").strip()) <= set("- :"):
        raise TableError("table header is not followed by a separator row")
    end = header + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return header + 2, end


def listed_names(lines: list[str]) -> set[str]:
    """File names in the last column of the table's data rows."""
    first, end = locate_table(lines)
    return {_cells(line)[-1].strip("`") for line in lines[first:end]}


def add_documents(
    text: str, documents: Iterable[tuple[str, Mapping[str, Any]]]
) -> tuple[str, list[str], list[str]]:
    """Append the rows of every ``(file_name, doc)`` not yet listed.

    Returns ``(new_text, added_names, skipped_names)``. Line endings of ``text`` are kept.
    """
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(newline)
    listed = listed_names(lines)
    new_rows: list[str] = []
    added: list[str] = []
    skipped: list[str] = []
    for name, doc in documents:
        if name in listed:
            skipped.append(name)
            continue
        listed.add(name)
        added.append(name)
        new_rows.extend(rows_for(doc, name))
    if new_rows:
        _, end = locate_table(lines)
        lines[end:end] = new_rows
    return newline.join(lines), added, skipped
