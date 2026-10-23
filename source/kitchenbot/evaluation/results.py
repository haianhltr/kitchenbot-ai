"""Reading result files for ``scripts/plot_results.py`` and ``scripts/update_experiments.py``.

Every document is parsed and checked with :func:`kitchenbot.evaluation.schema.validate_result`
before anyone uses it. Documents with ``run.synthetic`` true (the ``fake`` scenario) are refused
unless the caller opts in: their numbers describe a generator, not the robot.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema.exceptions import ValidationError

from kitchenbot.evaluation.schema import validate_result

#: Where ``scripts/evaluate.py`` writes by default.
DEFAULT_RESULTS_DIR = Path("outputs") / "eval"


class ResultError(Exception):
    """A result file is missing, unreadable, or does not validate."""


@dataclass(frozen=True)
class ResultFile:
    """One parsed, validated result document.

    ``name`` is the file name, the key by which ``docs/experiments.md`` lists the document.
    ``explicit`` is true when the file was named on the command line, false when it was found by
    scanning a directory.
    """

    path: Path
    doc: dict[str, Any]
    explicit: bool

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def synthetic(self) -> bool:
        return bool(self.doc["run"]["synthetic"])


def _chronological(path: Path) -> tuple[str, int]:
    """Sort key: file name, with a rerun ``<stem>-2.json`` after ``<stem>.json``."""
    match = re.fullmatch(r"(.*?)(?:-(\d+))?", path.stem)
    return (match.group(1), int(match.group(2) or 1))


def find_result_files(paths: Sequence[Path | str]) -> list[tuple[Path, bool]]:
    """Expand ``paths`` into ``(file, explicit)`` pairs, sorted by file name within a directory.

    A directory contributes its ``*.json`` files (``explicit`` false); a file is taken as given
    (``explicit`` true). A file name appears once even if several paths lead to it.
    """
    found: list[tuple[Path, bool]] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidates = [(p, False) for p in sorted(path.glob("*.json"), key=_chronological)]
        elif path.is_file():
            candidates = [(path, True)]
        else:
            raise ResultError(f"{path}: no such file or directory")
        for candidate, explicit in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append((candidate, explicit))
    return found


def load_result(path: Path, explicit: bool = True) -> ResultFile:
    """Parse and validate one result file. Raises :class:`ResultError` on any problem."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResultError(f"{path}: cannot read as JSON: {exc}") from exc
    try:
        validate_result(doc)
    except ValidationError as exc:
        where = "/".join(str(p) for p in exc.absolute_path) or "(document)"
        raise ResultError(f"{path}: not a valid result at {where}: {exc.message}") from exc
    except ValueError as exc:
        raise ResultError(f"{path}: inconsistent result: {exc}") from exc
    return ResultFile(path=Path(path), doc=doc, explicit=explicit)


def load_results(
    paths: Iterable[Path | str], *, allow_synthetic: bool = False
) -> tuple[list[ResultFile], list[ResultFile]]:
    """Load every result under ``paths``. Returns ``(accepted, refused)``.

    ``refused`` are the synthetic documents skipped because ``allow_synthetic`` is false. Raises
    :class:`ResultError` if anything cannot be read or does not validate, or if no file is found.
    """
    files = find_result_files(list(paths))
    if not files:
        raise ResultError("no result JSON files found")
    accepted: list[ResultFile] = []
    refused: list[ResultFile] = []
    for path, explicit in files:
        result = load_result(path, explicit)
        (refused if result.synthetic and not allow_synthetic else accepted).append(result)
    return accepted, refused
