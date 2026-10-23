"""CI smoke test: evaluate.py, plot_results.py and update_experiments.py chained on fake data.

Runs the three scripts as real subprocesses with their default paths (outputs/eval and
outputs/figures under a temporary working directory) and a scratch copy of docs/experiments.md,
so the committed document is never touched.
"""

import os
import subprocess
import sys

from eval_helpers import REPO

from kitchenbot.evaluation.experiments import COLUMNS, locate_table

SCRIPTS = REPO / "scripts"
DOC = REPO / "docs" / "experiments.md"


def _run(script, *args, cwd):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )


def test_fake_results_flow_into_figures_and_table_rows(tmp_path):
    before = DOC.read_bytes()
    doc = tmp_path / "experiments.md"
    doc.write_bytes(before)

    for policy in ("own", "agile"):
        _run(
            "evaluate.py",
            "--scenario",
            "fake",
            "--split",
            "unseen",
            "--policy",
            policy,
            cwd=tmp_path,
        )
    results = sorted((tmp_path / "outputs" / "eval").glob("*.json"))
    assert len(results) == 2

    plotted = _run("plot_results.py", "--allow-synthetic", cwd=tmp_path)
    figures = sorted(p.name for p in (tmp_path / "outputs" / "figures").glob("*.png"))
    assert figures == ["failure_tags.png", "success_rates.png", "torso_error.png"]
    assert plotted.stdout.count("wrote ") == 3

    updated = _run("update_experiments.py", "--allow-synthetic", "--doc", str(doc), cwd=tmp_path)
    assert "2 document(s) added" in updated.stdout
    lines = doc.read_text(encoding="utf-8").splitlines()
    first, end = locate_table(lines)
    rows = lines[first:end]
    assert len(rows) == 14  # seven rows per document: five rates and two continuous metrics
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        assert len(cells) == len(COLUMNS) and cells[COLUMNS.index("Scenario")] == "fake"
    assert {r.split("|")[-2].strip().strip("`") for r in rows} == {p.name for p in results}

    again = _run("update_experiments.py", "--allow-synthetic", "--doc", str(doc), cwd=tmp_path)
    assert "0 document(s) added to" in again.stdout and ", 2 skipped" in again.stdout

    # Without the flag the scripts do not touch synthetic results.
    refused = subprocess.run(
        [sys.executable, str(SCRIPTS / "update_experiments.py"), "--doc", str(doc)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert refused.returncode == 0 and "refused synthetic" in refused.stderr

    assert DOC.read_bytes() == before
