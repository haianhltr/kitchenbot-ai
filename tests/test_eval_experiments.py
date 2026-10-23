"""docs/experiments.md table generation: rows, idempotence, refusal of synthetic results."""

import copy

import pytest
from eval_helpers import REPO, as_real_s4, fake_doc, load_script, write_doc

from kitchenbot.evaluation.experiments import (
    COLUMNS,
    HEADER,
    TableError,
    add_documents,
    listed_names,
    locate_table,
    rows_for,
)

DOC = REPO / "docs" / "experiments.md"


def _cells(row):
    return [c.strip() for c in row.strip().strip("|").split("|")]


# ----------------------------------------------------------------------------- the committed doc


def test_committed_doc_has_the_expected_table_and_no_synthetic_rows():
    lines = DOC.read_text(encoding="utf-8").splitlines()
    first, end = locate_table(lines)
    assert lines[first - 2] == HEADER
    for row in lines[first:end]:
        cells = _cells(row)
        assert len(cells) == len(COLUMNS)
        assert cells[COLUMNS.index("Scenario")] != "fake", row
        assert "_fake_" not in cells[-1], row


# ----------------------------------------------------------------------------- rows


def test_rows_of_a_real_s4_document():
    doc = as_real_s4(fake_doc("unseen"), commit="1234567" + "0" * 33)
    name = "20261023T120000Z_1234567_S4_unseen.json"
    rows = rows_for(doc, name)
    metrics = [_cells(r)[COLUMNS.index("Metric")] for r in rows]
    assert metrics == [
        "success_rate",
        "fall_rate",
        "grasp_success_rate",
        "placement_success_rate",
        "collision_rate",
        "completion_time_s (median)",
    ]
    for row in rows:
        cells = _cells(row)
        assert len(cells) == len(COLUMNS)
        assert cells[:7] == [
            doc["run"]["started_at"][:10],
            "1234567",
            "S4",
            "unseen",
            "own",
            "dex3",
            "floating",
        ]
        assert cells[-1] == f"`{name}`"
    success = _cells(rows[0])
    assert success[COLUMNS.index("Value")] == "72.0%"
    assert success[COLUMNS.index("95% CI")] == "58.3% to 82.5%"
    assert success[COLUMNS.index("Trials")] == "50"
    completion = _cells(rows[-1])
    assert completion[COLUMNS.index("Value")].endswith(" s")
    assert completion[COLUMNS.index("95% CI")] == "n/a"
    assert completion[COLUMNS.index("Trials")] == "36 of 50"


def test_no_completion_row_without_successful_episodes():
    doc = copy.deepcopy(as_real_s4(fake_doc("unseen")))
    doc["aggregates"]["continuous"]["completion_time_s"] = {
        "n": 0,
        "mean": None,
        "std": None,
        "median": None,
    }
    metrics = [_cells(r)[COLUMNS.index("Metric")] for r in rows_for(doc, "x.json")]
    assert "completion_time_s (median)" not in metrics and len(metrics) == 5


def test_commit_cell_marks_a_dirty_tree_and_a_missing_commit():
    dirty = _cells(rows_for(as_real_s4(fake_doc(), dirty=True), "x.json")[0])[1]
    assert dirty == "abababa+dirty"
    doc = as_real_s4(fake_doc())
    doc["run"]["git_commit"], doc["run"]["git_dirty"] = "unknown", None
    assert _cells(rows_for(doc, "x.json")[0])[1] == "unknown"


# ----------------------------------------------------------------------------- the table


def test_add_documents_appends_inside_the_table_and_touches_nothing_else():
    text = DOC.read_text(encoding="utf-8")
    doc = as_real_s4(fake_doc("unseen"))
    new, added, skipped = add_documents(text, [("a_S4_unseen.json", doc)])
    assert added == ["a_S4_unseen.json"] and skipped == []
    old_lines, new_lines = text.split("\n"), new.split("\n")
    first, end = locate_table(old_lines)
    count = len(rows_for(doc, "a_S4_unseen.json"))
    assert new_lines[:end] == old_lines[:end]
    assert new_lines[end : end + count] == rows_for(doc, "a_S4_unseen.json")
    assert new_lines[end + count :] == old_lines[end:]
    assert listed_names(new_lines) == {"a_S4_unseen.json"}


def test_add_documents_is_idempotent_and_keyed_by_file_name():
    text = DOC.read_text(encoding="utf-8")
    real = as_real_s4(fake_doc("unseen"))
    once, _, _ = add_documents(text, [("a_S4_unseen.json", real)])
    twice, added, skipped = add_documents(once, [("a_S4_unseen.json", real)])
    assert twice == once and added == [] and skipped == ["a_S4_unseen.json"]
    # A different file name is a different result, even with identical content.
    thrice, added, _ = add_documents(twice, [("b_S4_unseen.json", real)])
    assert added == ["b_S4_unseen.json"] and thrice.count("b_S4_unseen.json") == 6
    # The same name twice in one call is added once.
    both, added, skipped = add_documents(text, [("c.json", real), ("c.json", real)])
    assert added == ["c.json"] and skipped == ["c.json"]


def test_add_documents_keeps_crlf_line_endings():
    text = DOC.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\n", "\r\n")
    new, added, _ = add_documents(text, [("a.json", as_real_s4(fake_doc()))])
    assert added == ["a.json"]
    assert "\n" not in new.replace("\r\n", "")


def test_table_errors():
    with pytest.raises(TableError, match="section"):
        add_documents("# nothing here\n", [])
    with pytest.raises(TableError, match="no table"):
        add_documents("## Evaluation results\n\ntext only\n", [])
    wrong = DOC.read_text(encoding="utf-8").replace("| Trials |", "| Count |")
    with pytest.raises(TableError, match="header"):
        add_documents(wrong, [])


# ----------------------------------------------------------------------------- the script


@pytest.fixture
def script():
    return load_script("update_experiments")


@pytest.fixture
def doc_copy(tmp_path):
    path = tmp_path / "experiments.md"
    path.write_bytes(DOC.read_bytes())
    return path


def _rows(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    first, end = locate_table(lines)
    return lines[first:end]


def test_script_adds_rows_once(script, tmp_path, doc_copy, capsys):
    results = tmp_path / "eval"
    write_doc(results, "20261023T100000Z_abababa_S4_unseen.json", as_real_s4(fake_doc("unseen")))
    agile = as_real_s4(fake_doc("unseen", "agile"))
    write_doc(results, "20261023T110000Z_abababa_S4_unseen.json", agile)
    before = DOC.read_bytes()

    assert script.main([str(results), "--doc", str(doc_copy)]) == 0
    rows = _rows(doc_copy)
    assert len(rows) == 12
    assert {_cells(r)[4] for r in rows} == {"own", "agile"}
    assert "2 document(s) added" in capsys.readouterr().out

    snapshot = doc_copy.read_bytes()
    assert script.main([str(results), "--doc", str(doc_copy)]) == 0
    assert doc_copy.read_bytes() == snapshot
    out = capsys.readouterr().out
    assert "already listed" in out and "0 document(s) added to" in out and ", 2 skipped" in out
    assert DOC.read_bytes() == before  # the committed document is never touched


def test_script_refuses_synthetic_results(script, tmp_path, doc_copy, capsys):
    results = tmp_path / "eval"
    fake_file = write_doc(results, "20261023T100000Z_abababa_fake_seen.json", fake_doc())
    snapshot = doc_copy.read_bytes()

    # Found by scanning a directory: skipped with a warning.
    assert script.main([str(results), "--doc", str(doc_copy)]) == 0
    assert "refused synthetic" in capsys.readouterr().err
    assert doc_copy.read_bytes() == snapshot

    # Named explicitly: refused with a failing status.
    assert script.main([str(fake_file), "--doc", str(doc_copy)]) == 1
    assert doc_copy.read_bytes() == snapshot
    capsys.readouterr()

    # Only with the flag are the rows added (to a scratch copy).
    assert script.main([str(fake_file), "--doc", str(doc_copy), "--allow-synthetic"]) == 0
    assert len(_rows(doc_copy)) == 6
    assert all(_cells(r)[2] == "fake" for r in _rows(doc_copy))


def test_script_mixed_directory_adds_only_the_real_result(script, tmp_path, doc_copy):
    results = tmp_path / "eval"
    write_doc(results, "a_fake_seen.json", fake_doc())
    write_doc(results, "b_S4_unseen.json", as_real_s4(fake_doc("unseen")))
    assert script.main([str(results), "--doc", str(doc_copy)]) == 0
    assert {_cells(r)[-1] for r in _rows(doc_copy)} == {"`b_S4_unseen.json`"}


@pytest.mark.parametrize("problem", ["invalid json", "schema violation", "missing doc", "no files"])
def test_script_errors_exit_2_and_write_nothing(script, tmp_path, doc_copy, capsys, problem):
    results = tmp_path / "eval"
    results.mkdir()
    doc_arg = doc_copy
    if problem == "invalid json":
        (results / "bad.json").write_text("{not json", encoding="utf-8")
    elif problem == "schema violation":
        bad = as_real_s4(fake_doc())
        bad["run"]["policy"] = "random"
        write_doc(results, "bad.json", bad)
    elif problem == "missing doc":
        write_doc(results, "ok.json", as_real_s4(fake_doc()))
        doc_arg = tmp_path / "missing.md"
    snapshot = doc_copy.read_bytes()
    assert script.main([str(results), "--doc", str(doc_arg)]) == 2
    assert doc_copy.read_bytes() == snapshot
    assert "error:" in capsys.readouterr().err
