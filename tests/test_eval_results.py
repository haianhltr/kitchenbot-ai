"""Reading result files: ordering, explicit versus scanned, validation errors."""

import pytest
from eval_helpers import as_real_s4, fake_doc, write_doc

from kitchenbot.evaluation.results import (
    ResultError,
    find_result_files,
    load_result,
    load_results,
)


def test_directory_scan_is_chronological_with_rerun_suffixes_after_the_base_name(tmp_path):
    for name in ("b_S4_seen.json", "a_S4_seen-10.json", "a_S4_seen-2.json", "a_S4_seen.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    found = find_result_files([tmp_path])
    assert [p.name for p, _ in found] == [
        "a_S4_seen.json",
        "a_S4_seen-2.json",
        "a_S4_seen-10.json",
        "b_S4_seen.json",
    ]
    assert not any(explicit for _, explicit in found)


def test_named_files_are_explicit_and_listed_once(tmp_path):
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    found = find_result_files([tmp_path / "a.json", tmp_path, tmp_path / "a.json"])
    assert found == [(tmp_path / "a.json", True)]


def test_missing_path_is_an_error(tmp_path):
    with pytest.raises(ResultError, match="no such file"):
        find_result_files([tmp_path / "missing"])


def test_load_result_validates(tmp_path):
    good = write_doc(tmp_path, "good.json", as_real_s4(fake_doc()))
    assert load_result(good).synthetic is False
    for name, text in (("garbage.json", "{nope"), ("list.json", "[]"), ("empty.json", "{}")):
        (tmp_path / name).write_text(text, encoding="utf-8")
        with pytest.raises(ResultError, match=name):
            load_result(tmp_path / name)
    inconsistent = as_real_s4(fake_doc())
    inconsistent["aggregates"]["success"]["count"] = 0
    inconsistent["aggregates"]["success"]["rate"] = 0.0
    path = write_doc(tmp_path, "inconsistent.json", inconsistent)
    with pytest.raises(ResultError, match="inconsistent"):
        load_result(path)


def test_load_results_splits_synthetic_from_the_rest(tmp_path):
    write_doc(tmp_path, "a_fake_seen.json", fake_doc())
    write_doc(tmp_path, "b_S4_seen.json", as_real_s4(fake_doc()))
    accepted, refused = load_results([tmp_path])
    assert [r.name for r in accepted] == ["b_S4_seen.json"]
    assert [r.name for r in refused] == ["a_fake_seen.json"]
    accepted, refused = load_results([tmp_path], allow_synthetic=True)
    assert [r.name for r in accepted] == ["a_fake_seen.json", "b_S4_seen.json"] and not refused
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    with pytest.raises(ResultError, match="no result"):
        load_results([empty])
