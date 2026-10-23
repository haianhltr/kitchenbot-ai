"""Figures: run selection, labels, the three plots, the synthetic marking, deterministic output."""

import copy
from pathlib import Path

import pytest
from eval_helpers import as_real_s4, fake_doc, load_script, write_doc
from matplotlib.figure import Figure

from kitchenbot.evaluation import plotting
from kitchenbot.evaluation.plotting import (
    FAILURE_FILE,
    SUCCESS_FILE,
    TORSO_FILE,
    Run,
    plot_failure_tags,
    plot_success_rates,
    plot_torso_error,
    render_all,
    select_runs,
    torso_title,
    variant_colors,
    variant_labels,
)
from kitchenbot.evaluation.results import ResultFile

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _result(name, doc):
    return ResultFile(path=Path(name), doc=doc, explicit=True)


def _runs(*docs):
    return [Run(f"run{i}.json", d) for i, d in enumerate(docs)]


def _suptitle(fig):
    return fig.get_suptitle()


# ----------------------------------------------------------------------------- selection and labels


def test_select_runs_keeps_the_newest_per_configuration():
    old = fake_doc("unseen", started_at="2026-10-23T08:00:00Z")
    new = fake_doc("unseen", started_at="2026-10-23T09:00:00Z")
    other = fake_doc("unseen", "agile", started_at="2026-10-23T07:00:00Z")
    runs, superseded = select_runs(
        [_result("a.json", new), _result("b.json", old), _result("c.json", other)]
    )
    assert [r.name for r in runs] == ["c.json", "a.json"]
    assert {r.variant[0] for r in runs} == {"own", "agile"}
    assert superseded == ["b.json"]
    # Order of arrival does not matter.
    runs2, superseded2 = select_runs(
        [_result("b.json", old), _result("c.json", other), _result("a.json", new)]
    )
    assert [r.name for r in runs2] == [r.name for r in runs] and superseded2 == ["b.json"]


def test_runs_are_ordered_by_scenario_then_split_then_variant():
    docs = [
        as_real_s4(fake_doc("unseen", "own")),
        fake_doc("seen", "agile"),
        fake_doc("unseen", "agile"),
        fake_doc("seen", "own"),
    ]
    runs, _ = select_runs([_result(f"{i}.json", d) for i, d in enumerate(docs)])
    assert [(r.group, r.variant[0]) for r in runs] == [
        (("S4", "unseen"), "own"),
        (("fake", "seen"), "agile"),
        (("fake", "seen"), "own"),
        (("fake", "unseen"), "agile"),
    ]


def test_labels_name_only_the_fields_that_differ():
    runs = _runs(fake_doc(policy="own"), fake_doc(policy="agile"))
    assert variant_labels(runs) == {
        ("agile", "dex3", "floating"): "policy agile",
        ("own", "dex3", "floating"): "policy own",
    }
    single = _runs(fake_doc())
    assert variant_labels(single) == {
        ("own", "dex3", "floating"): "policy own, hand dex3, base floating"
    }
    colors = variant_colors(runs)
    assert len(set(colors.values())) == 2 and colors == variant_colors(list(reversed(runs)))


# ----------------------------------------------------------------------------- the figures


def test_success_rate_figure_has_one_bar_per_run_side_by_side():
    runs = _runs(
        fake_doc("seen", "own"),
        fake_doc("seen", "agile"),
        fake_doc("unseen", "own"),
        fake_doc("unseen", "agile"),
    )
    fig = plot_success_rates(runs)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    bars = ax.patches
    assert len(bars) == 4
    heights = sorted(round(b.get_height(), 1) for b in bars)
    assert heights == [72.0, 72.0, 78.0, 78.0]
    # Two baselines of one group sit next to each other, not on top of each other.
    xs = sorted(b.get_x() for b in bars)
    assert xs[1] - xs[0] < 1.0 and xs[2] - xs[1] > 0
    assert [t.get_text() for t in ax.get_xticklabels()] == ["fake\nseen", "fake\nunseen"]
    assert "Wilson" in _suptitle(fig)
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend == ["policy agile", "policy own"]


def test_success_rate_figure_with_a_missing_combination():
    runs = _runs(fake_doc("seen", "own"), fake_doc("seen", "agile"), fake_doc("unseen", "own"))
    assert len(plot_success_rates(runs).axes[0].patches) == 3


def test_failure_tag_figure_has_a_panel_per_group():
    runs = _runs(
        fake_doc("seen", "own"),
        fake_doc("seen", "agile"),
        fake_doc("unseen", "own"),
        as_real_s4(fake_doc("unseen", "own")),
    )
    fig = plot_failure_tags(runs)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3
    assert [ax.get_title() for ax in visible] == ["S4 unseen", "fake seen", "fake unseen"]
    labels = [t.get_text() for t in visible[0].get_xticklabels()]
    assert labels == [
        "fall",
        "ik_infeasible",
        "grasp_miss",
        "drop",
        "rack_collision",
        "misplace",
        "timeout",
    ]
    counts = [b.get_height() for b in visible[1].patches]
    assert len(counts) == 14 and sum(counts) == 2 * 22  # two identical runs, 22 failures of 100
    assert len([ax for ax in fig.axes if not ax.get_visible()]) == 0


def test_failure_tag_figure_says_when_there_are_no_failures():
    doc = copy.deepcopy(fake_doc("seen"))
    for tag in doc["aggregates"]["failure_tag_counts"]:
        doc["aggregates"]["failure_tag_counts"][tag] = 0
    fig = plot_failure_tags(_runs(doc))
    assert any(t.get_text() == "no failures" for t in fig.axes[0].texts)


def test_torso_figure_uses_traces_when_present():
    runs = _runs(fake_doc("unseen", "own"), fake_doc("unseen", "agile"))
    fig = plot_torso_error(runs)
    assert "per-step traces" in _suptitle(fig) and "no per-step" not in _suptitle(fig)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "time (s)" and ax.get_title() == "fake unseen: traces"
    # 50 faint episode lines plus the median line, for each of the two variants.
    assert len(ax.lines) == 2 * (50 + 1)


def test_torso_figure_falls_back_to_the_distribution_and_says_so():
    runs = _runs(fake_doc("unseen", "own", traces=False), fake_doc("unseen", "agile", traces=False))
    fig = plot_torso_error(runs)
    title = _suptitle(fig)
    assert "per-episode RMS distribution" in title and "no per-step traces" in title.replace(
        "\n", " "
    )
    assert fig.axes[0].get_title() == "fake unseen: per-episode RMS"
    assert fig.axes[0].get_ylabel() == "per-episode RMS torso error (deg)"


def test_torso_figure_mixed_groups_and_partial_traces():
    runs = _runs(
        fake_doc("seen", "own"),
        fake_doc("unseen", "own", traces=False),
    )
    fig = plot_torso_error(runs)
    assert "where traces exist" in _suptitle(fig).replace("\n", " ")
    assert [ax.get_title() for ax in fig.axes] == [
        "fake seen: traces",
        "fake unseen: per-episode RMS",
    ]
    # Within one group a run without traces turns the whole panel into the distribution view.
    both = _runs(fake_doc("seen", "own"), fake_doc("seen", "agile", traces=False))
    assert plot_torso_error(both).axes[0].get_title() == "fake seen: per-episode RMS"


def test_torso_figure_is_none_without_runs_that_report_it():
    assert plot_torso_error([]) is None
    assert torso_title({("S4", "seen"): "traces"}).startswith("Torso orientation error over time")


def test_traces_with_another_sample_rate_are_ignored():
    doc = copy.deepcopy(fake_doc("unseen"))
    doc["episodes"][3]["traces"]["sample_rate_hz"] = 20.0
    rate, series = plotting._trace_arrays(Run("x.json", doc))
    assert rate == 10.0 and len(series) == 49


def test_synthetic_runs_are_marked_in_the_figure_title_and_real_ones_are_not():
    fake_runs = _runs(fake_doc("unseen"))
    real_runs = _runs(as_real_s4(fake_doc("unseen")))
    for plot in (plot_success_rates, plot_failure_tags, plot_torso_error):
        assert plotting.SYNTHETIC_NOTE in _suptitle(plot(fake_runs))
        assert plotting.SYNTHETIC_NOTE not in _suptitle(plot(real_runs))


# ----------------------------------------------------------------------------- files


def test_render_all_writes_deterministic_files(tmp_path):
    runs = _runs(fake_doc("unseen", "own"), fake_doc("unseen", "agile"))
    first = render_all(runs, tmp_path / "a")
    assert [p.name for p in first] == [SUCCESS_FILE, FAILURE_FILE, TORSO_FILE]
    assert (SUCCESS_FILE, FAILURE_FILE, TORSO_FILE) == (
        "success_rates.png",
        "failure_tags.png",
        "torso_error.png",
    )
    for path in first:
        data = path.read_bytes()
        assert data.startswith(PNG_MAGIC) and 5_000 < len(data) < 1_000_000
    second = render_all(runs, tmp_path / "b")
    assert [p.read_bytes() for p in first] == [p.read_bytes() for p in second]


# ----------------------------------------------------------------------------- the script


@pytest.fixture
def script():
    return load_script("plot_results")


def test_script_plots_a_directory(script, tmp_path, capsys):
    results = tmp_path / "eval"

    def real(policy, started_at):
        return as_real_s4(fake_doc("unseen", policy, started_at=started_at))

    write_doc(
        results, "20261023T100000Z_abababa_S4_unseen.json", real("own", "2026-10-23T10:00:00Z")
    )
    write_doc(
        results, "20261023T110000Z_abababa_S4_unseen.json", real("agile", "2026-10-23T11:00:00Z")
    )
    write_doc(
        results, "20261023T090000Z_abababa_S4_unseen.json", real("own", "2026-10-23T09:00:00Z")
    )
    out = tmp_path / "figs"
    assert script.main([str(results), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.glob("*.png")) == sorted(
        [SUCCESS_FILE, FAILURE_FILE, TORSO_FILE]
    )
    text = capsys.readouterr().out
    assert "superseded" in text and "20261023T090000Z" in text


def test_script_refuses_synthetic_unless_allowed(script, tmp_path, capsys):
    results = tmp_path / "eval"
    fake_file = write_doc(results, "20261023T100000Z_abababa_fake_seen.json", fake_doc())
    out = tmp_path / "figs"
    assert script.main([str(results), "--out", str(out)]) == 2  # a scan that finds nothing usable
    assert script.main([str(fake_file), "--out", str(out)]) == 1  # a named file is refused
    assert not out.exists()
    assert "refused synthetic" in capsys.readouterr().err
    assert script.main([str(fake_file), "--out", str(out), "--allow-synthetic"]) == 0
    assert len(list(out.glob("*.png"))) == 3


def test_script_errors(script, tmp_path, capsys):
    assert script.main([str(tmp_path / "missing"), "--out", str(tmp_path / "f")]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert script.main([str(empty), "--out", str(tmp_path / "f")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    assert script.main([str(bad), "--out", str(tmp_path / "f")]) == 2
    assert capsys.readouterr().err.count("error:") == 3
