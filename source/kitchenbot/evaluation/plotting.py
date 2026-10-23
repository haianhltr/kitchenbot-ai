"""Figures from result JSONs (docs/evaluation.md section 4, step 5). matplotlib only.

Three figures, one shared style, deterministic file names:

- ``success_rates.png``: success rate with its 95 percent Wilson interval for every run; runs of
  one scenario and split form a group and the variants inside it (policy, hand, base) sit side
  by side, so two baselines are directly comparable.
- ``failure_tags.png``: for each scenario and split, how many failed trials carry each failure
  tag of section 7, variants side by side.
- ``torso_error.png``: torso orientation error over time when the documents carry per-step
  ``traces``; otherwise the per-episode RMS distribution, and the figure title says so.

Figures are built with ``matplotlib.figure.Figure`` and never touch pyplot or the global backend,
so they render the same on a headless machine and do not interfere with a caller's own figures.
A figure drawn from synthetic data (``run.synthetic``) says so on the figure.

When several documents have the same scenario, split and variant, the newest one (by
``run.started_at``, then file name) is drawn and the others are reported as superseded.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from kitchenbot.evaluation.failure_tags import FAILURE_TAG_VALUES
from kitchenbot.evaluation.metrics import SCENARIO_NAMES
from kitchenbot.evaluation.results import ResultFile

SUCCESS_FILE = "success_rates.png"
FAILURE_FILE = "failure_tags.png"
TORSO_FILE = "torso_error.png"

#: Okabe and Ito colour-blind-safe palette; variants take colours in sorted order.
PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)

STYLE = {
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
}

SYNTHETIC_NOTE = "SYNTHETIC DATA (fake or dry scenario): not a measurement"

#: A variant is the part of the configuration that distinguishes baselines.
Variant = tuple[str, str, str]  # (policy, hand, base)
Group = tuple[str, str]  # (scenario, split)


@dataclass(frozen=True)
class Run:
    """One result document selected for plotting."""

    name: str
    doc: dict[str, Any]

    @property
    def group(self) -> Group:
        return (self.doc["run"]["scenario"], self.doc["run"]["split"])

    @property
    def variant(self) -> Variant:
        run = self.doc["run"]
        return (run["policy"], run["hand"], run["base"])

    @property
    def synthetic(self) -> bool:
        return bool(self.doc["run"]["synthetic"])


def _group_order(group: Group) -> tuple[int, str]:
    return (SCENARIO_NAMES.index(group[0]), group[1])


def select_runs(results: Iterable[ResultFile]) -> tuple[list[Run], list[str]]:
    """Newest document per (scenario, split, variant), sorted for stable figures.

    Returns ``(runs, superseded_names)``.
    """
    newest: dict[tuple[Group, Variant], tuple[tuple[str, str], Run]] = {}
    superseded: list[str] = []
    for result in results:
        run = Run(result.name, result.doc)
        key = (run.group, run.variant)
        rank = (result.doc["run"]["started_at"], result.name)
        if key in newest:
            old_rank, old = newest[key]
            if rank > old_rank:
                superseded.append(old.name)
                newest[key] = (rank, run)
            else:
                superseded.append(run.name)
        else:
            newest[key] = (rank, run)
    runs = sorted((r for _, r in newest.values()), key=lambda r: (_group_order(r.group), r.variant))
    return runs, sorted(superseded)


def variant_labels(runs: Sequence[Run]) -> dict[Variant, str]:
    """Legend labels: only the fields that differ between variants, or all three if none does."""
    variants = sorted({r.variant for r in runs})
    names = ("policy", "hand", "base")
    varying = [i for i in range(3) if len({v[i] for v in variants}) > 1] or [0, 1, 2]
    return {v: ", ".join(f"{names[i]} {v[i]}" for i in varying) for v in variants}


def variant_colors(runs: Sequence[Run]) -> dict[Variant, str]:
    variants = sorted({r.variant for r in runs})
    return {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(variants)}


def _groups(runs: Sequence[Run]) -> list[Group]:
    return sorted({r.group for r in runs}, key=_group_order)


def _title(fig: Figure, title: str, runs: Sequence[Run]) -> None:
    """Figure title, wrapped to the figure width; red with a warning when a run is synthetic."""
    title = textwrap.fill(title, int(fig.get_figwidth() * 10))
    if any(r.synthetic for r in runs):
        fig.suptitle(f"{SYNTHETIC_NOTE}\n{title}", color="#B00020")
    else:
        fig.suptitle(title)


# ----------------------------------------------------------------------------- (a) success rates


def plot_success_rates(runs: Sequence[Run]) -> Figure:
    """Grouped bars of the success rate with 95 percent Wilson intervals."""
    groups = _groups(runs)
    labels, colors = variant_labels(runs), variant_colors(runs)
    variants = sorted(labels)
    width = 0.8 / len(variants)
    by_key = {(r.group, r.variant): r for r in runs}
    with mpl.rc_context(STYLE):
        fig = Figure(
            figsize=(max(6.5, 1.6 * len(groups) * len(variants) + 1.5), 4.8), layout="constrained"
        )
        ax = fig.subplots()
        for j, variant in enumerate(variants):
            offset = (j - (len(variants) - 1) / 2) * width
            xs, rates, lows, highs, ns = [], [], [], [], []
            for i, group in enumerate(groups):
                run = by_key.get((group, variant))
                if run is None:
                    continue
                s = run.doc["aggregates"]["success"]
                xs.append(i + offset)
                rates.append(100 * s["rate"])
                lows.append(100 * (s["rate"] - s["ci_low"]))
                highs.append(100 * (s["ci_high"] - s["rate"]))
                ns.append(s["n"])
            ax.bar(xs, rates, width * 0.92, color=colors[variant], label=labels[variant])
            ax.errorbar(xs, rates, yerr=[lows, highs], fmt="none", ecolor="black", capsize=3)
            for x, rate, high, n in zip(xs, rates, highs, ns, strict=True):
                ax.text(
                    x,
                    rate + high + 1.5,
                    f"{rate:.0f}%\nn={n}",
                    ha="center",
                    va="bottom",
                    size=8,
                )
        ax.set_xticks(range(len(groups)), [f"{s}\n{sp}" for s, sp in groups])
        ax.set_ylim(0, 118)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("success rate (%)")
        _title(fig, "Success rate with 95% Wilson interval", runs)
        ax.grid(axis="x", visible=False)
        fig.legend(loc="outside lower center", ncols=min(len(variants), 3))
    return fig


# ----------------------------------------------------------------------------- (b) failure tags


def plot_failure_tags(runs: Sequence[Run]) -> Figure:
    """One panel per scenario and split: failed trials per failure tag, variants side by side."""
    groups = _groups(runs)
    labels, colors = variant_labels(runs), variant_colors(runs)
    variants = sorted(labels)
    by_key = {(r.group, r.variant): r for r in runs}
    ncols = min(3, len(groups))
    nrows = math.ceil(len(groups) / ncols)
    width = 0.8 / len(variants)
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(max(6.5, 4.6 * ncols), 3.6 * nrows + 1.2), layout="constrained")
        axes = fig.subplots(nrows, ncols, squeeze=False)
        for ax in axes.flat[len(groups) :]:
            ax.set_visible(False)
        for ax, group in zip(axes.flat, groups, strict=False):
            total_failures = 0
            for j, variant in enumerate(variants):
                run = by_key.get((group, variant))
                if run is None:
                    continue
                counts = run.doc["aggregates"]["failure_tag_counts"]
                values = [counts[tag] for tag in FAILURE_TAG_VALUES]
                total_failures += sum(values)
                offset = (j - (len(variants) - 1) / 2) * width
                xs = [i + offset for i in range(len(values))]
                label = labels[variant] if ax is axes.flat[0] else None
                ax.bar(xs, values, width * 0.92, color=colors[variant], label=label)
            ax.set_xticks(
                range(len(FAILURE_TAG_VALUES)), FAILURE_TAG_VALUES, rotation=35, ha="right"
            )
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylabel("failed trials")
            ax.set_title(f"{group[0]} {group[1]}")
            ax.grid(axis="x", visible=False)
            if total_failures == 0:
                ax.text(0.5, 0.5, "no failures", transform=ax.transAxes, ha="center", va="center")
        _title(fig, "Primary failure tag of failed trials (docs/evaluation.md section 7)", runs)
        fig.legend(loc="outside lower center", ncols=min(len(variants), 3))
    return fig


# ----------------------------------------------------------------------------- (c) torso error


def _trace_arrays(run: Run) -> tuple[float, list[np.ndarray]] | None:
    """``(sample_rate_hz, series)`` of the run's torso traces, or ``None`` if it has none.

    Episodes whose sample rate differs from the first episode's are left out (a backend uses one
    rate per run, so this only guards against a malformed file).
    """
    rate = None
    series: list[np.ndarray] = []
    for episode in run.doc["episodes"]:
        trace = episode.get("traces")
        if trace is None:
            continue
        if rate is None:
            rate = float(trace["sample_rate_hz"])
        if float(trace["sample_rate_hz"]) == rate:
            series.append(np.asarray(trace["torso_orientation_error_deg"], dtype=float))
    if rate is None:
        return None
    return rate, series


def torso_runs(runs: Sequence[Run]) -> list[Run]:
    """Runs whose scenario reports the torso orientation error."""
    return [r for r in runs if "torso_orientation_error_deg" in r.doc["episodes"][0]["metrics"]]


def torso_mode(runs: Sequence[Run]) -> dict[Group, str]:
    """``"traces"`` for a group whose every run has traces, else ``"distribution"``."""
    modes: dict[Group, str] = {}
    for group in _groups(runs):
        members = [r for r in runs if r.group == group]
        modes[group] = (
            "traces" if all(_trace_arrays(r) is not None for r in members) else "distribution"
        )
    return modes


def torso_title(modes: dict[Group, str]) -> str:
    kinds = set(modes.values())
    if kinds == {"traces"}:
        return "Torso orientation error over time (per-step traces, median and quartiles)"
    if kinds == {"distribution"}:
        return (
            "Torso orientation error: per-episode RMS distribution "
            "(these files have no per-step traces)"
        )
    return "Torso orientation error: over time where traces exist, per-episode RMS otherwise"


def _draw_traces(ax: Any, run: Run, color: str, label: str | None) -> None:
    rate, series = _trace_arrays(run)  # type: ignore[misc]
    length = max(len(s) for s in series)
    padded = np.full((len(series), length), np.nan)
    for i, s in enumerate(series):
        padded[i, : len(s)] = s
    t = np.arange(length) / rate
    for s in series:
        ax.plot(np.arange(len(s)) / rate, s, color=color, alpha=0.10, linewidth=0.6)
    q1, med, q3 = (np.nanpercentile(padded, q, axis=0) for q in (25, 50, 75))
    ax.fill_between(t, q1, q3, color=color, alpha=0.25, linewidth=0)
    ax.plot(t, med, color=color, label=label)


def _draw_distribution(ax: Any, runs: Sequence[Run], positions: dict[Variant, int], colors) -> None:
    for run in runs:
        values = [e["metrics"]["torso_orientation_error_deg"] for e in run.doc["episodes"]]
        x = positions[run.variant]
        color = colors[run.variant]
        ax.boxplot(
            [values],
            positions=[x],
            widths=0.5,
            showfliers=False,
            medianprops={"color": "black"},
            boxprops={"color": color},
            whiskerprops={"color": color},
            capprops={"color": color},
        )
        # Deterministic jitter so the figure is the same every time.
        jitter = [((i * 0.6180339887) % 1.0 - 0.5) * 0.3 for i in range(len(values))]
        ax.scatter([x + j for j in jitter], values, s=8, color=color, alpha=0.5, zorder=3)


def plot_torso_error(runs: Sequence[Run]) -> Figure | None:
    """Torso orientation error per scenario and split, or ``None`` if no run reports it."""
    runs = torso_runs(runs)
    if not runs:
        return None
    groups = _groups(runs)
    labels, colors = variant_labels(runs), variant_colors(runs)
    modes = torso_mode(runs)
    ncols = min(3, len(groups))
    nrows = math.ceil(len(groups) / ncols)
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(max(6.5, 5.0 * ncols), 3.8 * nrows + 1.3), layout="constrained")
        axes = fig.subplots(nrows, ncols, squeeze=False)
        for ax in axes.flat[len(groups) :]:
            ax.set_visible(False)
        for ax, group in zip(axes.flat, groups, strict=False):
            members = [r for r in runs if r.group == group]
            if modes[group] == "traces":
                for run in members:
                    label = labels[run.variant] if ax is axes.flat[0] else None
                    _draw_traces(ax, run, colors[run.variant], label)
                ax.set_xlabel("time (s)")
                ax.set_ylabel("torso tilt error (deg)")
                ax.set_title(f"{group[0]} {group[1]}: traces")
            else:
                positions = {v: i for i, v in enumerate(sorted({r.variant for r in members}))}
                _draw_distribution(ax, members, positions, colors)
                ax.set_xticks(
                    list(positions.values()),
                    [labels[v] for v in positions],
                    rotation=20,
                    ha="right",
                )
                ax.set_ylabel("per-episode RMS torso error (deg)")
                ax.set_title(f"{group[0]} {group[1]}: per-episode RMS")
                ax.grid(axis="x", visible=False)
            ax.set_ylim(bottom=0)
        _title(fig, torso_title(modes), runs)
        if "traces" in modes.values():
            fig.legend(loc="outside lower center", ncols=min(len(labels), 3))
    return fig


# ----------------------------------------------------------------------------- driver


def save(fig: Figure, path: Path) -> Path:
    """Save as PNG, creating the directory. Drops the matplotlib version from the metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", metadata={"Software": None})
    return path


def render_all(runs: Sequence[Run], out_dir: Path) -> list[Path]:
    """Write every figure that has data into ``out_dir`` and return the paths written."""
    out_dir = Path(out_dir)
    written = [
        save(plot_success_rates(runs), out_dir / SUCCESS_FILE),
        save(plot_failure_tags(runs), out_dir / FAILURE_FILE),
    ]
    torso = plot_torso_error(runs)
    if torso is not None:
        written.append(save(torso, out_dir / TORSO_FILE))
    return written
