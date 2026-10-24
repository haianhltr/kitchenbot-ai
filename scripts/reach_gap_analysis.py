r"""Reach-gap analysis (#56): the planner's palm targets of the evaluation splits against the
right-palm envelopes, waist at zero and waist sampled. numpy and matplotlib only, no simulator.

Run (about a minute; the numbers of ``docs/reach_gap_analysis.md``)::

    py -3.12 scripts/reach_gap_analysis.py --figure docs/figures/reach_gap.png

It writes under ``--out-dir`` (default ``outputs/reach_gap``, git-ignored): ``waypoints.csv``
(reachable fraction per envelope, stand-off, plate box centre ``y``, scenario, split and
waypoint), ``episodes.csv`` (the same for whole target sets), ``plate_overhang.csv``,
``rack_sweep.csv``, ``sensitivity.csv``, ``tables.md`` (the markdown tables of the doc) and the
figure (default ``<out-dir>/reach_gap.png``).

Method
------
For each of the ten (scenario, split) pairs every committed seed is drawn with
``scene.randomization.sample_split`` (``--max-seeds`` keeps the first N of each split). Each
episode becomes the three poses ``planning.plate_to_rack.plan_waypoints`` reads, in the kitchen
frame of ``scene.kitchen_dims`` (origin on the floor at the middle of the counter's front edge,
``x`` along the counter, which is the robot's right, ``y`` into the counter, ``z`` up):

- ``plate_pose``, the grasp frame on the rim (``--grasp-point near_rim``, the default): the rim
  point nearest the robot, ``(x, y - D / 2)`` of the drawn plate centre and diameter ``D``, at the
  rim top ``COUNTER_HEIGHT_M + PLATE_HEIGHT_M`` = 0.92 m, with the drawn plate yaw.
  ``--grasp-point centre`` uses the plate centre instead (#52's "0.61 to 0.84 m ahead").
- ``slot_pose``: the drawn slot (``EpisodeSpec.rack_slot_pose``, rack frame) placed by
  ``--rack-pose x y z yaw_deg``, the kitchen pose of the rack frame (centre of the underside of
  the rack base). The grasp frame at the slot entry sits ``D`` above the entry: the palm holds the
  top of the vertical plate whose lower edge is at the top of the tines (planner docstring: the
  diameter is folded in by the environment layer). **Provisional**: #59 adds
  ``DISHWASHER_POSE_KITCHEN``; until then the default :data:`PROVISIONAL_RACK_POSE` below is
  derived from docs/architecture.md section 10, and every slot number is labelled provisional.
- ``counter_edge_pose``: the counter's front edge (``y = 0``, ``z = COUNTER_HEIGHT_M``) at the
  grasp point's ``x``, the shortest way off the counter.

``plan_waypoints`` (``PlannerConfig()`` defaults: pre-grasp +8 cm, lift +15 cm, counter-edge
clearance 15 cm, above the slot +25 cm, insertion -5 cm) turns them into seven palm targets:
``pre_grasp``, ``grasp``, ``lift``, ``counter_edge``, ``above_slot``, ``aligned`` (the position of
``above_slot``, a new orientation) and ``insertion``. Each is expressed in the pelvis frame of the
episode: the pelvis stands at ``(dx, -standoff + dy, h)`` with yaw ``90 deg + dyaw`` (facing
``+y``), ``(dx, dy, dyaw)`` the spec's pelvis noise and ``h`` ``--pelvis-height``, default
:data:`PELVIS_HEIGHT_M`. A target counts as reachable if the envelope ``contains`` it, and with
margin if ``margin_contains(p, 0.02)``.

The point: the planner's palm target is the grasp frame itself (``palm_pose`` "coincides with the
grasp frame" at the grasp), and ``ArmTargetPipeline`` compares it directly with the envelope's
point. #56 ran with envelopes of the ``right_hand_palm_link`` origin and measured the finger-base
point (:data:`FINGER_BASE_OFFSET_M` in the palm link frame, D012's IK ``body_offset``) in the
sensitivity table, with an envelope built in memory with the same seeds. Since #66 the committed
envelopes are of the finger-base point, so the main tables are the finger-base point and the
sensitivity row re-samples the palm link origin instead (``"palm-link point"``); the tables of
``docs/reach_gap_analysis.md`` are #56's and were not regenerated.

Target sets per scenario (:data:`SCENARIO_TARGETS`): S1 has no reach-target distribution of its
own yet (docs/evaluation.md: "a reach target appears in the front workspace"), so its stand-in is
its plate draw's ``pre_grasp`` and ``grasp``; S2 adds ``lift``; S3 (plate already in hand) is the
three slot targets; S4 is all seven; S0 has no arm target. An episode is reachable if every target
of its set is.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from kitchenbot.controllers.reachability import (
    DEFAULT_PATH,
    FINGER_BASE_OFFSET_M,
    WAIST_PATH,
    ReachabilityEnvelope,
    build_envelope,
)
from kitchenbot.evaluation.seeds import SPLITS
from kitchenbot.geometry import Pose, quat_apply, quat_from_yaw, quat_to_matrix
from kitchenbot.planning.plate_to_rack import PlannerConfig, plan_waypoints
from kitchenbot.robots.kinematics import right_arm_chain
from kitchenbot.scene import kitchen_dims as kd
from kitchenbot.scene.randomization import (
    PLATE_NOMINAL_XY_M,
    SCENARIOS,
    EpisodeSpec,
    RandomizationRanges,
    sample_split,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_JOINTS = REPO / "tests" / "fixtures" / "g1_dex3_joints_urdf.json"
DEFAULT_OUT_DIR = REPO / "outputs" / "reach_gap"

#: Pelvis height above the floor, m: ``G1_29DOF_CFG.init_state.pos`` z (Isaac Lab v3.0.0
#: ``unitree.py:437``), which ``robots/g1_dex3.py`` (#46) inherits; also the stand env's
#: ``base_height_l2`` target. The settled standing height is #5's to measure.
PELVIS_HEIGHT_M = 0.75

#: Provisional kitchen pose ``(x, y, z, yaw_deg)`` of the rack frame (centre of the underside of
#: the lower-rack base) until #59's ``DISHWASHER_POSE_KITCHEN``. From docs/architecture.md
#: section 10 and its sources: ArtVIP ``dishwasher_2`` (D011) on the floor under the counter, on
#: the right where ``kitchen_dims`` leaves room for it (the plate box ends at x = 0.25 m, the
#: counter at 0.90 m; the 0.48 m wide unit centred at 0.55 m), its front flush with the counter's
#: front edge, door open; the lower rack, centred about 0.20 m behind the front in the 0.395 m deep
#: body, pulled out by its full 0.2 m travel to y = 0; its base at the inspected z = 0.088 m (#29,
#: scale 1.0, rounded to 0.09); rack x along the counter, so yaw 0.
PROVISIONAL_RACK_POSE: tuple[float, float, float, float] = (0.55, 0.0, 0.09, 0.0)

#: ``plan_waypoints`` keys, in execution order.
WAYPOINTS: tuple[str, ...] = (
    "pre_grasp",
    "grasp",
    "lift",
    "counter_edge",
    "above_slot",
    "aligned",
    "insertion",
)
PLATE_TARGETS: tuple[str, ...] = WAYPOINTS[:4]
SLOT_TARGETS: tuple[str, ...] = WAYPOINTS[4:]
#: The targets each scenario's episode needs (see the module docstring).
SCENARIO_TARGETS: dict[str, tuple[str, ...]] = {
    "S0": (),
    "S1": ("pre_grasp", "grasp"),
    "S2": ("pre_grasp", "grasp", "lift"),
    "S3": SLOT_TARGETS,
    "S4": WAYPOINTS,
}
#: Target sets reported per episode: the scenario's own, and S4's two halves.
PARTS: tuple[str, ...] = ("task", "plate", "slot")

PAIRS: tuple[tuple[str, str], ...] = tuple((s, sp) for s in SCENARIOS for sp in SPLITS)
STANDOFFS_M: tuple[float, ...] = (0.25, 0.30, 0.35, 0.40, 0.45)
PLATE_YS_M: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30)
MARGIN_M = 0.02
RACK_ZS_M: tuple[float, ...] = (0.09, 0.30, 0.50, 0.70)
RACK_XS_M: tuple[float, ...] = (0.25, 0.40, 0.55)
#: The envelopes of the comparison, by name: before (#48) and after (#56).
ENVELOPE_NAMES: tuple[str, ...] = ("waist0", "waist")
#: Smaller waist ranges for the sensitivity table (yaw, roll, pitch; deg), each re-sampled in
#: memory with the waist envelope's seed and sample count: how much of the lean is needed.
WAIST_VARIANTS_DEG: dict[str, tuple[tuple[float, float], ...]] = {
    "waist pitch 0 to 10, yaw +-20": ((-20.0, 20.0), (0.0, 0.0), (0.0, 10.0)),
    "waist pitch 0 to 20, yaw +-20": ((-20.0, 20.0), (0.0, 0.0), (0.0, 20.0)),
    "waist pitch 0 to 29.8, yaw 0": ((0.0, 0.0), (0.0, 0.0), (0.0, 30.0)),
    # one commanded posture instead of a range: what a planner-set lean would give
    "waist held at pitch +20, yaw 0": ((0.0, 0.0), (0.0, 0.0), (20.0, 20.0)),
    "waist held at pitch +29.8, yaw 0": ((0.0, 0.0), (0.0, 0.0), (30.0, 30.0)),
}


@dataclass(frozen=True)
class Setup:
    """Everything that places an episode's targets in the pelvis frame, besides the spec."""

    standoff_m: float = kd.ROBOT_STANDOFF_M
    plate_centre_y_m: float = PLATE_NOMINAL_XY_M[1]
    pelvis_height_m: float = PELVIS_HEIGHT_M
    rack_pose: tuple[float, float, float, float] = PROVISIONAL_RACK_POSE
    grasp_point: str = "near_rim"

    def __post_init__(self) -> None:
        if self.grasp_point not in ("near_rim", "centre"):
            raise ValueError(f"grasp_point must be near_rim or centre, got {self.grasp_point!r}")


# --- episodes and targets ---------------------------------------------------------------------


def ranges_for(split: str, plate_centre_y_m: float) -> RandomizationRanges:
    """The split's ranges with the plate box centre moved to ``plate_centre_y_m``.

    The draws are the same uniforms whatever the centre, so the sweep only shifts the plates.
    """
    base = RandomizationRanges.for_split(split)
    return replace(base, plate_centre_xy_m=(base.plate_centre_xy_m[0], float(plate_centre_y_m)))


def episode_specs(
    scenario: str, split: str, plate_centre_y_m: float, max_seeds: int | None = None
) -> tuple[EpisodeSpec, ...]:
    """Every committed seed of the pair (the first ``max_seeds`` if given)."""
    specs = sample_split(split, scenario, ranges_for(split, plate_centre_y_m))
    return specs if max_seeds is None else specs[:max_seeds]


def pelvis_pose(spec: EpisodeSpec, setup: Setup) -> Pose:
    """Kitchen pose of the pelvis: ``setup.standoff_m`` in front of the edge, facing ``+y``."""
    dx, dy = spec.pelvis_offset_m
    yaw = math.pi / 2.0 + math.radians(spec.pelvis_yaw_offset_deg)
    return Pose((dx, -setup.standoff_m + dy, setup.pelvis_height_m), quat_from_yaw(yaw))


def kitchen_poses(spec: EpisodeSpec, setup: Setup) -> tuple[Pose, Pose, Pose]:
    """``(plate_pose, slot_pose, counter_edge_pose)`` of the planner, kitchen frame."""
    x, y, z_counter = spec.plate_position_m
    diameter = spec.plate_diameter_m
    if setup.grasp_point == "near_rim":
        y = y - diameter / 2.0
    plate_q = quat_from_yaw(math.radians(spec.plate_yaw_deg))
    plate = Pose((x, y, z_counter + kd.PLATE_HEIGHT_M), plate_q)

    rx, ry, rz, ryaw_deg = setup.rack_pose
    rack_q = quat_from_yaw(math.radians(ryaw_deg))
    sx, sy, sz, syaw = spec.rack_slot_pose
    entry = np.array([rx, ry, rz]) + quat_apply(rack_q, (sx, sy, sz))
    slot_q = quat_from_yaw(math.radians(ryaw_deg) + syaw)
    slot = Pose(entry + np.array([0.0, 0.0, diameter]), slot_q)

    edge = Pose((x, 0.0, kd.COUNTER_HEIGHT_M))
    return plate, slot, edge


def pelvis_targets(
    spec: EpisodeSpec, setup: Setup, config: PlannerConfig | None = None
) -> np.ndarray:
    """``(7, 3)``: the ``WAYPOINTS`` positions of the episode in its pelvis frame.

    Same as ``kitchenbot.geometry.to_frame(target, pelvis_pose)`` (a test checks it), written
    with one rotation matrix per episode for speed.
    """
    plate, slot, edge = kitchen_poses(spec, setup)
    waypoints = plan_waypoints(plate, slot, edge, config or PlannerConfig())
    if tuple(waypoints) != WAYPOINTS:
        raise AssertionError(f"plan_waypoints keys changed: {tuple(waypoints)}")
    pelvis = pelvis_pose(spec, setup)
    world = np.array([waypoints[name].position for name in WAYPOINTS])
    return (world - pelvis.position) @ quat_to_matrix(pelvis.quat_wxyz)


def reach_flags(
    envelope: ReachabilityEnvelope, targets: np.ndarray, margin_m: float = MARGIN_M
) -> tuple[np.ndarray, np.ndarray]:
    """``(inside, with_margin)``, booleans of ``targets.shape[:-1]``."""
    pts = np.asarray(targets, dtype=np.float64)
    flat = pts.reshape(-1, 3)
    inside = envelope.contains_points(flat)
    margin = np.zeros_like(inside)
    for i in np.flatnonzero(inside):  # outside means outside with any margin too
        margin[i] = envelope.margin_contains(flat[i], margin_m)
    return inside.reshape(pts.shape[:-1]), margin.reshape(pts.shape[:-1])


def target_set(scenario: str, part: str) -> tuple[str, ...]:
    """The waypoints of ``part`` (``task``, ``plate`` or ``slot``) for ``scenario``."""
    if part == "task":
        return SCENARIO_TARGETS[scenario]
    return {"plate": PLATE_TARGETS, "slot": SLOT_TARGETS}[part]


def episode_fraction(flags: np.ndarray, names: Sequence[str]) -> float | None:
    """Fraction of episodes (rows of ``flags``, ``(N, 7)``) with every target in ``names``."""
    if not names:
        return None
    cols = [WAYPOINTS.index(n) for n in names]
    return float(flags[:, cols].all(axis=1).mean())


# --- the sweep --------------------------------------------------------------------------------


@dataclass
class PairResult:
    """One (scenario, split) pair under one setup: targets and per-envelope flags."""

    scenario: str
    split: str
    specs: tuple[EpisodeSpec, ...]
    targets: np.ndarray  # (N, 7, 3) pelvis frame
    inside: dict[str, np.ndarray]  # envelope name -> (N, 7)
    margin: dict[str, np.ndarray]


def evaluate_setup(
    envelopes: Mapping[str, ReachabilityEnvelope],
    setup: Setup,
    pairs: Iterable[tuple[str, str]] = PAIRS,
    max_seeds: int | None = None,
    margin_m: float = MARGIN_M,
    spec_cache: dict | None = None,
) -> dict[tuple[str, str], PairResult]:
    """Targets and flags of every pair under ``setup``."""
    cache = {} if spec_cache is None else spec_cache
    out = {}
    for scenario, split in pairs:
        key = (scenario, split, setup.plate_centre_y_m, max_seeds)
        if key not in cache:
            cache[key] = episode_specs(scenario, split, setup.plate_centre_y_m, max_seeds)
        specs = cache[key]
        targets = np.stack([pelvis_targets(s, setup) for s in specs])
        inside, margin = {}, {}
        for name, env in envelopes.items():
            inside[name], margin[name] = reach_flags(env, targets, margin_m)
        out[(scenario, split)] = PairResult(scenario, split, specs, targets, inside, margin)
    return out


def plate_overhang(specs: Sequence[EpisodeSpec]) -> float:
    """Fraction of draws whose plate reaches past the counter's front edge (``y - D/2 < 0``)."""
    return float(np.mean([s.plate_position_m[1] - s.plate_diameter_m / 2.0 < 0.0 for s in specs]))


@dataclass
class Analysis:
    """All results of one run; the tables and the figure are built from it."""

    envelopes: dict[str, ReachabilityEnvelope]
    base: Setup
    standoffs: tuple[float, ...]
    plate_ys: tuple[float, ...]
    margin_m: float
    max_seeds: int | None
    grid: dict[tuple[float, float], dict[tuple[str, str], PairResult]]
    rack_sweep: list[dict]
    sensitivity: list[dict]


def run_analysis(
    envelopes: Mapping[str, ReachabilityEnvelope],
    base: Setup | None = None,
    standoffs: Sequence[float] = STANDOFFS_M,
    plate_ys: Sequence[float] = PLATE_YS_M,
    max_seeds: int | None = None,
    margin_m: float = MARGIN_M,
    extra_envelopes: Mapping[str, Mapping[str, ReachabilityEnvelope]] | None = None,
    rack_zs: Sequence[float] = RACK_ZS_M,
    rack_xs: Sequence[float] = RACK_XS_M,
) -> Analysis:
    """The stand-off by plate-box sweep, the provisional rack sweep and the sensitivity rows.

    ``extra_envelopes`` maps a sensitivity label to more envelopes (by name) evaluated under the
    base setup: the other point (palm link origin or finger bases) and the smaller waist ranges
    of :func:`resampled_envelopes`.
    """
    base = base or Setup()
    envs = dict(envelopes)
    cache: dict = {}
    grid = {}
    for standoff in standoffs:
        for plate_y in plate_ys:
            setup = replace(base, standoff_m=float(standoff), plate_centre_y_m=float(plate_y))
            grid[(float(standoff), float(plate_y))] = evaluate_setup(
                envs, setup, PAIRS, max_seeds, margin_m, cache
            )

    s3_pairs = [("S3", sp) for sp in SPLITS]
    rack_rows = []
    for standoff in standoffs:
        for rz in rack_zs:
            for rx in rack_xs:
                pose = (float(rx), base.rack_pose[1], float(rz), base.rack_pose[3])
                setup = replace(base, standoff_m=float(standoff), rack_pose=pose)
                res = evaluate_setup(envs, setup, s3_pairs, max_seeds, margin_m, cache)
                for (scenario, split), r in res.items():
                    for name in envs:
                        rack_rows.append(
                            {
                                "envelope": name,
                                "standoff_m": float(standoff),
                                "rack_x_m": float(rx),
                                "rack_z_m": float(rz),
                                "scenario": scenario,
                                "split": split,
                                "episodes": len(r.specs),
                                "inside": episode_fraction(r.inside[name], SLOT_TARGETS),
                                "margin": episode_fraction(r.margin[name], SLOT_TARGETS),
                            }
                        )

    variants: list[tuple[str, Setup, dict[str, ReachabilityEnvelope]]] = [
        ("baseline", base, envs),
        ("grasp at plate centre", replace(base, grasp_point="centre"), envs),
        ("pelvis 0.72 m", replace(base, pelvis_height_m=base.pelvis_height_m - 0.03), envs),
        ("pelvis 0.78 m", replace(base, pelvis_height_m=base.pelvis_height_m + 0.03), envs),
    ]
    for label, more in (extra_envelopes or {}).items():
        variants.append((label, base, dict(more)))
    s2_pairs = [("S2", sp) for sp in SPLITS]
    sens_rows = []
    for label, setup0, variant_envs in variants:
        for standoff in standoffs:
            setup = replace(setup0, standoff_m=float(standoff))
            res = evaluate_setup(variant_envs, setup, s2_pairs, max_seeds, margin_m, cache)
            for (scenario, split), r in res.items():
                for name in variant_envs:
                    sens_rows.append(
                        {
                            "variant": label,
                            "envelope": name,
                            "standoff_m": float(standoff),
                            "plate_centre_y_m": setup.plate_centre_y_m,
                            "scenario": scenario,
                            "split": split,
                            "episodes": len(r.specs),
                            "inside": episode_fraction(r.inside[name], SCENARIO_TARGETS["S2"]),
                            "margin": episode_fraction(r.margin[name], SCENARIO_TARGETS["S2"]),
                        }
                    )
    return Analysis(
        envs,
        base,
        tuple(float(s) for s in standoffs),
        tuple(float(y) for y in plate_ys),
        margin_m,
        max_seeds,
        grid,
        rack_rows,
        sens_rows,
    )


# --- rows and tables --------------------------------------------------------------------------


def waypoint_rows(analysis: Analysis) -> list[dict]:
    rows = []
    for (standoff, plate_y), pairs in analysis.grid.items():
        for (scenario, split), r in pairs.items():
            for name in analysis.envelopes:
                for j, wp in enumerate(WAYPOINTS):
                    rows.append(
                        {
                            "envelope": name,
                            "standoff_m": standoff,
                            "plate_centre_y_m": plate_y,
                            "scenario": scenario,
                            "split": split,
                            "waypoint": wp,
                            "episodes": len(r.specs),
                            "inside": float(r.inside[name][:, j].mean()),
                            "margin": float(r.margin[name][:, j].mean()),
                        }
                    )
    return rows


def episode_rows(analysis: Analysis) -> list[dict]:
    rows = []
    for (standoff, plate_y), pairs in analysis.grid.items():
        for (scenario, split), r in pairs.items():
            for name in analysis.envelopes:
                for part in PARTS:
                    names = target_set(scenario, part)
                    if not names:
                        continue
                    rows.append(
                        {
                            "envelope": name,
                            "standoff_m": standoff,
                            "plate_centre_y_m": plate_y,
                            "scenario": scenario,
                            "split": split,
                            "part": part,
                            "targets": " ".join(names),
                            "episodes": len(r.specs),
                            "inside": episode_fraction(r.inside[name], names),
                            "margin": episode_fraction(r.margin[name], names),
                        }
                    )
    return rows


def overhang_rows(analysis: Analysis) -> list[dict]:
    rows = []
    for plate_y in analysis.plate_ys:
        pairs = analysis.grid[(analysis.standoffs[-1], plate_y)]
        for split in SPLITS:
            specs = [s for (sc, sp), r in pairs.items() if sp == split for s in r.specs]
            rows.append(
                {
                    "plate_centre_y_m": plate_y,
                    "split": split,
                    "episodes": len(specs),
                    "overhang": plate_overhang(specs),
                    "worst_front_edge_y_m": min(
                        s.plate_position_m[1] - s.plate_diameter_m / 2.0 for s in specs
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return path


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.0f}"


def _pair_cell(a: float | None, b: float | None) -> str:
    return f"{_pct(a)} / {_pct(b)}"


def _md_table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _ep(analysis: Analysis, standoff: float, plate_y: float, scenario: str, split: str,
        part: str, name: str, kind: str = "margin") -> float | None:
    r = analysis.grid[(standoff, plate_y)][(scenario, split)]
    flags = r.margin[name] if kind == "margin" else r.inside[name]
    return episode_fraction(flags, target_set(scenario, part))


def markdown_tables(analysis: Analysis) -> str:
    """The tables of docs/reach_gap_analysis.md, percentages, ``waist0 / waist`` per cell."""
    a = analysis
    base = a.base
    cur = (base.standoff_m, base.plate_centre_y_m)
    if cur not in a.grid:
        cur = (a.standoffs[-1], a.plate_ys[-1])
    w0, w1 = ENVELOPE_NAMES
    out: list[str] = []
    seeds = "all committed seeds" if a.max_seeds is None else f"first {a.max_seeds} seeds per pair"
    out.append(
        f"Cells are percent of episodes, `waist0 / waist` (the #48 envelope / the #56 envelope); "
        f"{seeds}; pelvis height {base.pelvis_height_m:.2f} m; grasp point {base.grasp_point}; "
        f"rack pose {base.rack_pose} (provisional); margin {a.margin_m * 100:.0f} cm."
    )
    out.append("")
    out.append(f"### Per waypoint at the current constants (stand-off {cur[0]:.2f} m, plate box "
               f"centre y {cur[1]:.2f} m)")
    for kind, title in (("inside", "`contains`"), ("margin", "`margin_contains` at 2 cm")):
        out.append("")
        out.append(f"{title}:")
        out.append("")
        rows = []
        for scenario, split in PAIRS:
            r = a.grid[cur][(scenario, split)]
            flags = r.inside if kind == "inside" else r.margin
            cells = [_pair_cell(flags[w0][:, j].mean(), flags[w1][:, j].mean())
                     for j in range(len(WAYPOINTS))]
            rows.append([f"{scenario} {split}", str(len(r.specs)), *cells])
        out += _md_table(["pair", "episodes", *WAYPOINTS], rows)

    out.append("")
    out.append(f"### Per waypoint per stand-off (all ten pairs pooled, plate box centre y "
               f"{cur[1]:.2f} m, 2 cm margin)")
    out.append("")
    rows = []
    for standoff in a.standoffs:
        pairs = a.grid[(standoff, cur[1])]
        cells = []
        for j in range(len(WAYPOINTS)):
            vals = [np.concatenate([r.margin[n][:, j] for r in pairs.values()]).mean()
                    for n in (w0, w1)]
            cells.append(_pair_cell(*vals))
        rows.append([f"{standoff:.2f}", *cells])
    out += _md_table(["stand-off (m)", *WAYPOINTS], rows)

    out.append("")
    out.append(f"### Whole episodes per stand-off (plate box centre y {cur[1]:.2f} m, 2 cm margin)")
    out.append("")
    rows = []
    for scenario, part, label in (
        ("S1", "task", "S1 (pre_grasp, grasp)"),
        ("S2", "task", "S2 (pre_grasp, grasp, lift)"),
        ("S4", "plate", "S4 plate half (+ counter_edge)"),
        ("S3", "task", "S3 slot targets, provisional"),
        ("S4", "task", "S4 all seven, provisional"),
    ):
        for split in SPLITS:
            cells = [_pair_cell(_ep(a, s, cur[1], scenario, split, part, w0),
                                _ep(a, s, cur[1], scenario, split, part, w1))
                     for s in a.standoffs]
            rows.append([label, split, *cells])
    out += _md_table(["episode", "split", *(f"{s:.2f} m" for s in a.standoffs)], rows)

    out.append("")
    out.append("Same, `contains` without margin:")
    out.append("")
    rows = []
    for scenario, part, label in (("S1", "task", "S1"), ("S2", "task", "S2"),
                                  ("S4", "plate", "S4 plate half")):
        for split in SPLITS:
            cells = [_pair_cell(_ep(a, s, cur[1], scenario, split, part, w0, "inside"),
                                _ep(a, s, cur[1], scenario, split, part, w1, "inside"))
                     for s in a.standoffs]
            rows.append([label, split, *cells])
    out += _md_table(["episode", "split", *(f"{s:.2f} m" for s in a.standoffs)], rows)

    out.append("")
    out.append("### S2 episodes per plate box centre y and stand-off (2 cm margin)")
    out.append("")
    over = {(r["plate_centre_y_m"], r["split"]): r for r in overhang_rows(a)}
    rows = []
    for plate_y in a.plate_ys:
        for split in SPLITS:
            cells = [_pair_cell(_ep(a, s, plate_y, "S2", split, "task", w0),
                                _ep(a, s, plate_y, "S2", split, "task", w1))
                     for s in a.standoffs]
            o = over[(plate_y, split)]
            rows.append([f"{plate_y:.2f}", split, *cells, _pct(o["overhang"]),
                         f"{o['worst_front_edge_y_m'] * 100:+.1f}"])
    out += _md_table(
        ["plate y (m)", "split", *(f"{s:.2f} m" for s in a.standoffs),
         "plate past edge (%)", "worst rim y (cm)"],
        rows,
    )

    if a.rack_sweep:
        out.append("")
        out.append("### S3 slot targets against the rack pose (provisional; rack y "
                   f"{base.rack_pose[1]:.2f} m, yaw {base.rack_pose[3]:g} deg; 2 cm margin)")
        for standoff in sorted({r["standoff_m"] for r in a.rack_sweep}):
            out.append("")
            out.append(f"Stand-off {standoff:.2f} m, `waist0 / waist`, seen then unseen per cell:")
            out.append("")
            xs = sorted({r["rack_x_m"] for r in a.rack_sweep})
            zs = sorted({r["rack_z_m"] for r in a.rack_sweep})
            idx = {(r["envelope"], r["standoff_m"], r["rack_x_m"], r["rack_z_m"], r["split"]): r
                   for r in a.rack_sweep}
            rows = []
            for rz in zs:
                cells = []
                for rx in xs:
                    parts = []
                    for split in SPLITS:
                        v0 = idx[(w0, standoff, rx, rz, split)]["margin"]
                        v1 = idx[(w1, standoff, rx, rz, split)]["margin"]
                        parts.append(_pair_cell(v0, v1))
                    cells.append("; ".join(parts))
                rows.append([f"{rz:.2f}", *cells])
            out += _md_table(["rack z (m)", *(f"rack x {x:.2f} m" for x in xs)], rows)

    if a.sensitivity:
        out.append("")
        out.append(f"### Sensitivity: S2 episodes, `seen / unseen` per cell, plate box centre y "
                   f"{base.plate_centre_y_m:.2f} m, 2 cm margin")
        out.append("")
        keys = list(dict.fromkeys((r["variant"], r["envelope"]) for r in a.sensitivity))
        idx = {(r["variant"], r["envelope"], r["standoff_m"], r["split"]): r
               for r in a.sensitivity}
        rows = []
        for label, name in keys:
            cells = [_pair_cell(idx[(label, name, s, "seen")]["margin"],
                                idx[(label, name, s, "unseen")]["margin"])
                     for s in a.standoffs]
            rows.append([label, f"`{name}`", *cells])
        out += _md_table(["variant", "envelope", *(f"{s:.2f} m" for s in a.standoffs)], rows)

    return "\n".join(out) + "\n"


# --- figure -----------------------------------------------------------------------------------

_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e6e5e1"
_SURFACE = "#fcfcfb"
_SERIES = {"waist0": "#2a78d6", "waist": "#eb6834"}
_LABEL = {"waist0": "waist at zero (#48)", "waist": "waist sampled (#56)"}


def _slice(env: ReachabilityEnvelope, axis: int, value: float):
    """The occupied voxel layer containing ``value`` along ``axis`` and its two cell-centre axes."""
    k = int(math.floor((value - env.origin[axis]) / env.spacing))
    k = min(max(k, 0), env.occupied.shape[axis] - 1)
    layer = np.take(env.occupied, k, axis=axis)
    others = [i for i in range(3) if i != axis]
    centres = [env.origin[i] + (np.arange(env.occupied.shape[i]) + 0.5) * env.spacing
               for i in others]
    return layer, centres


def _style(ax) -> None:
    ax.set_facecolor(_SURFACE)
    ax.grid(color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.tick_params(colors=_INK_2, labelsize=9)
    ax.xaxis.label.set_color(_INK_2)
    ax.yaxis.label.set_color(_INK_2)


def plot_figure(analysis: Analysis, out: Path, focus_standoff: float | None = None) -> Path:
    """Side and top slices of both envelopes with the plate targets, and S2 against stand-off.

    The slices show the plate targets of S2 and S4 (all splits) at the current stand-off (hollow)
    and at ``focus_standoff`` (filled, default the smallest of the sweep), with the counter.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    a = analysis
    plate_y = a.base.plate_centre_y_m if a.base.plate_centre_y_m in a.plate_ys else a.plate_ys[-1]
    current = a.base.standoff_m if a.base.standoff_m in a.standoffs else a.standoffs[-1]
    focus = focus_standoff if focus_standoff in a.standoffs else a.standoffs[0]
    targets = {
        s: np.concatenate(
            [a.grid[(s, plate_y)][(sc, sp)].targets for sc, sp in PAIRS if sc in ("S2", "S4")]
        )
        for s in {current, focus}
    }
    y_slice = -0.15
    counter_z = kd.COUNTER_HEIGHT_M - a.base.pelvis_height_m
    z_slice = counter_z + kd.PLATE_HEIGHT_M

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    fig.patch.set_facecolor(_SURFACE)
    views = (
        (axes[0], 1, y_slice, (0, 2), "x, forward of the pelvis (m)", "z, above the pelvis (m)",
         f"Side: envelopes sliced at y = {y_slice:+.2f} m (targets at every y)"),
        (axes[1], 2, z_slice, (0, 1), "x, forward of the pelvis (m)", "y, left of the pelvis (m)",
         f"Top: envelopes sliced at the grasp height, z = {z_slice:+.2f} m"),
    )
    for ax, axis, value, (i, j), xlabel, ylabel, title in views:
        _style(ax)
        if axis == 1:  # the counter block in front of the robot at the focus stand-off
            ax.add_patch(Rectangle((focus, -1.0), 2.0, counter_z + 1.0, facecolor="#ecebe7",
                                   edgecolor="none", zorder=0))
            ax.annotate(f"counter at stand-off {focus:.2f} m", (0.93, -0.27), fontsize=8,
                        color=_INK_2, ha="right", va="bottom")
        for standoff in sorted({current, focus}):
            ax.axvline(standoff, color=_INK_2, lw=1.0, ls=(0, (4, 3)), zorder=1)
            if axis == 2:
                ax.annotate(f"edge, {standoff:.2f} m", (standoff, 0.43), xytext=(3, 0),
                            textcoords="offset points", fontsize=8, color=_INK_2, rotation=90,
                            va="top")
        for name in ENVELOPE_NAMES:
            if name not in a.envelopes:
                continue
            layer, (c0, c1) = _slice(a.envelopes[name], axis, value)
            grid = layer.T.astype(float)
            if name == "waist":
                ax.contourf(c0, c1, grid, levels=[0.5, 1.5], colors=[_SERIES[name]], alpha=0.12)
            ax.contour(c0, c1, grid, levels=[0.5], colors=[_SERIES[name]], linewidths=2.0)
        cols = [WAYPOINTS.index(n) for n in (PLATE_TARGETS if axis == 1 else ("grasp",))]
        for standoff in (current, focus):
            pts = targets[standoff][:, cols, :].reshape(-1, 3)
            filled = standoff == focus
            ax.scatter(pts[:, i], pts[:, j], s=9, marker="o",
                       facecolors=_INK if filled else "none",
                       edgecolors=_INK if filled else _INK_2, linewidths=0.6,
                       alpha=0.5 if filled else 0.45, zorder=3)
        ax.plot([0.0], [0.0], marker="+", color=_INK, markersize=11, mew=1.6, zorder=4)
        ax.annotate("pelvis", (0.0, 0.0), xytext=(6, -12), textcoords="offset points",
                    fontsize=9, color=_INK_2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10.5, color=_INK, loc="left")
        ax.set_aspect("equal")
    axes[0].set_xlim(-0.3, 0.95)
    axes[0].set_ylim(-0.3, 0.8)
    axes[1].set_xlim(-0.3, 0.95)
    axes[1].set_ylim(-0.7, 0.45)

    ax = axes[2]
    _style(ax)
    xs = np.array(a.standoffs)
    for name in ENVELOPE_NAMES:
        if name not in a.envelopes:
            continue
        for split, ls, face in (("seen", "-", _SERIES[name]), ("unseen", (0, (5, 3)), _SURFACE)):
            ys = [100.0 * _ep(a, s, plate_y, "S2", split, "task", name) for s in a.standoffs]
            ax.plot(xs, ys, ls=ls, lw=2.0, color=_SERIES[name], marker="o", ms=6,
                    mfc=face, mec=_SERIES[name], mew=1.6, label=f"{_LABEL[name]}, {split}")
    ax.set_xlabel("stand-off from the counter edge (m)")
    ax.set_ylabel("S2 episodes with every target reachable (%)")
    ax.set_title(f"S2 vs stand-off (2 cm margin, plate y {plate_y:.2f} m)", fontsize=10.5,
                 color=_INK, loc="left")
    ax.set_ylim(-3, 103)
    ax.set_xticks(xs)
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=_INK_2, handlelength=3.2)

    handles = [
        Patch(facecolor="none", edgecolor=_SERIES["waist0"], lw=2, label=_LABEL["waist0"]),
        Patch(facecolor=_SERIES["waist"], alpha=0.35, edgecolor=_SERIES["waist"], lw=2,
              label=_LABEL["waist"]),
        Line2D([], [], marker="o", ls="", mfc="none", mec=_INK_2, ms=5,
               label=f"plate targets at stand-off {current:.2f} m"),
        Line2D([], [], marker="o", ls="", mfc=_INK, mec=_INK, ms=5,
               label=f"plate targets at stand-off {focus:.2f} m"),
    ]
    fig.legend(handles=handles, loc="lower left", ncol=4, fontsize=9, frameon=False,
               labelcolor=_INK_2, bbox_to_anchor=(0.02, 0.0))
    fig.suptitle(
        "Right-palm reach (URDF FK envelopes, pelvis frame) against the planner's plate targets "
        f"of S2 and S4, pelvis {a.base.pelvis_height_m:.2f} m above the floor",
        fontsize=12, color=_INK,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90, facecolor=_SURFACE)
    plt.close(fig)
    return out


# --- entry point ------------------------------------------------------------------------------


def _resample(
    chain, like: ReachabilityEnvelope, waist_range=None, tip_offset_m=None
) -> ReachabilityEnvelope:
    """``like``'s sampling (seed, samples, spacing, wrist margin, waist range, point) with another
    waist range or point."""
    return build_envelope(
        chain,
        samples=like.samples,
        seed=like.seed,
        spacing=like.spacing,
        wrist_margin_deg=like.wrist_margin_deg,
        joint_source=like.joint_source,
        waist_range=like.waist_range_deg if waist_range is None else waist_range,
        tip_offset_m=like.tip_offset_m if tip_offset_m is None else tip_offset_m,
    )


def resampled_envelopes(
    envelopes: Mapping[str, ReachabilityEnvelope],
    joints_path: Path = DEFAULT_JOINTS,
    offset: Sequence[float] = FINGER_BASE_OFFSET_M,
    waist_variants: Mapping[str, Sequence[Sequence[float]]] = WAIST_VARIANTS_DEG,
) -> dict[str, dict[str, ReachabilityEnvelope]]:
    """The sensitivity envelopes, in memory: both envelopes for the other point, and the waist
    envelope's sampling (its point too) with each smaller waist range.

    The other point is ``offset`` (the finger bases) for envelopes of the palm link origin, as in
    #56, and the palm link origin for envelopes that already sample ``offset``, as the committed
    ones do since #66.
    """
    joints = json.loads(Path(joints_path).read_text(encoding="utf-8"))["joints"]
    chain = right_arm_chain(joints)
    offset = tuple(float(v) for v in offset)
    if all(env.tip_offset_m == offset for env in envelopes.values()):
        label, other = "palm-link point", (0.0, 0.0, 0.0)
    else:
        label, other = "finger-base point", offset
    out = {
        label: {name: _resample(chain, env, tip_offset_m=other) for name, env in envelopes.items()}
    }
    like = envelopes["waist"]
    for label, waist in waist_variants.items():
        out[label] = {"waist": _resample(chain, like, waist_range=waist)}
    return out


def write_outputs(analysis: Analysis, out_dir: Path, figure: Path | None = None,
                  focus_standoff: float | None = None) -> dict[str, Path]:
    """CSV files, ``tables.md`` and the figure; returns their paths by name."""
    out_dir = Path(out_dir)
    paths = {
        "waypoints": write_csv(out_dir / "waypoints.csv", waypoint_rows(analysis)),
        "episodes": write_csv(out_dir / "episodes.csv", episode_rows(analysis)),
        "plate_overhang": write_csv(out_dir / "plate_overhang.csv", overhang_rows(analysis)),
    }
    if analysis.rack_sweep:
        paths["rack_sweep"] = write_csv(out_dir / "rack_sweep.csv", analysis.rack_sweep)
    if analysis.sensitivity:
        paths["sensitivity"] = write_csv(out_dir / "sensitivity.csv", analysis.sensitivity)
    tables = out_dir / "tables.md"
    tables.write_text(markdown_tables(analysis), encoding="utf-8")
    paths["tables"] = tables
    paths["figure"] = plot_figure(analysis, figure or out_dir / "reach_gap.png", focus_standoff)
    return paths


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.split(","))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--envelope", type=Path, default=DEFAULT_PATH, help="waist-at-zero .npz")
    parser.add_argument("--waist-envelope", type=Path, default=WAIST_PATH, help="waist .npz")
    parser.add_argument("--standoff", type=float, default=kd.ROBOT_STANDOFF_M,
                        help="current stand-off, m (the 'before' column)")
    parser.add_argument("--standoffs", type=_floats, default=STANDOFFS_M,
                        help="comma-separated sweep, m")
    parser.add_argument("--plate-ys", type=_floats, default=PLATE_YS_M,
                        help="comma-separated plate box centre y sweep, m")
    parser.add_argument("--pelvis-height", type=float, default=PELVIS_HEIGHT_M,
                        help="pelvis height above the floor, m")
    parser.add_argument("--rack-pose", type=float, nargs=4, default=PROVISIONAL_RACK_POSE,
                        metavar=("X", "Y", "Z", "YAW_DEG"),
                        help="kitchen pose of the rack frame (provisional default)")
    parser.add_argument("--rack-zs", type=_floats, default=RACK_ZS_M, help="rack z sweep, m")
    parser.add_argument("--rack-xs", type=_floats, default=RACK_XS_M, help="rack x sweep, m")
    parser.add_argument("--grasp-point", choices=("near_rim", "centre"), default="near_rim")
    parser.add_argument("--margin", type=float, default=MARGIN_M, help="margin_contains, m")
    parser.add_argument("--max-seeds", type=int, default=None, help="first N seeds per pair")
    parser.add_argument("--no-resample", action="store_true",
                        help="skip the re-sampled envelopes of the sensitivity table (the other "
                        "point, smaller waist ranges)")
    parser.add_argument("--focus-standoff", type=float, default=None,
                        help="stand-off drawn filled in the figure (default the smallest)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--figure", type=Path, default=None, help="PNG path")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    envelopes = {
        "waist0": ReachabilityEnvelope.load(args.envelope),
        "waist": ReachabilityEnvelope.load(args.waist_envelope),
    }
    base = Setup(
        standoff_m=args.standoff,
        pelvis_height_m=args.pelvis_height,
        rack_pose=tuple(args.rack_pose),
        grasp_point=args.grasp_point,
    )
    extra = None if args.no_resample else resampled_envelopes(envelopes)
    analysis = run_analysis(
        envelopes,
        base,
        standoffs=args.standoffs,
        plate_ys=args.plate_ys,
        max_seeds=args.max_seeds,
        margin_m=args.margin,
        extra_envelopes=extra,
        rack_zs=args.rack_zs,
        rack_xs=args.rack_xs,
    )
    paths = write_outputs(analysis, args.out_dir, args.figure, args.focus_standoff)
    print(markdown_tables(analysis))
    for name, path in paths.items():
        print(f"{name}: {path} ({path.stat().st_size} bytes)")
    print(f"done in {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
