"""Dump the joints and links of a URDF to JSON. Standard library only, no ROS, no simulator.

Usage::

    python scripts/smoke/dump_urdf_joints.py <urdf> <out.json> \
        [--source-url URL --source-commit SHA --source-licence SPDX --source-fetched YYYY-MM-DD]

Joints are listed in file order (the order of the ``<joint>`` elements, commented-out ones are
ignored) with name, type, parent link, child link, axis, lower, upper, effort and velocity. Fixed
joints are listed too; they carry ``null`` for axis and limits and are never "movable". Movable
joints are the revolute, prismatic and continuous ones: those are the joints that become degrees of
freedom in a simulator, and the ones ``kitchenbot.robots.joint_maps`` partitions into groups.

Used by issue #28 to turn Unitree's ``g1_29dof_with_hand_rev_1_0.urdf`` into the committed fixture
``tests/fixtures/g1_dex3_joints_urdf.json``. Issue #3 later dumps the same facts from the Isaac Sim
USD and diffs them against that fixture.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

MOVABLE_TYPES = ("revolute", "prismatic", "continuous")
# URDF spec: a movable joint without an <axis> element rotates or slides about x.
DEFAULT_AXIS = [1.0, 0.0, 0.0]


def _floats(text: str | None) -> list[float] | None:
    return None if text is None else [float(v) for v in text.split()]


def _opt_float(value: str | None) -> float | None:
    return None if value is None else float(value)


def parse_joint(element: ET.Element) -> dict:
    """Return one joint as a plain dict (JSON serialisable)."""
    jtype = element.get("type", "")
    movable = jtype in MOVABLE_TYPES
    axis_el = element.find("axis")
    limit_el = element.find("limit")
    axis = None
    if movable:
        axis = _floats(axis_el.get("xyz")) if axis_el is not None else list(DEFAULT_AXIS)
    limit = limit_el.attrib if limit_el is not None else {}
    parent_el = element.find("parent")
    child_el = element.find("child")
    return {
        "name": element.get("name"),
        "type": jtype,
        "movable": movable,
        "parent": parent_el.get("link") if parent_el is not None else None,
        "child": child_el.get("link") if child_el is not None else None,
        "axis": axis,
        # continuous joints have no lower/upper; fixed joints have no <limit> at all
        "lower": _opt_float(limit.get("lower")),
        "upper": _opt_float(limit.get("upper")),
        "effort": _opt_float(limit.get("effort")),
        "velocity": _opt_float(limit.get("velocity")),
    }


def dump_urdf(urdf_path: Path, source: dict | None = None) -> dict:
    """Parse ``urdf_path`` and return the dump as a dict."""
    root = ET.parse(urdf_path).getroot()
    if root.tag != "robot":
        raise ValueError(f"{urdf_path}: root element is <{root.tag}>, expected <robot>")
    joints = [parse_joint(el) for el in root.findall("joint")]
    links = [el.get("name") for el in root.findall("link")]
    movable = [j["name"] for j in joints if j["movable"]]
    out: dict = {}
    if source is not None:
        out["source"] = source
    out.update(
        {
            "robot": root.get("name"),
            "counts": {
                "joints": len(joints),
                "movable": len(movable),
                "fixed": sum(1 for j in joints if j["type"] == "fixed"),
                "links": len(links),
            },
            "joints": joints,
            "links": links,
        }
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("urdf", type=Path, help="input URDF file")
    parser.add_argument("out", type=Path, help="output JSON file")
    parser.add_argument("--source-url", help="where the URDF was downloaded from")
    parser.add_argument("--source-commit", help="commit SHA of the repository it came from")
    parser.add_argument("--source-licence", help="SPDX licence identifier of that repository")
    parser.add_argument("--source-fetched", help="date fetched, YYYY-MM-DD")
    args = parser.parse_args(argv)

    source = None
    if any([args.source_url, args.source_commit, args.source_licence, args.source_fetched]):
        source = {
            "url": args.source_url,
            "commit": args.source_commit,
            "licence": args.source_licence,
            "fetched": args.source_fetched,
        }
    dump = dump_urdf(args.urdf, source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dump, indent=2) + "\n", encoding="utf-8")
    c = dump["counts"]
    print(
        f"{dump['robot']}: {c['joints']} joints ({c['movable']} movable, {c['fixed']} fixed), "
        f"{c['links']} links -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
