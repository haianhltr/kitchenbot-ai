"""Dump the joints and links of a URDF to JSON. Standard library only, no ROS, no simulator.

Usage::

    python scripts/smoke/dump_urdf_joints.py <urdf> <out.json> \
        [--source-url URL --source-commit SHA --source-licence SPDX --source-fetched YYYY-MM-DD]

Joints are listed in file order (the order of the ``<joint>`` elements, commented-out ones are
ignored) with name, type, parent link, child link, origin, axis, lower, upper, effort and velocity.
The origin is the joint's ``<origin xyz rpy>``, the transform from the parent link frame to the
joint (child link) frame, as ``origin_xyz`` (m) and ``origin_rpy`` (rad, fixed-axis roll, pitch,
yaw); a missing element or attribute is the URDF default, zeros (added in #48 so the forward
kinematics in ``kitchenbot.robots.kinematics`` can be built from the dump). Fixed joints are listed
too, with their origins; they carry ``null`` for axis and limits and are never "movable". Movable
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


def _origin(element: ET.Element) -> tuple[list[float], list[float]]:
    """``(xyz, rpy)`` of a joint's ``<origin>``; the URDF default for anything missing is zeros."""
    origin_el = element.find("origin")
    attrib = origin_el.attrib if origin_el is not None else {}
    xyz = _floats(attrib.get("xyz")) or [0.0, 0.0, 0.0]
    rpy = _floats(attrib.get("rpy")) or [0.0, 0.0, 0.0]
    if len(xyz) != 3 or len(rpy) != 3:
        raise ValueError(f"joint {element.get('name')!r}: origin needs 3 values in xyz and rpy")
    return xyz, rpy


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
    origin_xyz, origin_rpy = _origin(element)
    return {
        "name": element.get("name"),
        "type": jtype,
        "movable": movable,
        "parent": parent_el.get("link") if parent_el is not None else None,
        "child": child_el.get("link") if child_el is not None else None,
        "origin_xyz": origin_xyz,
        "origin_rpy": origin_rpy,
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
    # newline="\n": identical bytes on Windows and Linux, so the committed fixture never churns
    args.out.write_text(json.dumps(dump, indent=2) + "\n", encoding="utf-8", newline="\n")
    c = dump["counts"]
    print(
        f"{dump['robot']}: {c['joints']} joints ({c['movable']} movable, {c['fixed']} fixed), "
        f"{c['links']} links -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
