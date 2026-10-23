"""URDF joint origins in the dump (#48) and the forward kinematics built on them. No simulator."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "g1_dex3_joints_urdf.json"
SCRIPT = REPO / "scripts" / "smoke" / "dump_urdf_joints.py"


@pytest.fixture(scope="module")
def dump():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def joints(dump):
    return dump["joints"]


# --- origins in the dump ----------------------------------------------------------------------

ORIGIN_URDF = """<robot name="origins">
  <link name="base"/>
  <link name="a"/>
  <link name="b"/>
  <link name="c"/>
  <joint name="j_full" type="fixed">
    <origin xyz="0.1 -0.2 0.3" rpy="0.5 0 -1.5"/><parent link="base"/><child link="a"/>
  </joint>
  <joint name="j_xyz_only" type="revolute">
    <origin xyz="0 0 1E-2"/><parent link="a"/><child link="b"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="j_none" type="continuous"><parent link="b"/><child link="c"/></joint>
</robot>
"""


def test_dump_records_origins_with_urdf_defaults(tmp_path):
    urdf = tmp_path / "origins.urdf"
    out = tmp_path / "origins.json"
    urdf.write_text(ORIGIN_URDF, encoding="utf-8")
    subprocess.run([sys.executable, str(SCRIPT), str(urdf), str(out)], check=True)
    full, xyz_only, none = json.loads(out.read_text(encoding="utf-8"))["joints"]
    assert (full["origin_xyz"], full["origin_rpy"]) == ([0.1, -0.2, 0.3], [0.5, 0.0, -1.5])
    assert (xyz_only["origin_xyz"], xyz_only["origin_rpy"]) == ([0.0, 0.0, 0.01], [0.0, 0.0, 0.0])
    assert (none["origin_xyz"], none["origin_rpy"]) == ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_fixture_has_an_origin_on_every_joint(joints):
    for joint in joints:
        assert len(joint["origin_xyz"]) == 3 and len(joint["origin_rpy"]) == 3, joint["name"]
    by_name = {j["name"]: j for j in joints}
    # spot checks against the URDF text at the recorded commit
    assert by_name["right_shoulder_pitch_joint"]["origin_xyz"] == [0.0039563, -0.10021, 0.24778]
    assert by_name["right_shoulder_pitch_joint"]["origin_rpy"] == [-0.27931, 5.4949e-05, 0.00019159]
    assert by_name["right_hand_palm_joint"]["origin_xyz"] == [0.0415, -0.003, 0.0]
    assert by_name["waist_roll_joint"]["origin_xyz"] == [-0.0039635, 0.0, 0.044]
