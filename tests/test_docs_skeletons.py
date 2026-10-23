"""The report skeleton and the demo video script hold placeholders, never results (#49).

``docs/report.md`` and ``docs/demo_video_script.md`` fix the shape of the final report and of the
video before any number exists. These tests keep them honest: every empty slot is a to-do comment
that names a real issue, no results table carries a digit, the video lasts the 2 to 3 minutes of
docs/evaluation.md section 8 and every shot has a label from the architecture's vocabulary, and
the pieces copied from other documents (decision titles, licences, commands, figure names) still
agree with their sources.
"""

import ast
import json
import re
from pathlib import Path

import pytest

from kitchenbot.evaluation.failure_tags import FAILURE_TAG_VALUES

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
REPORT = DOCS / "report.md"
SCRIPT = DOCS / "demo_video_script.md"
BOTH = pytest.mark.parametrize("path", [REPORT, SCRIPT], ids=lambda p: p.name)

# Issue numbers a placeholder may name. docs/roadmap.md maps P01 to P26 onto the issue numbers
# one to one, so 1 to 26 must have their row in its table. The roadmap note lists the numbers added
# later (IN_ROADMAP_NOTE). The issues after it were opened the day the skeletons were written and
# are not in the roadmap table yet (OPENED_AFTER_ROADMAP). The director keeps both sets current.
IN_ROADMAP_NOTE = frozenset({28, 29, 30, 35, 40, 42, 43})
OPENED_AFTER_ROADMAP = frozenset({46, 47, 48, 49})

TODO_COMMENT = re.compile(r"<!--\s*TODO\(#(\d+)\):\s*(\S.*?)-->", re.DOTALL)

REPORT_SECTIONS = (
    "Abstract",
    "Problem and scope",
    "What was built",
    "Toolchain and host",
    "Reused versus developed",
    "Method",
    "Evaluation protocol",
    "Results",
    "Baselines and ablations",
    "Failure analysis",
    "Compute (GPU time)",
    "Decisions taken",
    "Limitations and honest caveats",
    "Next steps toward the KitchenBot vision",
    "Reproduction",
    "Traceability",
    "Acknowledgements and licences",
)
SUBSYSTEMS = ("lower-body", "arm controller", "hand controllers", "task planner", "scene")
SCENARIOS = ("S0", "S1", "S2", "S3", "S4")
# Sections whose tables hold measured values once the results exist.
RESULT_SECTIONS = ("Results", "Baselines and ablations", "Failure analysis", "Compute (GPU time)")
# What may contain a digit in a results table row: an issue number in a placeholder, a scenario
# name, a controller identifier (Dex1, Dex3), the policy's version label and the name of the
# checksum column. Anything else with a digit would be a number someone typed in.
ALLOWED_IN_RESULT_ROWS = re.compile(
    r"TODO\(#\d+\)|\bS[0-4]\b|\bDex[13]\w*|\bown policy v1\b|\bSHA256\b", re.IGNORECASE
)

PLANNER_STATES = (
    "idle",
    "pre_grasp",
    "grasp",
    "lift",
    "transport",
    "align",
    "insert",
    "release",
    "done",
    "failed",
)
LABEL_ROWS = ("LOWER BODY", "ARM", "HAND", "STATE")
LABEL_VALUES = {
    "LOWER BODY": {"own policy v1", "Agile baseline", "pelvis fixed (no policy)"},
    "ARM": {"held posture (no IK)", "DifferentialIK (right), held posture (left)"},
    "HAND": {
        "held open (no grasp)",
        "KinematicFollowHand (attach baseline)",
        "Dex3SynergyHand",
        "Dex1GripperHand",
    },
    "STATE": {"n/a (no planner)", *PLANNER_STATES},
}
FAILURE_RANKS = ("most frequent", "second most frequent", "third most frequent")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_headings(text: str, prefix: str) -> dict[str, str]:
    """Map each heading that starts with ``prefix`` to the text below it (code fences skipped)."""
    out: dict[str, list[str]] = {}
    title = None
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith(prefix):
            title = line[len(prefix) :].strip()
            out[title] = []
            continue
        if title is not None:
            out[title].append(line)
    return {name: "\n".join(lines) for name, lines in out.items()}


def sections(path: Path) -> dict[str, str]:
    return split_headings(read(path), "## ")


def tables(text: str) -> list[list[str]]:
    """Every markdown table in ``text`` as its list of row strings, header and rule included."""
    found: list[list[str]] = []
    current: list[str] = []
    fenced = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
        if not fenced and stripped.startswith("|"):
            current.append(stripped)
        elif current:
            found.append(current)
            current = []
    if current:
        found.append(current)
    return found


def cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def code_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("```"):
            if current is None:
                current = []
            else:
                blocks.append(current)
                current = None
        elif current is not None:
            current.append(line)
    return blocks


def planned_ids() -> set[int]:
    return {int(n) for n in re.findall(r"^\| P(\d{2}) \|", read(DOCS / "roadmap.md"), re.M)}


@BOTH
def test_skeleton_exists_with_lf_endings(path):
    assert path.is_file(), f"{path.name} is missing"
    raw = path.read_bytes()
    assert raw.strip(), f"{path.name} is empty"
    assert b"\r" not in raw, f"{path.name} must use LF line endings"


def test_issue_number_constants_agree_with_the_roadmap():
    planned = planned_ids()
    assert set(range(1, 27)) <= planned, "the roadmap table lost a row between P01 and P26"
    assert IN_ROADMAP_NOTE <= planned, "the roadmap table lost an issue this test accepts"


@BOTH
def test_every_todo_names_a_known_issue(path):
    text = read(path)
    found = TODO_COMMENT.findall(text)
    assert found, f"{path.name} has no placeholders"
    assert text.count("TODO") == len(found), (
        "a TODO that is not a comment of the form <!-- TODO(#n): what goes here -->"
    )
    planned = planned_ids()
    known = {n for n in range(1, 27) if n in planned} | IN_ROADMAP_NOTE | OPENED_AFTER_ROADMAP
    unknown = sorted({int(n) for n, _ in found} - known)
    assert not unknown, f"{path.name} names issues that are not in the plan: {unknown}"


@BOTH
def test_every_table_row_has_as_many_cells_as_its_header(path):
    for table in tables(read(path)):
        width = len(cells(table[0]))
        for row in table:
            assert len(cells(row)) == width, f"{path.name}: {row[:80]}"


def test_report_has_the_sections_in_order():
    titles = list(sections(REPORT))
    positions = []
    for wanted in REPORT_SECTIONS:
        assert wanted in titles, f"report section missing: {wanted}"
        positions.append(titles.index(wanted))
    assert positions == sorted(positions), "report sections are out of order"


def test_report_has_a_paragraph_slot_per_subsystem_and_per_scenario():
    parts = sections(REPORT)
    for name in ("What was built", "Method"):
        headings = [h.lower() for h in split_headings(parts[name], "### ")]
        for subsystem in SUBSYSTEMS:
            assert any(subsystem in h for h in headings), f"{name}: no subsection for {subsystem}"
    results = split_headings(parts["Results"], "### ")
    for scenario in SCENARIOS:
        assert any(h.startswith(scenario) for h in results), f"Results: no {scenario} subsection"


def test_report_binds_figures_and_tables_to_their_sources():
    text = read(REPORT)
    docstring = ast.get_docstring(ast.parse(read(REPO / "scripts" / "plot_results.py")))
    names = set(re.findall(r"\b[a-z_]+\.png\b", docstring))
    assert names, "scripts/plot_results.py no longer names its figures"
    for name in sorted(names):
        assert f"outputs/figures/{name}" in text, f"report does not name outputs/figures/{name}"
    assert "## Evaluation results" in read(DOCS / "experiments.md")
    assert "Evaluation results" in text and "(experiments.md)" in text
    assert "(evaluation.md)" in text and "(architecture.md)" in text


def test_report_failure_analysis_lists_the_taxonomy_in_precedence_order():
    section = sections(REPORT)["Failure analysis"]
    positions = []
    for tag in FAILURE_TAG_VALUES:
        assert f"`{tag}`" in section, f"failure tag missing from the report: {tag}"
        positions.append(section.index(f"`{tag}`"))
    assert positions == sorted(positions), "tags must be listed in the precedence order"


def test_report_results_tables_hold_no_numbers_and_every_row_is_bound():
    parts = sections(REPORT)
    for title in RESULT_SECTIONS:
        found = tables(parts[title])
        assert found, f"{title} has no table"
        for table in found:
            for row in table:
                leftover = ALLOWED_IN_RESULT_ROWS.sub("", row)
                assert not re.search(r"\d", leftover), f"{title}: a digit in {row[:100]}"
            for row in table[2:]:
                assert TODO_COMMENT.search(row), f"{title}: row bound to no issue: {row[:100]}"


def test_report_lists_every_decision_by_title():
    headings = re.findall(
        r"^## (D\d{3}) — (.+?) \(\d{4}-\d{2}-\d{2}\)$", read(DOCS / "decisions.md"), re.M
    )
    titles = dict(headings)
    section = sections(REPORT)["Decisions taken"]
    for number in range(1, 12):
        key = f"D{number:03d}"
        assert key in titles, f"docs/decisions.md lost {key}"
        assert f"{key}: {titles[key]}" in section, f"report does not list {key} by its title"


def test_report_acknowledges_every_manifest_source_and_the_unitree_urdf():
    manifest = read(REPO / "source/kitchenbot/scene/assets/third_party/MANIFEST.md")
    sources = manifest.split("## Sources", 1)[1].split("\n## ", 1)[0]
    rows = [r for r in sources.splitlines() if r.startswith("| ") and "`" in r.split("|")[1]]
    assert rows, "could not read the sources table of MANIFEST.md"
    text = sections(REPORT)["Acknowledgements and licences"]
    components = [cells(row)[0] for row in tables(text)[0][2:]]
    for row in rows:
        source, _url, _revision, licence, reference = cells(row)
        name = re.match(r"[^(`]+", source).group(0).strip()
        assert any(c.startswith(name) for c in components), f"report does not acknowledge {name}"
        assert reference in text, f"report does not cite {reference} for {name}"
        for token in re.findall(r"[A-Za-z]+(?:-[A-Za-z0-9.]+)+", licence):
            assert token in text, f"report does not state the licence {token} of {name}"
    urdf = json.loads(read(REPO / "tests/fixtures/g1_dex3_joints_urdf.json"))
    assert "Unitree" in text and urdf["source"]["licence"] in text


def test_report_reproduction_commands_are_copied_from_the_documents():
    corpus = read(DOCS / "setup.md") + read(DOCS / "evaluation.md")
    blocks = code_blocks(sections(REPORT)["Reproduction"])
    lines = [line.strip() for block in blocks for line in block if line.strip()]
    assert lines, "the reproduction section has no commands"
    invented = [line for line in lines if line not in corpus]
    assert not invented, f"commands that setup.md and evaluation.md do not contain: {invented}"


def test_evaluation_section_8_links_both_skeletons():
    section = read(DOCS / "evaluation.md").split("## 8.", 1)[1]
    assert "(report.md)" in section and "(demo_video_script.md)" in section


def video_shots() -> list[dict[str, str]]:
    table = tables(sections(SCRIPT)["Shot list"])[0]
    keys = ("shot", "scenario", "base", "camera", "seconds", "label", "failure")
    return [dict(zip(keys, cells(row), strict=True)) for row in table[2:]]


def shot_details() -> dict[str, list[list[str]]]:
    details = split_headings(sections(SCRIPT)["Shot details"], "### ")
    return {
        re.match(r"Shot (\d+)", title).group(1): code_blocks(body)
        for title, body in details.items()
    }


def test_video_script_lasts_two_to_three_minutes():
    shots = video_shots()
    total = sum(int(shot["seconds"]) for shot in shots)
    assert 120 <= total <= 180, f"{len(shots)} shots add up to {total} s"


def test_video_shots_are_numbered_and_each_has_a_label_block():
    shots = video_shots()
    assert [s["shot"] for s in shots] == [f"{i:02d}" for i in range(1, len(shots) + 1)]
    details = shot_details()
    assert list(details) == [s["shot"] for s in shots], "shot details do not match the shot list"
    for shot in shots:
        assert shot["scenario"] in (*SCENARIOS, "card"), shot
        assert shot["label"].strip() and shot["camera"].strip(), shot
        blocks = details[shot["shot"]]
        assert blocks and all(any(line.strip() for line in b) for b in blocks), shot["shot"]


def test_video_labels_use_the_architecture_vocabulary_and_agree_with_the_table():
    architecture = read(DOCS / "architecture.md")
    for value in ("DifferentialIK", "KinematicFollowHand", "Dex3SynergyHand", "Dex1GripperHand"):
        assert value in architecture, f"architecture.md no longer names {value}"
    for state in PLANNER_STATES[:-1]:
        assert state in architecture, f"architecture.md no longer names the state {state}"
    details = shot_details()
    for shot in video_shots():
        if shot["scenario"] == "card":
            continue
        panels = []
        for block in details[shot["shot"]]:
            rows = {}
            for line in block:
                match = re.match(r"(LOWER BODY|ARM|HAND|STATE)\s{2,}(.+)$", line)
                if match:
                    rows[match.group(1)] = match.group(2).strip()
            if rows:
                panels.append((rows, block))
        assert panels, f"shot {shot['shot']} has no label panel"
        for rows, block in panels:
            assert tuple(rows) == LABEL_ROWS, f"shot {shot['shot']}: rows {tuple(rows)}"
            for name, value in rows.items():
                assert value in LABEL_VALUES[name], f"shot {shot['shot']}: {name} {value!r}"
            early = shot["scenario"] in ("S0", "S1")
            assert (rows["ARM"] == "held posture (no IK)") == (shot["scenario"] == "S0")
            assert (rows["HAND"] == "held open (no grasp)") == early
            assert (rows["STATE"] == "n/a (no planner)") == early
            fixed = rows["LOWER BODY"] == "pelvis fixed (no policy)"
            assert fixed == (shot["base"] == "fixed pelvis"), shot
            failing = any(line.startswith("FAILURE: <tag>") for line in block)
            assert failing == (shot["failure"] != "none"), shot
        shown = {rows["LOWER BODY"] for rows, _ in panels}
        assert shown == set(re.findall(r"`([^`]+)`", shot["label"])), shot
        assert shot["failure"] in ("none", *FAILURE_RANKS), shot


def test_video_overlay_specification_is_complete():
    spec = sections(SCRIPT)["Overlay specification"]
    for needle in (
        "record_demo.py",
        "frame height",
        "seed",
        "commit",
        "own policy v1",
        "Agile baseline",
        "top left",
        "bottom right",
    ):
        assert needle in spec, f"overlay specification does not mention {needle!r}"
    assert len(set(re.findall(r"#[0-9A-Fa-f]{6}\b", spec))) >= 2, "no colours in the specification"
    for tag in FAILURE_TAG_VALUES:
        assert f"`{tag}`" in spec, f"overlay specification does not list the tag {tag}"
