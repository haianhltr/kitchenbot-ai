"""configs/eval/*.yaml (#52, #78): the fourteen files equal a fresh generation; the --check mode.

Ten files, one per scenario and split, with the evaluator's default floating base, and since #78
four fixed-pelvis variants of S2 and S3 (docs/evaluation.md section 1).
"""

import pytest
import yaml
from eval_helpers import REPO, load_script

import kitchenbot
from kitchenbot.evaluation.runner import FIXED_BASE_SCENARIOS, EvalConfig
from kitchenbot.evaluation.seeds import SPLITS, default_seeds
from kitchenbot.scene.randomization import SCENARIOS, RandomizationRanges

CONFIG_DIR = REPO / "configs" / "eval"
make = load_script("make_eval_configs")

EXPECTED_NUM_ENVS = {  # docs/evaluation.md section 4, step 2
    ("S0", "seen"): 100,
    ("S1", "seen"): 100,
    ("S2", "seen"): 100,
    ("S3", "seen"): 100,
    ("S4", "seen"): 50,
    ("S0", "unseen"): 50,
    ("S1", "unseen"): 50,
    ("S2", "unseen"): 50,
    ("S3", "unseen"): 50,
    ("S4", "unseen"): 50,
}
BOTH = [(sc, sp) for sc in SCENARIOS for sp in SPLITS]
FIXED = [(sc, sp) for sc in FIXED_BASE_SCENARIOS for sp in SPLITS]
ALL = make.variants()  # (scenario, split, base) of all fourteen files


def _load(scenario, split, base=make.DEFAULT_BASE):
    path = CONFIG_DIR / make.file_name(scenario, split, base)
    return yaml.safe_load(path.read_text("utf-8"))


# ------------------------------------------------------------------ the fourteen files


def test_there_are_exactly_fourteen_named_files():
    assert make.all_file_names() == [
        "s0_seen.yaml", "s0_unseen.yaml", "s1_seen.yaml", "s1_unseen.yaml",
        "s2_seen.yaml", "s2_seen_fixed.yaml", "s2_unseen.yaml", "s2_unseen_fixed.yaml",
        "s3_seen.yaml", "s3_seen_fixed.yaml", "s3_unseen.yaml", "s3_unseen_fixed.yaml",
        "s4_seen.yaml", "s4_unseen.yaml",
    ]  # fmt: skip
    assert set(make.all_file_names()) <= {p.name for p in CONFIG_DIR.glob("*.yaml")}
    assert (make.DEFAULT_BASE, make.FIXED_BASE) == ("floating", "fixed")


@pytest.mark.parametrize(("scenario", "split", "base"), ALL)
def test_committed_file_is_byte_identical_to_a_fresh_generation(scenario, split, base):
    committed = (CONFIG_DIR / make.file_name(scenario, split, base)).read_bytes()
    assert committed == make.render(scenario, split, base).encode("utf-8")
    assert b"\r" not in committed and committed.endswith(b"\n")


@pytest.mark.parametrize(("scenario", "split", "base"), ALL)
def test_each_file_names_physx_and_the_evaluator_defaults(scenario, split, base):
    cfg = _load(scenario, split, base)
    defaults = EvalConfig(scenario=scenario, split=split)
    assert list(cfg) == [
        "scenario", "split", "physics", "num_envs", "seeds", "policy", "hand", "base",
        "randomization",
    ]  # fmt: skip
    assert cfg["physics"] == "isaacsim_physx" == kitchenbot.PHYSICS_PRESET
    assert (cfg["scenario"], cfg["split"]) == (scenario, split)
    assert (cfg["policy"], cfg["hand"]) == ("own", "dex3") == (defaults.policy, defaults.hand)
    assert cfg["base"] == base
    if base == make.DEFAULT_BASE:
        assert cfg["base"] == "floating" == defaults.base
    assert cfg["num_envs"] == EXPECTED_NUM_ENVS[(scenario, split)]


@pytest.mark.parametrize(("scenario", "split"), FIXED)
def test_a_fixed_pelvis_file_is_its_floating_file_with_base_fixed(scenario, split):
    floating = _load(scenario, split)
    fixed = _load(scenario, split, make.FIXED_BASE)
    assert fixed["base"] == "fixed" and floating["base"] == "floating"
    assert {k: v for k, v in fixed.items() if k != "base"} == {
        k: v for k, v in floating.items() if k != "base"
    }
    text = (CONFIG_DIR / make.file_name(scenario, split, make.FIXED_BASE)).read_text("utf-8")
    header = [line for line in text.splitlines() if line.startswith("#")]
    assert "fixed pelvis" in header[0] and "start the run with --base fixed" in header[-1]


def test_only_the_section_1_scenarios_get_a_fixed_pelvis_file():
    assert set(FIXED_BASE_SCENARIOS) == {"S2", "S3"}
    fixed_files = {p.name for p in CONFIG_DIR.glob("*_fixed.yaml")}
    assert fixed_files == {make.file_name(sc, sp, "fixed") for sc, sp in FIXED}
    for scenario in sorted(set(SCENARIOS) - set(FIXED_BASE_SCENARIOS)):
        with pytest.raises(ValueError, match="no fixed base"):
            make.build_config(scenario, "seen", "fixed")
    with pytest.raises(ValueError, match="base 'wheeled'"):
        make.file_name("S2", "seen", "wheeled")


@pytest.mark.parametrize(("scenario", "split", "base"), ALL)
def test_seeds_are_first_last_count_of_the_split(scenario, split, base):
    seeds = default_seeds(scenario, split)
    cfg = _load(scenario, split, base)
    assert cfg["seeds"] == {"first": seeds[0], "last": seeds[-1], "count": len(seeds)}
    assert seeds == list(range(cfg["seeds"]["first"], cfg["seeds"]["last"] + 1))
    assert cfg["num_envs"] == cfg["seeds"]["count"]
    if split == "unseen":
        assert (cfg["seeds"]["first"], cfg["seeds"]["last"]) == (1000, 1049)


@pytest.mark.parametrize(("scenario", "split", "base"), ALL)
def test_randomization_block_mirrors_the_ranges_of_the_split(scenario, split, base):
    block = _load(scenario, split, base)["randomization"]
    ranges = RandomizationRanges.for_split(split)
    assert block == ranges.to_dict()
    assert RandomizationRanges.from_dict(block) == ranges
    assert block["split"] == split


def test_seen_and_unseen_files_differ_only_in_the_unseen_only_rows():
    for scenario in SCENARIOS:
        seen = _load(scenario, "seen")["randomization"]
        unseen = _load(scenario, "unseen")["randomization"]
        differing = {k for k in seen if seen[k] != unseen[k]}
        assert differing == {
            "split",
            "plate_box_shift_toward_robot_m",
            "plate_diameters_m",
            "rack_slots",
            "push_force_range_n",
        }


def test_generation_is_deterministic(tmp_path):
    first = {p.name: p.read_bytes() for p in make.generate(tmp_path / "a")}
    second = {p.name: p.read_bytes() for p in make.generate(tmp_path / "b")}
    assert first == second
    assert first == {n: (CONFIG_DIR / n).read_bytes() for n in make.all_file_names()}


# ------------------------------------------------------------------ the --check mode


def test_check_passes_on_the_committed_files(capsys):
    assert make.main(["--check"]) == 0
    assert "ok: 14 files" in capsys.readouterr().out


def test_check_fails_on_a_changed_file_and_shows_the_diff(tmp_path, capsys):
    make.generate(tmp_path)
    assert make.main(["--check", "--out-dir", str(tmp_path)]) == 0
    target = tmp_path / "s4_unseen.yaml"
    target.write_text(target.read_text("utf-8").replace("[5]", "[4, 5]"), encoding="utf-8")
    assert make.main(["--check", "--out-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "s4_unseen.yaml: differs" in err and "-  rack_slots: [4, 5]" in err
    assert "+  rack_slots: [5]" in err and "1 of 14 files differ" in err


def test_check_fails_on_a_hand_edited_fixed_file(tmp_path, capsys):
    make.generate(tmp_path)
    target = tmp_path / "s3_seen_fixed.yaml"
    target.write_text(target.read_text("utf-8").replace("base: fixed", "base: floating"), "utf-8")
    assert make.main(["--check", "--out-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "s3_seen_fixed.yaml: differs" in err and "+base: fixed" in err


def test_check_fails_on_a_missing_file_and_on_crlf(tmp_path, capsys):
    make.generate(tmp_path)
    (tmp_path / "s0_seen.yaml").unlink()
    (tmp_path / "s2_unseen_fixed.yaml").unlink()
    crlf = tmp_path / "s1_seen.yaml"
    crlf.write_bytes(crlf.read_bytes().replace(b"\n", b"\r\n"))
    assert make.main(["--check", "--out-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "s0_seen.yaml: missing" in err and "s1_seen.yaml: differs" in err
    assert "s2_unseen_fixed.yaml: missing" in err
    assert "3 of 14 files differ" in err


def test_generate_writes_fourteen_files_into_the_given_directory(tmp_path, capsys):
    assert make.main(["--out-dir", str(tmp_path / "nested")]) == 0
    assert len(list((tmp_path / "nested").glob("*.yaml"))) == 14
    assert capsys.readouterr().out.count("wrote ") == 14
