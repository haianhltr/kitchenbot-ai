"""configs/eval/*.yaml (#52): the ten files equal a fresh generation; the generator's --check."""

import pytest
import yaml
from eval_helpers import REPO, load_script

import kitchenbot
from kitchenbot.evaluation.runner import EvalConfig
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


def _load(scenario, split):
    return yaml.safe_load((CONFIG_DIR / make.file_name(scenario, split)).read_text("utf-8"))


# ------------------------------------------------------------------ the ten files


def test_there_are_exactly_ten_named_files():
    assert make.all_file_names() == [
        "s0_seen.yaml", "s0_unseen.yaml", "s1_seen.yaml", "s1_unseen.yaml", "s2_seen.yaml",
        "s2_unseen.yaml", "s3_seen.yaml", "s3_unseen.yaml", "s4_seen.yaml", "s4_unseen.yaml",
    ]  # fmt: skip
    assert set(make.all_file_names()) <= {p.name for p in CONFIG_DIR.glob("*.yaml")}


@pytest.mark.parametrize(("scenario", "split"), BOTH)
def test_committed_file_is_byte_identical_to_a_fresh_generation(scenario, split):
    committed = (CONFIG_DIR / make.file_name(scenario, split)).read_bytes()
    assert committed == make.render(scenario, split).encode("utf-8")
    assert b"\r" not in committed and committed.endswith(b"\n")


@pytest.mark.parametrize(("scenario", "split"), BOTH)
def test_each_file_names_physx_and_the_evaluator_defaults(scenario, split):
    cfg = _load(scenario, split)
    defaults = EvalConfig(scenario=scenario, split=split)
    assert list(cfg) == [
        "scenario", "split", "physics", "num_envs", "seeds", "policy", "hand", "base",
        "randomization",
    ]  # fmt: skip
    assert cfg["physics"] == "isaacsim_physx" == kitchenbot.PHYSICS_PRESET
    assert (cfg["scenario"], cfg["split"]) == (scenario, split)
    assert (cfg["policy"], cfg["hand"], cfg["base"]) == ("own", "dex3", "floating")
    assert (cfg["policy"], cfg["hand"], cfg["base"]) == (
        defaults.policy,
        defaults.hand,
        defaults.base,
    )
    assert cfg["num_envs"] == EXPECTED_NUM_ENVS[(scenario, split)]


@pytest.mark.parametrize(("scenario", "split"), BOTH)
def test_seeds_are_first_last_count_of_the_split(scenario, split):
    seeds = default_seeds(scenario, split)
    cfg = _load(scenario, split)
    assert cfg["seeds"] == {"first": seeds[0], "last": seeds[-1], "count": len(seeds)}
    assert seeds == list(range(cfg["seeds"]["first"], cfg["seeds"]["last"] + 1))
    assert cfg["num_envs"] == cfg["seeds"]["count"]
    if split == "unseen":
        assert (cfg["seeds"]["first"], cfg["seeds"]["last"]) == (1000, 1049)


@pytest.mark.parametrize(("scenario", "split"), BOTH)
def test_randomization_block_mirrors_the_ranges_of_the_split(scenario, split):
    block = _load(scenario, split)["randomization"]
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
    assert "ok: 10 files" in capsys.readouterr().out


def test_check_fails_on_a_changed_file_and_shows_the_diff(tmp_path, capsys):
    make.generate(tmp_path)
    assert make.main(["--check", "--out-dir", str(tmp_path)]) == 0
    target = tmp_path / "s4_unseen.yaml"
    target.write_text(target.read_text("utf-8").replace("[5]", "[4, 5]"), encoding="utf-8")
    assert make.main(["--check", "--out-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "s4_unseen.yaml: differs" in err and "-  rack_slots: [4, 5]" in err
    assert "+  rack_slots: [5]" in err and "1 of 10 files differ" in err


def test_check_fails_on_a_missing_file_and_on_crlf(tmp_path, capsys):
    make.generate(tmp_path)
    (tmp_path / "s0_seen.yaml").unlink()
    crlf = tmp_path / "s1_seen.yaml"
    crlf.write_bytes(crlf.read_bytes().replace(b"\n", b"\r\n"))
    assert make.main(["--check", "--out-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "s0_seen.yaml: missing" in err and "s1_seen.yaml: differs" in err
    assert "2 of 10 files differ" in err


def test_generate_writes_ten_files_into_the_given_directory(tmp_path, capsys):
    assert make.main(["--out-dir", str(tmp_path / "nested")]) == 0
    assert len(list((tmp_path / "nested").glob("*.yaml"))) == 10
    assert capsys.readouterr().out.count("wrote ") == 10
