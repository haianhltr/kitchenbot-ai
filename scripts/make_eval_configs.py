"""Write the evaluation configs ``configs/eval/*.yaml`` (#52, #78, no simulator).

Fourteen files: the ten ``s<N>_<split>.yaml``, one per scenario (S0 to S4) and split, with the
evaluator's default floating base, and the four fixed-pelvis variants
``s2_seen_fixed.yaml``, ``s2_unseen_fixed.yaml``, ``s3_seen_fixed.yaml`` and
``s3_unseen_fixed.yaml`` (#78): docs/evaluation.md section 1 runs S2 and S3 with a "fixed pelvis,
then balance policy" (``runner.FIXED_BASE_SCENARIOS``), section 6 reports fixed pelvis against the
balance policy as a baseline, and AGENTS.md section 6 says a run without a committed config did
not happen. A fixed file is its floating file with ``base: fixed`` and a different header: same
seeds, ``num_envs``, policy, hand and ranges.

Generate (deterministic; LF line endings; a fixed key order)::

    py -3.12 scripts/make_eval_configs.py
    py -3.12 scripts/make_eval_configs.py --out-dir some/other/dir

Check that the committed files are exactly what the generator writes (regenerates into a
temporary directory and diffs; exit code 1 on any difference or missing file)::

    py -3.12 scripts/make_eval_configs.py --check
    py -3.12 scripts/make_eval_configs.py --check --out-dir some/other/dir

Keys, in this order:

``scenario``, ``split``
    as for ``scripts/evaluate.py``.
``physics``
    ``isaacsim_physx`` (``kitchenbot.PHYSICS_PRESET``; ``tests/test_package.py`` checks it).
``num_envs``
    the trial count of docs/evaluation.md section 4 step 2: 100 for S0 to S3 seen, 50 for every
    unseen split and for S4 (``kitchenbot.evaluation.seeds.default_seeds``).
``seeds``
    ``first``, ``last`` and ``count`` of those seeds (they are consecutive).
``policy``, ``hand``, ``base``
    the defaults of ``scripts/evaluate.py`` (``EvalConfig``), except ``base: fixed`` in the four
    fixed-pelvis files. The evaluator reads only ``num_envs`` from a file (#78), so a fixed-pelvis
    run is started with ``--base fixed --config configs/eval/s2_seen_fixed.yaml``.
``randomization``
    ``RandomizationRanges.for_split(split).to_dict()``: every row of docs/evaluation.md section 2.
    ``kitchenbot.scene.randomization.sample_episode`` draws each episode from it.

The files are the committed config of AGENTS.md section 6; ``scripts/evaluate.py --config`` records
a file's SHA256 in the result. Standard library, numpy and pyyaml only; nothing here imports
Isaac Sim or Isaac Lab.
"""

from __future__ import annotations

import argparse
import difflib
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

import kitchenbot
from kitchenbot.evaluation.runner import BASES, FIXED_BASE_SCENARIOS, EvalConfig
from kitchenbot.evaluation.seeds import SPLITS
from kitchenbot.scene.randomization import SCENARIOS, RandomizationRanges

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "configs" / "eval"

#: The evaluator's default base (floating, the balance policy active): the ten unsuffixed files.
DEFAULT_BASE = EvalConfig(scenario=SCENARIOS[0]).base
#: The base of the fixed-pelvis variants, written as ``s<N>_<split>_fixed.yaml``.
FIXED_BASE = "fixed"

_TITLES = {
    "S0": "S0 stand",
    "S1": "S1 stand and reach",
    "S2": "S2 grasp and lift",
    "S3": "S3 place in rack",
    "S4": "S4 full task",
}
_SPLIT_NOTES = {
    "seen": "seen split, development seeds",
    "unseen": "unseen split, held out: report only, never tune on these seeds",
}


def file_name(scenario: str, split: str, base: str = DEFAULT_BASE) -> str:
    """``s4_unseen.yaml`` for ``("S4", "unseen")``; ``s2_seen_fixed.yaml`` for a fixed base."""
    if base not in BASES:
        raise ValueError(f"base {base!r} not in {BASES}")
    suffix = "" if base == DEFAULT_BASE else f"_{base}"
    return f"{scenario.lower()}_{split}{suffix}.yaml"


def variants() -> list[tuple[str, str, str]]:
    """``(scenario, split, base)`` of every file, in file order.

    Scenario-major, then split; within one split the default-base file, then the fixed-pelvis one
    for the scenarios of ``FIXED_BASE_SCENARIOS``.
    """
    out = []
    for scenario in SCENARIOS:
        for split in SPLITS:
            out.append((scenario, split, DEFAULT_BASE))
            if scenario in FIXED_BASE_SCENARIOS:
                out.append((scenario, split, FIXED_BASE))
    return out


def all_file_names() -> list[str]:
    """The fourteen file names: ``s0_seen.yaml``, ``s0_unseen.yaml``, ..., ``s2_seen.yaml``,
    ``s2_seen_fixed.yaml``, ``s2_unseen.yaml``, ``s2_unseen_fixed.yaml``, ...
    """
    return [file_name(*variant) for variant in variants()]


def build_config(scenario: str, split: str, base: str = DEFAULT_BASE) -> dict[str, Any]:
    """The document of one file, as a dict in the file's key order.

    ``base`` other than the default is allowed for ``FIXED_BASE_SCENARIOS`` only.
    """
    if base != DEFAULT_BASE and scenario not in FIXED_BASE_SCENARIOS:
        raise ValueError(
            f"{scenario}: docs/evaluation.md section 1 has no {base} base for it; only "
            f"{FIXED_BASE_SCENARIOS} get a {FIXED_BASE} file"
        )
    # The defaults of scripts/evaluate.py, with the base of this file.
    defaults = EvalConfig(scenario=scenario, split=split, base=base)
    seeds = list(defaults.seeds)
    if seeds != list(range(seeds[0], seeds[-1] + 1)):
        raise ValueError(f"{scenario} {split}: the seeds are not consecutive")
    return {
        "scenario": scenario,
        "split": split,
        "physics": kitchenbot.PHYSICS_PRESET,
        "num_envs": len(seeds),
        "seeds": {"first": seeds[0], "last": seeds[-1], "count": len(seeds)},
        "policy": defaults.policy,
        "hand": defaults.hand,
        "base": defaults.base,
        "randomization": RandomizationRanges.for_split(split).to_dict(),
    }


def render(scenario: str, split: str, base: str = DEFAULT_BASE) -> str:
    """The exact text of one file: a comment header, then the YAML.

    The header has four lines; a fixed-pelvis file names its base in the first line and has two
    more saying how to start the run. The ten default-base files are unchanged by #78.
    """
    base_note = "" if base == DEFAULT_BASE else ", fixed pelvis"
    header = (
        f"# {_TITLES[scenario]}{base_note}, {_SPLIT_NOTES[split]}.\n"
        "# Generated by scripts/make_eval_configs.py from kitchenbot.scene.randomization; do not\n"
        "# edit by hand (`make_eval_configs.py --check` fails on any difference). Ranges:\n"
        "# docs/evaluation.md section 2; seeds: kitchenbot.evaluation.seeds.\n"
    )
    if base != DEFAULT_BASE:
        header += (
            "# Fixed-pelvis run of docs/evaluation.md section 1. "
            "The evaluator reads only num_envs\n"
            f"# from this file: start the run with --base {base}.\n"
        )
    body = yaml.safe_dump(
        build_config(scenario, split, base), sort_keys=False, default_flow_style=None, width=100
    )
    return header + body


def generate(out_dir: Path) -> list[Path]:
    """Write every file of :func:`variants` into ``out_dir`` (created if needed); return paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for scenario, split, base in variants():
        path = out_dir / file_name(scenario, split, base)
        path.write_text(render(scenario, split, base), encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


def check(out_dir: Path, *, max_diff_lines: int = 40) -> list[str]:
    """Regenerate into a temporary directory and compare bytes with ``out_dir``.

    Returns one message per problem (missing file, or the unified diff of a changed one, cut at
    ``max_diff_lines`` lines); an empty list means the committed files are exactly the output.
    """
    out_dir = Path(out_dir)
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for fresh in generate(Path(tmp)):
            committed = out_dir / fresh.name
            if not committed.is_file():
                problems.append(f"{committed.as_posix()}: missing")
                continue
            if committed.read_bytes() == fresh.read_bytes():
                continue
            diff = list(
                difflib.unified_diff(
                    committed.read_text(encoding="utf-8").splitlines(),
                    fresh.read_text(encoding="utf-8").splitlines(),
                    fromfile=f"{committed.as_posix()} (committed)",
                    tofile=f"{fresh.name} (generated)",
                    lineterm="",
                )
            )
            if not diff:  # same lines, different bytes: line endings
                diff = ["same text, different bytes (line endings?)"]
            shown = diff[:max_diff_lines]
            if len(diff) > max_diff_lines:
                shown.append(f"... {len(diff) - max_diff_lines} more diff lines")
            problems.append("\n".join([f"{committed.as_posix()}: differs", *shown]))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or check configs/eval/*.yaml (issues #52, #78): the randomization "
        "ranges of docs/evaluation.md section 2 and the seeds, one file per scenario and split, "
        "plus a fixed-pelvis file per split for S2 and S3."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory to write or check (default configs/eval)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; regenerate into a temporary directory and diff against --out-dir",
    )
    args = parser.parse_args(argv)

    if args.check:
        problems = check(args.out_dir)
        if problems:
            print("\n\n".join(problems), file=sys.stderr)
            print(
                f"{len(problems)} of {len(all_file_names())} files differ; run "
                "scripts/make_eval_configs.py to regenerate",
                file=sys.stderr,
            )
            return 1
        print(f"ok: {len(all_file_names())} files in {args.out_dir.as_posix()} match the generator")
        return 0

    for path in generate(args.out_dir):
        print(f"wrote {path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
