"""Seed splits (docs/evaluation.md sections 1 and 4).

``seen``: seeds 0 to 99, randomization ranges of section 2.
``unseen``: held-out seeds 1000 to 1049, plus the held-out plate diameter and rack slot (those
are scenario configuration chosen by the split, not by the seed).

Trial counts (section 4, step 2): 100 seeds for S0 to S3 seen, 50 for every unseen split and
for S4. :func:`default_seeds` applies that rule, so S4 seen defaults to seeds 0 to 49.
"""

from __future__ import annotations

SPLITS: tuple[str, ...] = ("seen", "unseen")

SEEN_SEEDS: tuple[int, ...] = tuple(range(0, 100))
UNSEEN_SEEDS: tuple[int, ...] = tuple(range(1000, 1050))

_SPLIT_SEEDS = {"seen": SEEN_SEEDS, "unseen": UNSEEN_SEEDS}

#: Scenarios that use 50 trials on both splits (section 4, step 2). ``dry`` mirrors S4 (#60).
_FIFTY_TRIAL_SCENARIOS = frozenset({"S4", "dry"})


def split_seeds(split: str) -> list[int]:
    """Return the full seed list of ``split`` (``"seen"`` or ``"unseen"``) as a new list."""
    try:
        return list(_SPLIT_SEEDS[split])
    except KeyError:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}") from None


def split_of_seed(seed: int) -> str | None:
    """Return the split that owns ``seed``, or ``None`` if it belongs to neither."""
    for name, seeds in _SPLIT_SEEDS.items():
        if seed in seeds:
            return name
    return None


def default_seeds(scenario: str, split: str) -> list[int]:
    """Seeds run when the caller gives none: the trial count of section 4 step 2.

    Unseen: all 50 held-out seeds. Seen: the first 100 seeds, except S4, which uses the first 50.
    """
    seeds = split_seeds(split)
    if split == "seen" and scenario in _FIFTY_TRIAL_SCENARIOS:
        return seeds[:50]
    return seeds


def parse_seeds(spec: str | None, scenario: str, split: str) -> list[int]:
    """Turn the ``--seeds`` argument into a list of seeds belonging to ``split``.

    Accepted forms:

    * omitted: :func:`default_seeds`;
    * a bare integer ``N``: a count, the first ``N`` seeds of the split;
    * a comma list of integers and inclusive ranges, ``1000-1049`` or ``0,5,10-12``. A single
      explicit seed is written as a one-element range such as ``7-7``.

    Raises ``ValueError`` for malformed input, duplicates, an empty selection, a count larger than
    the split, or any seed outside the chosen split (a result labelled ``unseen`` must not have
    been produced on seeds 0 to 99).
    """
    valid = split_seeds(split)
    if spec is None or not spec.strip():
        return default_seeds(scenario, split)
    text = spec.strip()
    if text.isdigit():
        count = int(text)
        if not 1 <= count <= len(valid):
            raise ValueError(f"seed count must be between 1 and {len(valid)} for split {split!r}")
        return valid[:count]
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part.isdigit():
            seeds.append(int(part))
        elif part.count("-") == 1 and all(p.strip().isdigit() for p in part.split("-")):
            lo, hi = (int(p) for p in part.split("-"))
            if hi < lo:
                raise ValueError(f"empty seed range {part!r}")
            if hi - lo + 1 > len(valid):
                raise ValueError(f"seed range {part!r} is larger than split {split!r}")
            seeds.extend(range(lo, hi + 1))
        else:
            raise ValueError(
                f"cannot parse seeds {spec!r}: use a count (50), a list (0,1,2) "
                "or ranges (1000-1049)"
            )
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in {spec!r}")
    outside = [s for s in seeds if s not in valid]
    if outside:
        shown = ", ".join(str(s) for s in outside[:5])
        raise ValueError(
            f"seeds not in split {split!r} ({valid[0]} to {valid[-1]}): {shown}"
            + (" ..." if len(outside) > 5 else "")
        )
    return seeds
