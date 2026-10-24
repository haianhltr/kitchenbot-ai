"""Evaluator runner: scenario registry and the result document (docs/evaluation.md section 4).

``run_evaluation(config)`` runs every seed of one scenario once and returns the result document
that ``schema.json`` describes. ``scripts/evaluate.py`` is the command line in front of it.

Nothing here imports Isaac Sim, Isaac Lab or torch. A real scenario registers a *loader*: a
callable that takes the :class:`EvalConfig` and returns a :class:`ScenarioBackend`. The loader
body imports the environment config, so the simulator is only imported when that scenario is
actually run::

    def _load_s0(config: EvalConfig) -> ScenarioBackend:
        from kitchenbot.locomotion.eval_s0 import S0Backend  # imports isaaclab here, not above

        return S0Backend(config)

    register_scenario(ScenarioSpec(name="S0", title="Stand", loader=_load_s0), replace=True)

S0 to S4 currently register placeholder loaders that raise ``NotImplementedError`` naming the
issue that creates the environment config. ``fake`` and ``dry`` are synthetic scenarios for CI;
``dry`` (#60, :mod:`kitchenbot.evaluation.dry_run`) is the template of a real backend: the
sampler, the planner and the judges chained, with test doubles where the simulation will be.
"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import kitchenbot
from kitchenbot.evaluation.failure_tags import FailureTag
from kitchenbot.evaluation.metrics import SCENARIO_NAMES, metrics_for
from kitchenbot.evaluation.seeds import SPLITS, default_seeds, split_seeds
from kitchenbot.evaluation.stats import aggregate_episodes

#: Version of the result document layout, ``schema_version`` in the JSON. Must equal the ``const``
#: of ``schema_version`` in ``schema.json`` (a test checks it). 2: the optional ``run.config_path``
#: and ``run.config_sha256`` (#52).
SCHEMA_VERSION = 2

POLICIES: tuple[str, ...] = ("own", "agile")
HANDS: tuple[str, ...] = ("kinematic", "dex3", "dex1")
BASES: tuple[str, ...] = ("fixed", "floating")

#: Scenarios that docs/evaluation.md section 1 runs with a fixed pelvis first, then with the
#: balance policy ("fixed pelvis, then balance policy"). Besides the default floating-base file,
#: each has a ``base: fixed`` config per split, ``configs/eval/s<N>_<split>_fixed.yaml``
#: (#78); ``tests/test_contracts.py`` pins the set to the section 1 table.
FIXED_BASE_SCENARIOS: tuple[str, ...] = ("S2", "S3")

#: ``num_envs`` of a run that names neither ``--num_envs`` nor a ``--config`` YAML with a
#: ``num_envs`` (docs/evaluation.md section 4, step 1: ``--num_envs 50``).
DEFAULT_NUM_ENVS = 50

#: Default output directory (git-ignored, AGENTS.md section 6).
DEFAULT_OUTPUT_DIR = Path("outputs") / "eval"


@dataclass(frozen=True)
class EvalConfig:
    """One evaluator invocation. Validated on construction.

    ``split`` defaults to ``seen``: the held-out ``unseen`` split (report only, never tune on
    it) has to be asked for. ``seeds`` defaults to
    :func:`kitchenbot.evaluation.seeds.default_seeds` when left empty.
    The defaults for ``policy``, ``hand`` and ``base`` are the headline configuration: our own
    balance policy, the Dex3 hand, a floating base.

    ``config_path`` optionally names the committed ``configs/eval/*.yaml`` the run was started
    from (AGENTS.md section 6: a run without a committed config did not happen). The file must
    exist; ``run_evaluation`` records its path and SHA256 in ``run.config_path`` and
    ``run.config_sha256``. It is not part of :meth:`to_dict`, so ``run.config_hash`` is the same
    with and without it. The runner does not read the file's contents; a scenario's loader may,
    and the command line takes the file's ``num_envs`` when ``--num_envs`` is not given (#78).

    ``extras`` are options only one scenario's loader reads (``dry``: ``reach_gate`` and
    ``gate_envelope``, see :mod:`kitchenbot.evaluation.dry_run`). They are not part of
    :meth:`to_dict`, so they are not in the ``run`` block and the schema is untouched; a loader
    that uses one puts it in its ``resolved_config()``, which is how it reaches
    ``run.config_hash``. The command line fills
    them with ``--extra KEY=VALUE``; ``run_evaluation`` rejects keys the scenario does not list in
    ``ScenarioSpec.extra_keys``.
    """

    scenario: str
    split: str = "seen"
    seeds: tuple[int, ...] = ()
    num_envs: int = DEFAULT_NUM_ENVS
    policy: str = "own"
    hand: str = "dex3"
    base: str = "floating"
    headless: bool = False
    config_path: Path | None = None
    extras: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        for field_name, allowed in (
            ("scenario", SCENARIO_NAMES),
            ("split", SPLITS),
            ("policy", POLICIES),
            ("hand", HANDS),
            ("base", BASES),
        ):
            value = getattr(self, field_name)
            if value not in allowed:
                raise ValueError(f"{field_name} {value!r} not in {allowed}")
        if self.num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        seeds = tuple(self.seeds) or tuple(default_seeds(self.scenario, self.split))
        if len(set(seeds)) != len(seeds):
            raise ValueError("duplicate seeds")
        outside = sorted(set(seeds) - set(split_seeds(self.split)))
        if outside:
            raise ValueError(f"seeds {outside[:5]} are not in split {self.split!r}")
        object.__setattr__(self, "seeds", seeds)
        if self.config_path is not None:
            path = Path(self.config_path)
            if not path.is_file():
                raise ValueError(f"config_path {str(self.config_path)!r} is not a file")
            object.__setattr__(self, "config_path", path)
        if not all(isinstance(key, str) for key in self.extras):
            raise ValueError("extras keys must be strings")
        object.__setattr__(self, "extras", dict(self.extras))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "split": self.split,
            "seeds": list(self.seeds),
            "num_envs": self.num_envs,
            "policy": self.policy,
            "hand": self.hand,
            "base": self.base,
            "headless": self.headless,
        }


@dataclass(frozen=True)
class EpisodeResult:
    """What a backend reports for one seed. ``metrics`` keys must equal ``metrics_for(scenario)``.

    ``duration_s`` is simulated episode time in seconds. ``failure_tag`` is ``FailureTag.SUCCESS``
    exactly when ``success`` is true. ``traces`` is optional: ``{"sample_rate_hz": 10.0,
    "torso_orientation_error_deg": [...]}`` as described by ``schema.json``.
    """

    seed: int
    success: bool
    failure_tag: FailureTag
    duration_s: float
    metrics: Mapping[str, Any] = field(default_factory=dict)
    traces: Mapping[str, Any] | None = None


class ScenarioConfigError(ValueError):
    """A scenario's loader rejected the run's configuration (a YAML it read, an ``extras`` key).

    The command line reports it as a usage error (exit status 2) instead of a traceback.
    """


class ScenarioBackend(Protocol):
    """Runs episodes of one scenario. Created by the scenario's loader."""

    def resolved_config(self) -> Mapping[str, Any]:
        """The scenario's resolved environment configuration, JSON-serialisable.

        It is hashed into ``run.config_hash`` together with the evaluator options.
        """
        ...

    def run_batch(self, seeds: Sequence[int]) -> Sequence[EpisodeResult]:
        """Run one episode per seed (at most ``num_envs`` seeds), return results in seed order."""
        ...

    def close(self) -> None:
        """Release the simulator. Called once, even when ``run_batch`` raised."""
        ...


Loader = Callable[[EvalConfig], ScenarioBackend]


@dataclass(frozen=True)
class ScenarioSpec:
    """A registered scenario. ``uses_gpu`` false records ``gpu_name`` as ``"none"``.

    ``extra_keys`` are the :attr:`EvalConfig.extras` keys the scenario's loader reads;
    ``run_evaluation`` rejects any other key, so an option typed on the command line
    (``--extra``) never silently does nothing.
    """

    name: str
    title: str
    loader: Loader
    uses_gpu: bool = True
    synthetic: bool = False
    extra_keys: tuple[str, ...] = ()


_REGISTRY: dict[str, ScenarioSpec] = {}


def register_scenario(spec: ScenarioSpec, *, replace: bool = False) -> None:
    """Add ``spec`` to the registry. Re-registering a name needs ``replace=True``."""
    if spec.name not in SCENARIO_NAMES:
        raise ValueError(
            f"unknown scenario {spec.name!r}; add it to metrics.py and schema.json first"
        )
    if spec.name in _REGISTRY and not replace:
        raise ValueError(f"scenario {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec


def get_scenario(name: str) -> ScenarioSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown scenario {name!r}; registered: {scenario_names()}") from None


def scenario_names() -> list[str]:
    """Registered scenario names in the order S0 to S4, fake, dry."""
    return [n for n in SCENARIO_NAMES if n in _REGISTRY]


def _placeholder_loader(name: str, issue: int, what: str) -> Loader:
    def load(config: EvalConfig) -> ScenarioBackend:
        raise NotImplementedError(
            f"scenario {name} cannot run yet: its environment config does not exist. "
            f"Issue #{issue} creates it ({what}); that issue replaces this loader."
        )

    return load


def _load_fake(config: EvalConfig) -> ScenarioBackend:
    from kitchenbot.evaluation.fake import FakeBackend

    return FakeBackend(config)


def _load_dry(config: EvalConfig) -> ScenarioBackend:
    from kitchenbot.evaluation.dry_run import DryRunBackend

    return DryRunBackend(config)


def _register_builtin() -> None:
    placeholders = (
        ("S0", "Stand", 5, "stand environment config"),
        ("S1", "Stand and reach", 14, "arm IK environment, fixed base first, balance in #15"),
        ("S2", "Grasp and lift", 17, "kinematic-follow hand and the S2 fixed-base environment"),
        ("S3", "Place in rack", 21, "plate-to-rack state machine and S3 fixed-base environment"),
        ("S4", "Full task", 22, "full S4 environment with success detection"),
    )
    for name, title, issue, what in placeholders:
        register_scenario(
            ScenarioSpec(name=name, title=title, loader=_placeholder_loader(name, issue, what))
        )
    register_scenario(
        ScenarioSpec(
            name="fake",
            title="Synthetic scenario for CI (no simulation)",
            loader=_load_fake,
            uses_gpu=False,
            synthetic=True,
        )
    )
    register_scenario(
        ScenarioSpec(
            name="dry",
            title="Dry run: sampler, planner against fakes and judges (no simulation)",
            loader=_load_dry,
            uses_gpu=False,
            synthetic=True,
            extra_keys=("reach_gate", "gate_envelope"),  # dry_run.EXTRA_KEYS; a test checks
        )
    )


_register_builtin()


# ----------------------------------------------------------------------------------------------
# Run metadata


def _git(*args: str) -> str | None:
    """Output of ``git <args>`` run in this checkout, or ``None`` without git or a repository."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def git_info() -> tuple[str, bool | None]:
    """``(commit, dirty)``: the checkout's HEAD hash and whether the tree has uncommitted changes.

    Returns ``("unknown", None)`` outside a git checkout.
    """
    head = _git("rev-parse", "HEAD")
    if head is None or len(head.strip()) != 40:
        return "unknown", None
    status = _git("status", "--porcelain")
    return head.strip(), (None if status is None else bool(status.strip()))


def detect_gpu_name() -> str:
    """Name of the first NVIDIA GPU from ``nvidia-smi``, or ``"none"`` if there is none."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "none"
    names = [line.strip() for line in out.splitlines() if line.strip()]
    return names[0] if names else "none"


def file_sha256(path: Path | str) -> str:
    """SHA256 (hex) of the bytes of ``path``. The repository stores text as LF (.gitattributes)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_hash(options: Mapping[str, Any], scenario_config: Mapping[str, Any]) -> str:
    """SHA256 (hex) of the canonical JSON of the evaluator options plus the scenario config."""
    blob = json.dumps(
        {"options": options, "scenario": scenario_config},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------------
# Running


def _episode_record(result: EpisodeResult, scenario: str) -> dict[str, Any]:
    expected = set(metrics_for(scenario))
    if set(result.metrics) != expected:
        raise ValueError(
            f"seed {result.seed}: metrics {sorted(result.metrics)} != "
            f"{sorted(expected)} required for scenario {scenario}"
        )
    tag = FailureTag(result.failure_tag)
    if result.success != (tag is FailureTag.SUCCESS):
        raise ValueError(f"seed {result.seed}: success={result.success} but failure_tag={tag}")
    record: dict[str, Any] = {
        "seed": int(result.seed),
        "success": bool(result.success),
        "failure_tag": tag.value,
        "duration_s": round(float(result.duration_s), 4),
        "metrics": dict(result.metrics),
    }
    if result.traces is not None:
        record["traces"] = dict(result.traces)
    return record


def run_evaluation(config: EvalConfig, *, validate: bool = True) -> dict[str, Any]:
    """Run ``config.seeds`` once each, in batches of ``config.num_envs``, return the document.

    The document is validated against ``schema.json`` (and checked for internal consistency)
    before it is returned, unless ``validate`` is false. Raises ``NotImplementedError`` for a
    scenario whose loader is still a placeholder.
    """
    spec = get_scenario(config.scenario)
    unknown = sorted(set(config.extras) - set(spec.extra_keys))
    if unknown:
        takes = f"reads {list(spec.extra_keys)}" if spec.extra_keys else "takes no extras"
        raise ScenarioConfigError(f"unknown extras {unknown}: scenario {spec.name} {takes}")
    config_file: dict[str, str] = {}
    if config.config_path is not None:
        # Hashed before the backend loads, so what is recorded is what the run started with.
        config_file = {
            "config_path": config.config_path.as_posix(),
            "config_sha256": file_sha256(config.config_path),
        }
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    backend = spec.loader(config)
    try:
        scenario_config = dict(backend.resolved_config())
        results: list[EpisodeResult] = []
        for i in range(0, len(config.seeds), config.num_envs):
            batch = config.seeds[i : i + config.num_envs]
            out = list(backend.run_batch(batch))
            if [r.seed for r in out] != list(batch):
                raise RuntimeError(f"backend returned seeds {[r.seed for r in out]} for {batch}")
            results.extend(out)
    finally:
        backend.close()
    wall_clock_s = time.perf_counter() - t0

    episodes = [_episode_record(r, config.scenario) for r in results]
    options = config.to_dict()
    commit, dirty = git_info()
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            **options,
            "synthetic": spec.synthetic,
            "git_commit": commit,
            "git_dirty": dirty,
            "config_hash": config_hash(options, scenario_config),
            **config_file,
            "gpu_name": detect_gpu_name() if spec.uses_gpu else "none",
            "host": socket.gethostname(),
            "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wall_clock_s": round(wall_clock_s, 3),
            "package_version": kitchenbot.__version__,
        },
        "episodes": episodes,
        "aggregates": aggregate_episodes(config.scenario, episodes),
    }
    if validate:
        from kitchenbot.evaluation.schema import validate_result

        validate_result(doc)
    return doc


def default_output_path(doc: Mapping[str, Any], out_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """``<out_dir>/<UTC timestamp>_<short commit>_<scenario>_<split>.json``.

    docs/evaluation.md section 4 names ``<date>_<commit>_<scenario>_<split>``. The date carries
    the time of day so two runs of one scenario on one day (own policy and Agile, say) do not
    overwrite each other. If the file already exists (two runs within one second) ``-2``, ``-3``
    and so on is appended to the stem: a result is never overwritten.
    """
    run = doc["run"]
    stamp = run["started_at"].replace("-", "").replace(":", "")
    commit = run["git_commit"][:7] if run["git_commit"] != "unknown" else "nogit"
    stem = f"{stamp}_{commit}_{run['scenario']}_{run['split']}"
    path = Path(out_dir) / f"{stem}.json"
    n = 2
    while path.exists():
        path = Path(out_dir) / f"{stem}-{n}.json"
        n += 1
    return path


def _dumps(obj: Any, level: int = 0) -> str:
    """``json.dumps(indent=2)`` except that a list of numbers stays on one line.

    Seed lists and per-step traces would otherwise take one line per number.
    """
    pad, end = "  " * (level + 1), "  " * level
    if isinstance(obj, Mapping):
        if not obj:
            return "{}"
        items = [f"{pad}{json.dumps(k)}: {_dumps(v, level + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + end + "}"
    if isinstance(obj, list | tuple):
        if not obj:
            return "[]"
        if all(isinstance(x, int | float) and not isinstance(x, bool) for x in obj):
            return "[" + ", ".join(json.dumps(x, allow_nan=False) for x in obj) + "]"
        return "[\n" + ",\n".join(pad + _dumps(v, level + 1) for v in obj) + "\n" + end + "]"
    return json.dumps(obj, allow_nan=False)


def write_result(doc: Mapping[str, Any], path: Path) -> Path:
    """Write ``doc`` as strict JSON (no NaN), creating parent directories.

    Lists of numbers (``run.seeds``, ``episodes[].traces``) are written on one line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _dumps(doc) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
