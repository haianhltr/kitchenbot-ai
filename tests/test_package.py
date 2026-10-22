"""Smoke tests that run without Isaac Sim."""

from pathlib import Path

import yaml

import kitchenbot

REPO = Path(__file__).resolve().parents[1]


def test_version():
    assert kitchenbot.__version__ == "0.1.0"


def test_physics_preset_is_physx():
    assert kitchenbot.PHYSICS_PRESET == "isaacsim_physx"


def test_required_docs_exist():
    for name in [
        "architecture.md",
        "roadmap.md",
        "setup.md",
        "evaluation.md",
        "experiments.md",
        "decisions.md",
        "status.md",
    ]:
        assert (REPO / "docs" / name).is_file(), name


def test_every_config_uses_physx():
    """Any YAML under configs/ that names a physics preset must name isaacsim_physx (D003)."""
    offenders = []
    for path in (REPO / "configs").rglob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        preset = data.get("physics")
        if preset is not None and preset != kitchenbot.PHYSICS_PRESET:
            offenders.append(f"{path.relative_to(REPO)}: {preset}")
    assert not offenders, offenders
