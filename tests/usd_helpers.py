"""Shared helpers for the procedural USD tests (issue #47). No simulator.

``pxr`` (the ``usd-core`` wheel) is optional: tests that read or write USD carry ``requires_usd``
and skip when it is missing, the rest of the suite stays green.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

requires_usd = pytest.mark.skipif(
    importlib.util.find_spec("pxr") is None, reason="usd-core not installed"
)

FORBIDDEN_IMPORTS = {"isaaclab", "isaacsim", "omni", "torch"}


def load_script(name: str):
    """Import ``scripts/<name>.py`` as a module (the scripts directory is not a package)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def imported_top_level_modules(path: Path) -> set[str]:
    """Top-level names of everything ``path`` imports, anywhere in the file."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def lf_bytes(path: Path) -> bytes:
    """File content with CRLF folded to LF, so a Windows checkout compares equal."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def run_script(name: str, *args) -> subprocess.CompletedProcess:
    """Run ``scripts/<name>.py`` in a fresh interpreter, capturing text output."""
    return subprocess.run(
        [sys.executable, str(SCRIPTS / f"{name}.py"), *map(str, args)],
        capture_output=True,
        text=True,
    )


@contextlib.contextmanager
def edit_stage(path):
    """Open a USD file, yield the stage for tampering, and save it back."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    yield stage
    stage.GetRootLayer().Save()
    del stage


def nudge(stage, prim_path: str, dx=0.0, dy=0.0, dz=0.0) -> None:
    """Move a prim by editing its first (translate) transform op."""
    from pxr import Gf, UsdGeom

    op = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).GetOrderedXformOps()[0]
    old = op.Get()
    op.Set(Gf.Vec3d(old[0] + dx, old[1] + dy, old[2] + dz))
