"""Pytest configuration.

Tests marked ``sim`` need Isaac Sim and a GPU. They are skipped unless ``--run-sim`` is given, so
CI and quick local runs stay under a few seconds.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-sim",
        action="store_true",
        default=False,
        help="run tests marked 'sim' (needs Isaac Sim and a GPU)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-sim"):
        return
    skip = pytest.mark.skip(reason="needs --run-sim")
    for item in items:
        if "sim" in item.keywords:
            item.add_marker(skip)
