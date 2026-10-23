"""Placeholder that proves the ``sim`` marker plumbing works.

It is skipped by ``pytest -q`` (see ``conftest.py``) and collected and run by
``pytest --run-sim``. It does not import Isaac Sim; replace it once a real sim test exists.
"""

import pytest


@pytest.mark.sim
def test_sim_marker_placeholder():
    assert True
