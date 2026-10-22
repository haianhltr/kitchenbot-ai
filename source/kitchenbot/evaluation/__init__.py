"""Evaluation infrastructure (docs/evaluation.md).

Planned files:

- ``runner.py``: scenario registry S0 to S4 and the trial loop behind ``scripts/evaluate.py``. P23.
- ``schema.json``: the result file schema. P23.
- ``splits.py``: seed to split mapping (seen 0 to 99, unseen 1000 to 1049). P13.
- ``failure_tags.py``: the failure taxonomy enum. P23.
- ``metrics.py``: success predicates, collision tagging, Wilson intervals. P22 and P23.
"""
