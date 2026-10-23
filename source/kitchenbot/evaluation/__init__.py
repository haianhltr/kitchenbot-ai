"""Evaluation infrastructure (docs/evaluation.md). Importing this package never imports a simulator.

Files, all created by P23 unless noted:

- ``runner.py``: scenario registry S0 to S4 plus ``fake``, ``run_evaluation`` and the result
  document, behind ``scripts/evaluate.py``.
- ``cli.py``: the argument parser and ``main(argv)`` of ``scripts/evaluate.py``.
- ``schema.json`` and ``schema.py``: the result file schema and ``validate_result``.
- ``seeds.py``: the seed splits (seen 0 to 99, unseen 1000 to 1049) and ``--seeds`` parsing.
- ``failure_tags.py``: the failure taxonomy enum and its tagging precedence.
- ``stats.py``: Wilson intervals and the aggregates of a run.
- ``metrics.py``: which per-episode metrics each scenario reports. Success predicates and rack
  collision tagging from simulation state join it in P22.
- ``fake.py``: the synthetic scenario CI runs instead of a simulation.

Created by P24 (``scripts/plot_results.py`` and ``scripts/update_experiments.py``):

- ``results.py``: reads and validates result JSONs, refuses synthetic ones unless told otherwise.
- ``plotting.py``: the three figures, one shared style, matplotlib only.
- ``experiments.py``: the rows of the "Evaluation results" table in ``docs/experiments.md``.
"""
