"""Run an evaluation scenario and write the result JSON (docs/evaluation.md section 4).

    python scripts/evaluate.py --scenario S4 --split unseen --seeds 1000-1049 \
        --num_envs 50 --headless
    python scripts/evaluate.py --scenario fake        # synthetic, no simulation, runs on CI
    python scripts/evaluate.py --scenario dry --config configs/eval/s4_seen.yaml
        # synthetic: the sampler, the planner against fakes and the judges, no simulation
    python scripts/evaluate.py --scenario dry --config configs/eval/s4_seen.yaml \
        --extra reach_gate=false    # the same without the reachability gate
    python scripts/evaluate.py --scenario dry --config configs/eval/s4_seen.yaml \
        --extra gate_envelope=waist    # gated on the waist envelope (default waist0)

Without ``--num_envs`` a run started from ``--config`` takes the YAML's ``num_envs`` (the trial
count of its scenario and split); the flag overrides it, without either the default is 50, and the
result's ``run.num_envs`` records the value used (#78).

``--extra KEY=VALUE`` is repeatable and fills ``EvalConfig.extras``, the options one scenario
reads (the value is a YAML scalar: ``false``, ``0.3`` or text); today only ``dry`` reads them,
``reach_gate`` and ``gate_envelope``. A key the scenario does not read is a usage error.

The logic lives in ``kitchenbot.evaluation.cli`` so tests and other scripts can import it. Run from
the repository root so ``outputs/eval/`` lands in the git-ignored output directory.
"""

import sys

from kitchenbot.evaluation.cli import main

if __name__ == "__main__":
    sys.exit(main())
