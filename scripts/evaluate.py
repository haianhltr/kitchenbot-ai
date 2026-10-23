"""Run an evaluation scenario and write the result JSON (docs/evaluation.md section 4).

    python scripts/evaluate.py --scenario S4 --split unseen --seeds 1000-1049 \
        --num_envs 50 --headless
    python scripts/evaluate.py --scenario fake        # synthetic, no simulation, runs on CI
    python scripts/evaluate.py --scenario dry --config configs/eval/s4_seen.yaml
        # synthetic: the sampler, the planner against fakes and the judges, no simulation

The logic lives in ``kitchenbot.evaluation.cli`` so tests and other scripts can import it. Run from
the repository root so ``outputs/eval/`` lands in the git-ignored output directory.
"""

import sys

from kitchenbot.evaluation.cli import main

if __name__ == "__main__":
    sys.exit(main())
