# Evaluation protocol

This document defines how KitchenBot results are measured. **Every number in this file is a
target.** Measured results live in `docs/experiments.md`, each with date, commit, seed list,
`num_envs` and the exact command. A result that was not produced by `scripts/evaluate.py` does not
exist.

## 1. Scenarios

| ID | Name | Robot base | What happens | Milestone |
| --- | --- | --- | --- | --- |
| S0 | Stand | balance policy | G1 stands on flat floor, no task, random push every 3 s | M1 |
| S1 | Stand and reach | balance policy | A reach target appears in the front workspace; IK moves the hand to it | M3 |
| S2 | Grasp and lift | fixed pelvis, then balance policy | Plate on the counter at a random pose; hand grasps and lifts 15 cm | M4 |
| S3 | Place in rack | fixed pelvis, then balance policy | Plate held in hand; insert into a free rack slot | M5 |
| S4 | Full task | balance policy | S2 then S3 end to end from a random plate pose | M5, M6 |

Each scenario has two variants: **seen** (randomization ranges below, seeds 0 to 99) and
**unseen** (held-out seeds 1000 to 1049 plus the held-out plate diameter and the held-out rack slot).
Unseen results are the headline numbers.

## 2. Randomization ranges

| Quantity | Seen range | Unseen only |
| --- | --- | --- |
| Plate position on counter (x, y) | uniform within a 20 cm x 20 cm box | box shifted 5 cm toward the robot |
| Plate yaw | uniform in [-30 deg, 30 deg] | same |
| Plate diameter | {22 cm, 25 cm} | 27 cm |
| Rack slot used | slots 1 to 4 | slot 5 |
| Robot initial pelvis pose | position noise 2 cm, yaw noise 5 deg | same |
| Floor and object friction | uniform in [0.6, 1.0] | same |
| External push (S0 only) | 10 N to 30 N impulse at the torso, random horizontal direction, every 3 s | 40 N |

Ranges are frozen once M4 starts. Changing them resets `docs/experiments.md`.

## 3. Metrics

| Metric | Definition | Measured in |
| --- | --- | --- |
| Standing duration | seconds until termination (fall or timeout) | S0 |
| Fall rate | fraction of trials where pelvis height drops below 0.5 m or torso tilt exceeds 60 deg | all |
| Torso orientation error | RMS of torso roll and pitch over the trial, degrees | S0, S1, S4 |
| Hand target error | final hand-to-target distance, cm, and fraction under 3 cm | S1 |
| Grasp success rate | plate lifted 15 cm and held for 2 s without contact with the counter | S2, S4 |
| Placement success rate | plate inside the slot volume, within 15 deg of vertical, stable for 2 s after release | S3, S4 |
| Collision rate | fraction of trials with any robot-rack or plate-rack contact force above 20 N before release | S3, S4 |
| Completion time | seconds from episode start to stable placement, successes only | S4 |
| Task success rate | grasp and placement both succeed, no fall, within the 30 s timeout | S4 |

Success detection is automatic, from simulation state. No metric is read off a video.

## 4. Protocol

1. `scripts/evaluate.py --scenario S4 --split unseen --seeds 1000-1049 --num_envs 50 --headless`
   runs every seed once and writes `outputs/eval/<date>_<commit>_<scenario>_<split>.json`.
2. Trial counts: 100 seeds for S0 to S3 seen, 50 for every unseen split and for S4.
3. Every reported number carries a 95 percent Wilson confidence interval. With 50 trials the
   interval on a success rate is about plus or minus 12 points; say so next to the number.
4. Each metric is reported for every baseline in section 6 that applies.
5. `scripts/plot_results.py` turns the JSON files into the figures in `docs/experiments.md`.
6. A result is reproducible only if a fresh clone can run step 1 from the documented checkpoint
   URL in `docs/experiments.md` and land within the confidence interval.

Precedents for these conventions: BiGym evaluates its humanoid dishwasher-loading tasks as binary
success over 50 episodes; RoboCasa uses 50 trials per task across fixed scenes with unseen object
instances; HOMIE and AMO report fall rate and base orientation and height tracking error for the
lower-body policy. Our 50-trial unseen split and the S0 metrics follow them.

### Running the evaluator

```
python scripts/evaluate.py --scenario S4 --split unseen --seeds 1000-1049 --num_envs 50 --headless
```

From the repository root. `--seeds` takes a count, a list or ranges (`50`, `0,5,10-12`, `1000-1049`)
and defaults to the trial counts of step 2; seeds must belong to the chosen split. `--split`
defaults to `seen`: `unseen` is the held-out split, so a run on it prints a reminder to report only
and never tune on those seeds. The result is written to `outputs/eval/<UTC timestamp>_<short commit>_<scenario>_<split>.json` (the timestamp
stands in for the date of step 1 so two baselines run on one day do not overwrite each other),
checked against `source/kitchenbot/evaluation/schema.json`, and summarised on stdout. When several
causes of section 7 occur in one trial, the one listed first there is the primary tag.
`--scenario fake` is synthetic, needs no simulator, is what CI runs, and is marked
`run.synthetic`: it is never a result. A new scenario registers a `ScenarioSpec` with
`register_scenario` in `kitchenbot/evaluation/runner.py`, whose loader imports its environment
config inside the function body and returns a backend with `run_batch(seeds)`.

Figures and the results table are regenerated from the JSON files, never edited by hand:
`python scripts/plot_results.py` writes `success_rates.png`, `failure_tags.png` and
`torso_error.png` to `outputs/figures/`, and `python scripts/update_experiments.py` appends the
rows of every file in `outputs/eval/` that `docs/experiments.md` does not list yet (keyed by file
name, so it is safe to run twice). Both refuse synthetic `fake` results unless `--allow-synthetic`
is given, which is only for CI and scratch copies (`--doc`); never commit such rows.

## 5. Targets

Targets, not results. They are what "the MVP works" means.

| Metric | Target (unseen split) |
| --- | --- |
| S0 standing duration | 60 s without a fall in 95 of 100 trials, under 30 N pushes |
| S0 torso orientation error | under 5 deg RMS |
| S1 hand target error | under 3 cm in 90 percent of trials, no fall |
| S2 grasp success, Dex3 | at least 70 percent |
| S2 grasp success, kinematic attach baseline | at least 95 percent |
| S3 placement success | at least 80 percent |
| S3 collision rate | under 20 percent |
| S4 task success, balance policy active | at least 60 percent |
| S4 fall rate | under 5 percent |
| S4 completion time | under 20 s median |

If a target is missed, the report says so and the failure analysis explains why. Targets are not
lowered to match results.

## 6. Baselines and ablations

| Comparison | Why it matters |
| --- | --- |
| Fixed pelvis vs balance policy active | Isolates the cost of real whole-body balance |
| Kinematic attach vs Dex3 friction grasp | Isolates contact physics from planning |
| IK arm vs learned arm policy (if M3b is reached) | The robot-learning claim of the project |
| Seen vs unseen split | Generalization, not memorization |
| With vs without domain randomization during training | Robustness story |

## 7. Failure taxonomy

Every failed trial is tagged with exactly one primary cause, logged in the JSON:

`fall` (balance lost), `ik_infeasible` (no IK solution within limits), `grasp_miss` (fingers closed,
plate not held), `drop` (plate lost after lift), `rack_collision` (contact force threshold hit),
`misplace` (released outside slot volume), `timeout`.

`scripts/plot_results.py` produces the failure histogram per scenario. The technical report
discusses the top two causes per scenario.

## 8. Deliverables checklist (M6)

- [ ] `outputs/eval/*.json` for every scenario, both splits, every applicable baseline
- [ ] Figures: success rates with intervals, failure histograms, torso error over time
- [ ] 2 to 3 minute demo video, with on-screen labels saying which controller is active
- [ ] `docs/experiments.md` with measured values, dates, commits, seeds, GPU time
- [ ] Reused vs developed table in the README
- [ ] Failure analysis section in the technical report
