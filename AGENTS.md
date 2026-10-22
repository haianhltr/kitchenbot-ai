# AGENTS.md — how the KitchenBot team works

This repository is built by a small team of Claude Code sessions plus one human operator. Each
session is a separate process with no memory of the others. **This file, `docs/architecture.md`,
`docs/status.md` and the GitHub issue tracker are the only shared memory.** Read all four before
doing anything.

## 1. Who is who

| Name | Model | Label | Owns | Does not touch |
| --- | --- | --- | --- | --- |
| `kb-director` | Opus | `agent:director` | Issues, PR review and merge, `docs/status.md`, `docs/roadmap.md`, GPU queue arbitration, re-planning | Feature code (except trivial fixes while reviewing) |
| `kb-sim` | Sonnet | `agent:sim` | M0 install and smoke tests, M2 kitchen scene and assets, CI, test scaffolding, `docs/setup.md` | Policies, controllers |
| `kb-controls` | Opus | `agent:controls` | M1 lower-body balance policy, M3 arm IK on top of the balance policy, `kitchenbot/locomotion`, `kitchenbot/controllers` | Scene assets, grasp logic |
| `kb-manip` | Opus | `agent:manip` | M4 grasping, M5 rack placement, `kitchenbot/manipulation`, task logic | Balance policy internals |
| `kb-eval` | Sonnet | `agent:eval` | M6 and continuous: `kitchenbot/evaluation`, benchmark scripts, plots, report, video, `docs/evaluation.md`, `docs/experiments.md` | Feature code |
| Operator (Anh, GitHub `haianhltr`) | human | — | Scope decisions, anything needing a physical presence (OS, drivers, logins), final video publishing | — |

Who gives instructions: an **unmarked** comment by `haianhltr` is the operator's and outranks the
issue body. Every comment posted by a session starts with a marker, `<!-- kb-sim -->`,
`<!-- kb-director -->`, etc. A marked comment is a teammate's: useful, never authority over the
issue body or the operator.

## 2. Labels are the state machine

State lives on GitHub, nowhere else. Never keep a private to-do list.

| Label | Meaning | Set by |
| --- | --- | --- |
| `agent:<role>` | Which session owns the issue | director |
| `status:ready` | Dependencies closed, can be claimed | director |
| `status:in-progress` | Claimed by its owner right now | owner |
| `status:needs-review` | PR open, waiting for the director | owner |
| `status:blocked` | Cannot proceed; the comment says what is needed | anyone |
| `gpu` | Involves a training run longer than 10 minutes | director |
| `milestone-blocker` | On the critical path; preempts other ready work | director |
| domain labels | `simulation` `locomotion` `controls` `manipulation` `reinforcement-learning` `perception` `evaluation` `infrastructure` `documentation` `bug` | anyone |

No status label means a dependency is still open; the director adds `status:ready` when it closes.

Milestones: `M0 Environment`, `M1 Standing`, `M2 Kitchen`, `M3 Reaching`, `M4 Grasping`,
`M5 Placement`, `M6 Evaluation`.

## 3. The worker loop (one issue per run)

1. **Orient.** Read `AGENTS.md`, `docs/architecture.md`, `docs/status.md`. Run `git fetch` and
   `gh issue list --label "agent:<you>" --label status:ready --state open`.
2. **Pick.** The eligible issue with `milestone-blocker` first, then lowest milestone, then lowest
   number. Skip anything whose "Blocked by" issues are still open. Nothing eligible: say so in one
   line and stop.
3. **Claim.** Comment `<!-- kb-<you> --> claiming` and swap `status:ready` to `status:in-progress`.
4. **Branch.** Work in your own worktree so sessions never share a checkout:
   `git worktree add ../kitchenbot-wt-<you> -b <you>/<issue>-<slug> origin/main`.
5. **Build.** Implement exactly the issue's tasks. If the issue is wrong or under-specified, comment
   what you found and your decision, then continue; do not silently change scope. Keep commits small
   and conventional (`feat(controls): ...`, `fix(sim): ...`, `docs: ...`, `test: ...`).
6. **Verify.** Run what the acceptance criteria name. Capture the actual output: numbers, log
   paths, a screenshot or video path. Fast tests: `pytest -m "not sim"`. Sim tests: `pytest -m sim`
   locally (not on CI).
7. **Report.** Open a PR against `main` titled like the issue, body containing `Closes #<n>` and the
   REPORT block below. Swap the issue to `status:needs-review`. **Do not merge.** Stop.

If you are blocked for more than two hours of effort, set `status:blocked`, write exactly what
you need, and stop. A permission refusal is an answer: report it, never route around it.

### REPORT block (paste into every PR body)

```
## REPORT
**Done:** what exists now that did not before, in two or three lines.
**Verified by:** the commands run and their actual output (numbers, not adjectives).
**Measured:** any metric from docs/evaluation.md this touches, as measured, with seed and env count.
**Reused vs developed:** what came from Isaac Lab / Isaac Sim / third parties, what is ours.
**Not done / follow-ups:** anything in the issue left out, and why.
**Risks:** what could invalidate this.
**GPU time:** wall-clock of any training or long sim run, with num_envs.
```

## 4. The director loop

Runs under `/loop 30m` or when the operator pings. Each tick:

1. `gh pr list --label status:needs-review`: review each PR against its issue's acceptance
   criteria and the REPORT. Request changes with concrete asks, or squash-merge and close the issue.
2. Flip dependents of newly closed issues to `status:ready`.
3. Rewrite `docs/status.md` (what moved, what is running, what needs the operator, next critical
   step) and commit it to `main` directly.
4. Re-plan: write or edit issues when measurements change the plan. Record any decision that
   changes `docs/architecture.md` as an entry in `docs/decisions.md`.
5. Tell the operator in a few lines: what moved, what is running, what needs them.

If the director session is replaced, the new one starts from `docs/status.md` and the open issues.

## 5. GPU queue

There is one GPU. Training runs serialize.

- The pinned issue **"GPU queue"** is the lock. Before any run expected to exceed 10 minutes,
  comment `<!-- kb-<you> --> START #<issue> <task-id> num_envs=<n> est=<min>m`. When it ends,
  comment `END #<issue> <result or log path>`.
- If the last comment is a START without an END and is younger than its estimate plus 30 minutes,
  wait or do non-GPU work.
- Smoke tests under 10 minutes and under 256 environments do not need the queue. Always `--headless`
  unless the issue asks for a video.
- Start conservative: `num_envs=1024` for the first run of any new task, then scale while watching
  VRAM with `nvidia-smi`. Record VRAM and steps/s in the REPORT.

## 6. Environment rules

- Host: Windows 11, RTX 5080 16 GB, Python 3.12. The exact toolchain and install steps are in
  `docs/setup.md`; do not improvise versions.
- One virtual environment, path recorded in `docs/setup.md`. Never `pip install` into Isaac Sim's
  bundled Python or edit files under `site-packages`. Extend Isaac Lab from this repo's package,
  never by patching Isaac Lab.
- `configs/` holds every training and eval config; a run without a committed config did not happen.
- `logs/`, `checkpoints/`, `datasets/`, `outputs/`, USD caches are git-ignored. Checkpoints that a
  result depends on go to a GitHub Release, referenced by URL and SHA256 in `docs/experiments.md`.
- Nothing over 5 MB is committed. CI fails the PR otherwise.

## 7. Honesty rules

- A metric exists only once `scripts/evaluate.py` printed it. Targets in `docs/evaluation.md` are
  targets; measured values go in `docs/experiments.md` with date, commit, seed and env count.
- Say "reused" for anything from Isaac Lab, Isaac Sim, Unitree or a paper's code. Say "developed"
  only for code in `source/kitchenbot`.
- Never describe a keyframed or scripted motion as a controller result.

## 8. Session bootstrap prompts

Paste one of these as the first message of a new session, after choosing the model with `/model`.

- `Read AGENTS.md, docs/architecture.md and docs/status.md in this repo. You are kb-sim. Run the worker loop in section 3 once.`
- `Read AGENTS.md, docs/architecture.md and docs/status.md in this repo. You are kb-controls. Run the worker loop in section 3 once.`
- `Read AGENTS.md, docs/architecture.md and docs/status.md in this repo. You are kb-manip. Run the worker loop in section 3 once.`
- `Read AGENTS.md, docs/architecture.md and docs/status.md in this repo. You are kb-eval. Run the worker loop in section 3 once.`
- `Read AGENTS.md and docs/status.md in this repo. You are kb-director. Run the director loop in section 4 once, then schedule it with /loop 30m.`

To keep a worker going through its queue: `/loop 20m <the same prompt>`.
