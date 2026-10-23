# Environment setup

Target host: Windows 11, NVIDIA RTX 5080 (16 GB), 32 GB RAM, Python 3.12. Everything below was
checked against the Isaac Lab 3.0.0 and Isaac Sim 6.2 documentation on 2026-10-22. Versions are
pinned; do not substitute.

## 1. Pinned toolchain

| Component | Version | Source |
| --- | --- | --- |
| Isaac Lab | 3.0.0 | https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0 |
| Isaac Sim | 6.2.0.0 (pip extra `isaaclab[isaacsim]`) | https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/installation.html |
| Python | 3.12.x (`>=3.12,<3.13`) | Isaac Lab `pyproject.toml` |
| PyTorch | 2.11.0, CUDA 13.0 wheels (`--torch-backend cu130`) | same |
| RSL-RL | `rsl-rl-lib` 5.5.1 | same |
| Newton / Warp | 1.6.1 / 1.17.0 (installed, not used; we run PhysX) | same |
| NVIDIA driver (Windows) | 595.97 tested by NVIDIA; host currently has 616.92 | https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html |
| Package manager | `uv` | Isaac Lab 3.0 install docs |

Physics backend: **`physics=isaacsim_physx`** on every command. Isaac Lab 3.0 defaults to Newton
(`newton_mjwarp`); forgetting the flag silently trains on the wrong engine.

## 2. One-time Windows preparation (operator)

Run the host check first. It changes nothing, prints one PASS / WARN / FAIL line per item below with the fix for each FAIL, and exits 1 if any FAIL remains:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\host_check.ps1
```

`-Fix` (elevated shell) enables long paths, sets the git option and installs `uv`; it never installs Visual Studio or a driver. Re-run the check after the reboot and before #1. The steps below are the manual equivalents.

1. Enable long paths, in an elevated PowerShell, then reboot:
   ```powershell
   New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
   git config --global core.longpaths true
   ```
   Both Isaac Lab and Isaac Sim docs require this before cloning or installing.
2. Install Visual Studio 2022 Build Tools with the "Desktop development with C++" workload. Isaac
   Sim's known-issues page maps the Windows `WinError 126` on import to a missing VS 2022 runtime.
3. Install `uv`: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.
4. Keep the NVIDIA driver at 595.97 or newer. If Isaac Sim fails to start on 616.92, the documented
   tested version is 595.97; record what you observe in `docs/experiments.md`.
5. Free disk: Isaac Sim lists 50 GB minimum. Budget 100 GB for Isaac Sim, caches, logs and checkpoints.

## 3. Install (kb-sim, M0)

The virtual environment lives **outside** the repository so every worktree shares it.

```bat
uv venv --python 3.12 C:\venvs\kitchenbot
C:\venvs\kitchenbot\Scripts\activate
set OMNI_KIT_ACCEPT_EULA=YES
uv pip install "isaaclab[isaacsim]==3.0.0" ^
  --overrides "https://raw.githubusercontent.com/isaac-sim/IsaacLab/v3.0.0/tools/wheel_builder/uv-overrides.txt" ^
  --extra-index-url https://pypi.nvidia.com ^
  --torch-backend cu130 ^
  --index-strategy unsafe-best-match
```

The `--overrides` file is required because Isaac Sim 6.2 pins dependencies that conflict with
Isaac Lab. The docs show this command only with bash line continuations; the `^` form above is the
Windows equivalent and is one of the things M0 verifies.

Then install this repository in editable mode into the same environment:

```bat
cd <repo>
uv pip install -e .
```

Migration note: Isaac Lab 3.0 removed the `isaaclab.bat --install/--conda/--uv` flow. Use `uv` and
the `isaaclab` CLI (`isaaclab list_envs`, `isaaclab train`, `isaaclab play`). Do not follow 2.x
tutorials that call `isaaclab.bat -i`.

### Pre-commit hook (once per clone)

Lint runs locally before each commit with the same check CI runs (`ruff check .`), pinned in
`.pre-commit-config.yaml`:

```bat
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

CI does not check formatting, so the hook only lints. It does not replace running `pytest -q`.

## 4. Verify (M0 acceptance, in order)

| Step | Command | Pass when |
| --- | --- | --- |
| 1 | `isaacsim isaacsim.exp.compatibility_check` | Reports GPU and driver as compatible |
| 2 | `isaacsim` | Kit window opens (first launch can take over 10 minutes while extensions compile) |
| 3 | `python scripts/smoke/create_empty.py --viz kit` (copied from Isaac Lab `scripts/tutorials/00_sim/create_empty.py`) | Empty stage renders, exits cleanly |
| 4 | `isaaclab list_envs --show_presets` | Lists `Isaac-Velocity-Flat-G1` and `Isaac-Velocity-Rough-G1` with `isaacsim_physx` among presets |
| 5 | `isaaclab play --task Isaac-Velocity-Flat-G1 --checkpoint pretrained --num_envs 16 --viz kit physics=isaacsim_physx` | 16 G1 robots walk on the NVIDIA pretrained policy |
| 6 | `isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-G1 --num_envs 1024 --max_iterations 100 --viz none physics=isaacsim_physx` | 100 iterations finish, reward increases, peak VRAM recorded from `nvidia-smi` |

Exact flag names in steps 5 and 6 come from the documented CLI syntax; confirm each with
`isaaclab train --help` and `isaaclab play --help` before filing the result. Windows note: stop a run
with Ctrl+Break, not Ctrl+C.

## 5. The Windows go/no-go gate

The operator chose to try Windows 11 first instead of dual-booting Ubuntu 24.04. The gate:

- **GO** if steps 1 to 6 all pass within 8 hours of effort.
- **NO-GO** if any step is still failing after 8 hours, or if step 6 runs below 20,000 FPS
  (environment steps per second, as printed by RSL-RL) at 1024 envs, which would make M1 training
  impractically slow. On NO-GO, kb-sim files a `status:blocked` comment on the M0 issue with the
  failure log and the operator installs Ubuntu 24.04 alongside Windows. The pip install above is
  identical on Ubuntu except for the venv activation path.

Known Windows limits that do **not** affect the MVP, documented by NVIDIA: multi-GPU and Ray
tuning are Linux-only; Pink IK (`PinkInverseKinematicsAction`), Isaac Lab Mimic and XR teleop are
Linux-only. The architecture uses `DifferentialIKController` instead of Pink for this reason.

## 6. Reference numbers for sizing runs

Published Isaac Lab 3.0 benchmarks, rsl_rl, `isaacsim_physx`, measured on an RTX PRO 6000 Blackwell
with 96 GB VRAM. They do not transfer to the RTX 5080 directly; use them for relative sizing only.

| Task | num_envs | FPS | Peak VRAM |
| --- | --- | --- | --- |
| Isaac-Velocity-Flat-G1 | 8192 | 136,950 | 11.6 GB |
| Isaac-Velocity-Rough-G1 | 8192 | 112,289 | 13.2 GB |
| Isaac-Velocity-Rough-G1 | 4096 | 74,529 | 14.7 GB |

Older NVIDIA numbers for an RTX 4090 put Rough-G1 at 4096 envs near 82,000 FPS and 6.1 GB VRAM on
Isaac Lab 2.x. The RSL-RL config trains Flat-G1 for 1,500 iterations and Rough-G1 for 5,000, with
24 steps per env per iteration. Start every new task at 1024 envs on this host and scale up while
watching `nvidia-smi`.

Published wall-clock anecdotes, all on an RTX 4090 at 4096 envs and all on Linux: Unitree's G1
velocity policy in `unitree_rl_lab` converges in about 2,000 iterations in 30 to 45 minutes; an H1
rough-terrain run of 3,000 iterations was logged at about 1 hour 20 minutes on a DGX Spark. These
calibrate expectations only. The first measured run on this host replaces them.

## 7. Known gaps (verify in M0)

- The `^`-continued install command has not been run on this host yet.
- Whether driver 616.92 behaves like the tested 595.97.
- Disk footprint of the `isaacsim[all,extscache]` extra after first launch.
