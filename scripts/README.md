# Scripts

Entry points. Each is created by the issue named; rows with a `#` number exist, rows with a `P`
number are planned.

| Script | Purpose | Issue |
| --- | --- | --- |
| `smoke/create_empty.py` | Isaac Lab empty-scene tutorial, copied with attribution, for the install check | P01 |
| `smoke/dump_urdf_joints.py` | parse a URDF with the standard library and dump joints, origins (`origin_xyz`, `origin_rpy`), limits and links to JSON (no simulator) | #28, #48 |
| `make_plate_usd.py` | generate the rimmed plate assets under `source/kitchenbot/scene/assets/plates/` and validate them with `--check` (no simulator) | #30 |
| `make_rack_tines_usd.py` | generate `source/kitchenbot/scene/assets/dishwasher/rack_tines.usda` (base frame, two rows of capsule tines, slot markers; static primitive colliders) from `kitchenbot.scene.kitchen_dims`; `--check FILE` compares a file with it to 1 mm and exits 1 on any mismatch (needs `usd-core`, no simulator) | #47 |
| `make_counter_usd.py` | generate `source/kitchenbot/scene/assets/kitchen/counter.usda` (counter with a recessed sink) and `floor.usda` (top face z = 0) from `kitchenbot.scene.kitchen_dims`; `--check FILE...` verifies the counter-top height and the floor top (needs `usd-core`, no simulator) | #47 |
| `make_reach_envelope.py` | sample the right palm's reachability envelope with URDF forward kinematics (2 cm pelvis-frame voxels, waist at zero) into `source/kitchenbot/controllers/data/right_arm_envelope.npz`; `--seed`, `--samples`; `--check FILE` prints fill fraction, bounding box and file size; `--plot` writes `outputs/figures/right_arm_envelope.png` (no simulator) | #48 |
| `make_reach_envelope.py --waist-*` | sample the waist too: `--waist-yaw-deg` and `--waist-roll-deg` (half ranges, +-), `--waist-pitch-min-deg` and `--waist-pitch-max-deg` (positive leans forward), all default 0 (the #48 envelope, bit for bit) and clipped to `joint_limits.WAIST_LIMITS`; the ranges go into the `.npz` and `--check` prints them; with a range and no `--out` it writes `right_arm_envelope_waist.npz` (committed: `--seed 48 --samples 8000000 --waist-yaw-deg 20 --waist-pitch-max-deg 30`) | #56 |
| `reach_gap_analysis.py` | the planner's palm targets (`plan_waypoints`) of every eval seed of the ten (scenario, split) pairs, in the pelvis frame, against both envelopes: stand-off and plate box sweeps, a provisional rack-pose sweep, sensitivity rows; `--rack-pose X Y Z YAW_DEG` (provisional default), `--pelvis-height` (0.75 m), `--grasp-point`, `--max-seeds`; writes CSV and `tables.md` to `outputs/reach_gap/` and `--figure` (`docs/figures/reach_gap.png`); results in `docs/reach_gap_analysis.md` (no simulator) | #56 |
| `make_eval_configs.py` | write the ten `configs/eval/s<N>_<split>.yaml` (scenario, split, physics, `num_envs`, seeds, evaluator defaults, the `randomization` block of `RandomizationRanges`) from `kitchenbot.scene.randomization`; `--check` regenerates into a temporary directory, diffs, and exits 1 on any difference (no simulator) | #52 |
| `host_check.ps1` | Windows host readiness check for the install: long paths, VS 2022 C++ tools, `uv`, NVIDIA driver, disk, RAM, pending reboot; read-only, `-Fix` for the safe fixes (PowerShell 5.1, no simulator) | #35 |
| `smoke/dump_g1_joints.py` | spawn the G1 + Dex3 asset, print joints, links, DoF, defaults | P03 |
| `smoke/load_kitchen.py` | load scene layers and step; report FPS | P10 to P12 |
| `fetch_assets.py` | manifest-driven download of third-party assets (`assets_manifest.yaml`) into `source/kitchenbot/scene/assets/third_party/` with SHA256 check; `--set dishwasher|cabinets|all`, `--dry-run` | P29 |
| `usd_inspect.py` | print prim tree, revolute and prismatic joints, mesh count and collision approximations of a USD file (needs `usd-core`) | P29 |
| `train_stand.py` | thin wrapper around `isaaclab train` for the standing task with the pinned config | P06 |
| `export_policy.py` | TorchScript export plus joint-order and observation manifest | P06 |
| `evaluate.py` | run scenarios S0 to S4 per `docs/evaluation.md` and write JSON; `--config FILE` records the SHA256 of the committed `configs/eval` YAML in `run.config_sha256`; `--scenario fake` and `--scenario dry` are synthetic and need no simulator (`dry` chains the sampler, the planner against test doubles and the judges, and reads the YAML's ranges) | #23, #52, #60 |
| `plot_results.py` | figures from `outputs/eval/*.json`; synthetic `fake` and `dry` results only with `--allow-synthetic`, and the figures say so | #24, #60 |
| `update_experiments.py` | append measured rows to `docs/experiments.md` | #24 |
| `record_demo.py` | demo video with on-screen controller labels | P25 |

Every script that touches simulation takes `--headless`, `--num_envs`, `--seed` and passes
`physics=isaacsim_physx`.
