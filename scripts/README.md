# Scripts

Entry points. Each is created by the issue named; rows with a `#` number exist, rows with a `P`
number are planned.

| Script | Purpose | Issue |
| --- | --- | --- |
| `smoke/create_empty.py` | Isaac Lab empty-scene tutorial, copied with attribution, for the install check | P01 |
| `smoke/dump_urdf_joints.py` | parse a URDF with the standard library and dump joints, limits and links to JSON (no simulator) | #28 |
| `make_plate_usd.py` | generate the rimmed plate assets under `source/kitchenbot/scene/assets/plates/` and validate them with `--check` (no simulator) | #30 |
| `smoke/dump_g1_joints.py` | spawn the G1 + Dex3 asset, print joints, links, DoF, defaults | P03 |
| `smoke/load_kitchen.py` | load scene layers and step; report FPS | P10 to P12 |
| `fetch_assets.py` | download third-party assets into `source/kitchenbot/scene/assets/third_party/` with SHA256 manifest | P11 |
| `train_stand.py` | thin wrapper around `isaaclab train` for the standing task with the pinned config | P06 |
| `export_policy.py` | TorchScript export plus joint-order and observation manifest | P06 |
| `evaluate.py` | run scenarios S0 to S4 per `docs/evaluation.md` and write JSON | P23 |
| `plot_results.py` | figures from `outputs/eval/*.json` | P24 |
| `update_experiments.py` | append measured rows to `docs/experiments.md` | P24 |
| `record_demo.py` | demo video with on-screen controller labels | P25 |

Every script that touches simulation takes `--headless`, `--num_envs`, `--seed` and passes
`physics=isaacsim_physx`.
