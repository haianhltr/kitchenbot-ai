# Isaac Lab 3.0 API notes for the stand environment (#46)

Every Isaac Lab symbol the #46 drafts use, with the file and line where it is defined in the
`v3.0.0` tag, its signature and what the drafts rely on. Written by reading the source only;
nothing here was run.

- Source: `git clone --depth 1 --branch v3.0.0 https://github.com/isaac-sim/IsaacLab.git`,
  read on 2026-10-23. Tag commit **`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`**
  ("Pin Python 3.12 in release documentation uvx commands (#8738)", 2026-10-19), `VERSION` = `3.0.0`.
- Paths are relative to the repository root, so `source/isaaclab/isaaclab/...` is the `isaaclab`
  package. A permalink is
  `https://github.com/isaac-sim/IsaacLab/blob/e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4/<path>#L<line>`.
- Line numbers point at the `class`/`def` line or the field line quoted.
- **Not found** marks something our docs assume that the source does not have, with what exists
  instead. The drafts follow the source, not the docs.

## 1. Environment and simulation config

| Symbol | Where (v3.0.0) | What the drafts rely on |
| --- | --- | --- |
| `ManagerBasedRLEnvCfg` | `source/isaaclab/isaaclab/envs/manager_based_rl_env_cfg.py:15` | Fields `episode_length_s` (:63), `rewards` (:77), `terminations` (:83), `curriculum` (:89), `commands` (:95), method `play_mode()` (:101) |
| `ManagerBasedEnvCfg` | `source/isaaclab/isaaclab/envs/manager_based_env_cfg.py:43` | Fields `sim` (:50), `seed` (:66), `decimation` (:74), `scene` (:82), `observations` (:94), `actions` (:100), `events` (:106) |
| `ManagerBasedRLEnv` | `source/isaaclab/isaaclab/envs/manager_based_rl_env.py:22` | Gym entry point `"isaaclab.envs:ManagerBasedRLEnv"` |
| `SimulationCfg` | `source/isaaclab/isaaclab/sim/simulation_cfg.py:36` | `dt` (:53, default 1/60), `physics_material` (:62), `render_interval` (:82), `physics: PhysicsCfg \| None = None` (:114) |
| `PhysxCfg` | `source/isaaclab_physx/isaaclab_physx/physics/physx_manager_cfg.py:33` | The Isaac Sim PhysX backend config; `launcher_type = "isaaclab_physx.app:KitLauncher"` (:58), `gpu_max_rigid_patch_count` (:207) |
| `PhysxAutoCfg` | `source/isaaclab/isaaclab/physics/physics_manager_cfg.py:55` | Fields `isaacsim_physx` (:61) and `ovphysx`; resolved to one of them at launch (:68 `_resolve_physx_auto_cfg`) |
| `PresetCfg` | `source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py:103` | A config whose fields are named alternatives; the field `default` is used when no CLI selector is given |
| `InteractiveSceneCfg` | `source/isaaclab/isaaclab/scene/interactive_scene_cfg.py:22` | `num_envs` (:86), `env_spacing` (:89) |
| `TerrainImporterCfg` | `source/isaaclab/isaaclab/terrains/terrain_importer_cfg.py:21` | `terrain_type: Literal["generator", "plane", "usd"]` (:46); the drafts use `"plane"` |
| `PhysxRigidBodyMaterialCfg` | `source/isaaclab_physx/isaaclab_physx/sim/spawners/materials/physics_materials_cfg.py:176` | Ground material; `friction_combine_mode` (:219), `restitution_combine_mode` (:229) |
| `RigidBodyMaterialCfg` (alias) | same file `:308` | Docstring: "Deprecated ... `.. deprecated:: 3.1` ... scheduled for removal in 4.0". The velocity task still uses `sim_utils.RigidBodyMaterialCfg` (`velocity_env_cfg.py:91`); the drafts use `PhysxRigidBodyMaterialCfg` instead |
| `AssetBaseCfg` | `source/isaaclab/isaaclab/assets/asset_base_cfg.py:19` | Light prim |
| `DomeLightCfg` | `source/isaaclab/isaaclab/sim/spawners/lights/lights_cfg.py:108` | Light |
| `ContactSensorCfg` | `source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_cfg.py:19` | `track_air_time` (:63), `history_length` (:74); data property `net_normal_forces_w_history` (`base_contact_sensor_data.py:93`, a `ProxyArray`) |
| `configclass`, `clone`, `replace` | `source/isaaclab/isaaclab/utils/configclass.py:40`, `:158`, `:142` | `clone(cfg)` is `dataclasses.replace(cfg)`, and configclass deep-copies every field on init (`_custom_post_init`, :483-501), so editing a clone never changes `G1_29DOF_CFG`; exported from `isaaclab.utils` (`utils/__init__.pyi:117`) |
| `TerrainImporterCfg` fields | `source/isaaclab/isaaclab/terrains/terrain_importer_cfg.py:30`, `:94` | `collision_group: int = -1`, `physics_material` |
| `SensorBaseCfg.update_period` | `source/isaaclab/isaaclab/sensors/sensor_base_cfg.py:45` | Set to `sim.dt` for the contact sensor, as `velocity_env_cfg.py:365-366` does |
| `ManagerBasedEnv.step_dt` / `num_envs` | `source/isaaclab/isaaclab/envs/manager_based_env.py:268` / `:255` | `step_dt` = `decimation * sim.dt` (0.02 s here); the command term advances its interpolation by it |
| `ManagerBasedRLEnv.episode_length_buf` | `source/isaaclab/isaaclab/envs/manager_based_rl_env.py:75` | Steps in the current episode, per env; the curriculum skips envs where it is 0 |

**Physics preset: CLI selector, and it needs a preset field in the config.** The docs say "physics
preset `isaacsim_physx` on every command". In the source, `physics=isaacsim_physx` on the
`isaaclab train/play/zero_agent/random_agent` command line is a *typed preset selector*, parsed in
`source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py:589-608` ("`physics=NAME` / `renderer=NAME`
are typed selectors that must resolve against a config of that type"). It works only if the
environment config holds a `PresetCfg` with a field of that name whose value is a `PhysicsCfg`;
otherwise `_validate_typed_presets` raises `ValueError("Typed preset selector(s) ... did not match a
selectable preset of that type for this task ...")` (`hydra.py:531-556`). It is **not** a field
of `SimulationCfg`: that class has only `physics: PhysicsCfg | None` (:114). The reference
pattern is `RoughPhysicsCfg(PresetCfg)` (`source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py:50-72`):

```python
@configclass
class RoughPhysicsCfg(PresetCfg):
    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    ovphysx = OvPhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp = NewtonCfg(...)
    newton_kamino = NewtonCfg(...)
    default = newton_mjwarp
...
sim: SimulationCfg = SimulationCfg(physics=RoughPhysicsCfg())   # velocity_env_cfg.py:339
```

The default really is Newton (`default = newton_mjwarp`, :72; the flat G1 task sets
`self.sim.physics.default = newton_mjwarp`, `config/g1/flat_env_cfg.py:25`). The draft
`StandPhysicsCfg` declares only `isaacsim_physx` and sets `default = isaacsim_physx`, so the run is
PhysX even if the flag is forgotten, and the flag still resolves. A second, different mechanism
exists for standalone scripts: the launcher argument `physics` replaces every `PhysicsCfg` in the
tree with a fresh default `PhysxCfg()` (`source/isaaclab/isaaclab/app/sim_launcher.py:108-136`,
`:370-375`), which would drop our `gpu_max_rigid_patch_count`; the drafts are written for the
`isaaclab` CLI path.

**Play variant: Not found as a `_PLAY` class.** The issue and the docs ask for `StandEnvCfg_PLAY`.
Isaac Lab 3.0 has no `_PLAY` config classes for the velocity tasks: the G1 registration lists one
`env_cfg_entry_point` per task (`config/g1/__init__.py:16-38`), and play overrides live in a method,
`ManagerBasedRLEnvCfg.play_mode()` (`manager_based_rl_env_cfg.py:101`): "Play scripts call this
method after the configuration is fully initialized ... caps the number of environments at 50 ...
disables observation corruption". The velocity task overrides it (`velocity_env_cfg.py:377-389`)
to remove the push events. The draft overrides `play_mode()` and also keeps a thin
`StandEnvCfg_PLAY` subclass that calls it from `__post_init__`, for scripts that want a class.

## 2. Robot asset

| Symbol | Where (v3.0.0) | What the drafts rely on |
| --- | --- | --- |
| `G1_29DOF_CFG` | `source/isaaclab_assets/isaaclab_assets/robots/unitree.py:413` | Module docstring: "G1 humanoid robot configured for locomanipulation tasks" (:17) |
| its USD | same, :415 | `usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd"` (Isaac Sim asset root) |
| `activate_contact_sensors` | same, :416 | **`False`**. Contact sensors need it `True` (`spawner_cfg.py:165`, "Activate contact reporting on all rigid bodies"), so `G1_DEX3_CFG` sets it |
| `fix_root_link` | same, :434 | `fix_root_link=False,  # Configurable - can be set to True for fixed base`; the spawner field is `FileCfg.fix_root_link: bool \| None` (`source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py:80`) |
| `init_state` | same, :436-445 | `pos=(0.0, 0.0, 0.75)`, `rot=(0, 0, 0.7071, 0.7071)`, joint_pos only for hip pitch -0.10, knee 0.30, ankle pitch -0.20; everything else 0 |
| quaternion order | `source/isaaclab/isaaclab/assets/asset_base_cfg.py:41-42` | `rot` is "Quaternion rotation (x, y, z, w)". Our `planning/geometry.py` uses w, x, y, z; #21 must convert |
| `soft_joint_pos_limit_factor` | `unitree.py:446` | `0.9` |
| actuator groups | `unitree.py:448` legs `DCMotorCfg`, `:485` feet `DCMotorCfg`, `:506` waist `ImplicitActuatorCfg`, `:532` arms `ImplicitActuatorCfg`, `:550` hands `ImplicitActuatorCfg` | Hands regexes `".*_index_.*"`, `".*_middle_.*"`, `".*_thumb_.*"` |
| hip roll limits | `unitree.py:457` `".*_hip_roll_joint": 88.0` (effort), `:463` `".*_hip_roll_joint": 32.0` (velocity) | The known defect; the URDF (`tests/fixtures/g1_dex3_joints_urdf.json`) says 139 N·m, 20 rad/s |
| knee limits | `unitree.py:459` `".*_knee_joint": 139.0`, `:465` `".*_knee_joint": 20.0` | **Already the rev 1.0 values**; the draft re-states them so the intent is explicit, no value changes |
| `ArticulationCfg` | `source/isaaclab/isaaclab/assets/articulation/articulation_cfg.py:21` | `init_state` (:58), `soft_joint_pos_limit_factor` (:61), `joint_ordering` (:71), `actuators` (:103); `prim_path` is inherited from `AssetBaseCfg` (`asset_base_cfg.py:63`) |
| `ArticulationCfg.joint_ordering` | same, :71 | "Public joint-name ordering convention or complete explicit permutation ... a list or tuple ... containing every backend joint name exactly once." `G1_DEX3_CFG` sets it to `joint_maps.ALL_JOINTS` so every joint-indexed tensor uses our order; a name mismatch fails loudly at init, which is #3's check |
| `ActuatorBaseCfg` | `source/isaaclab/isaaclab/actuators/actuator_base_cfg.py:23` | `actuator_effort_limit` (:39), `actuator_velocity_limit` (:52), `joint_effort_limit` (:73), `joint_velocity_limit` (:79), `stiffness` (:100), `damping` (:110) |
| deprecated limit names | same, `effort_limit_sim` (:86), `velocity_limit_sim` (:93), `effort_limit` (:162), `velocity_limit` (:170) | "`.. deprecated:: 3.0` ... This alias will be removed in 3.1"; the drafts use the new names |
| `DCMotorCfg` | `source/isaaclab/isaaclab/actuators/actuator_pd_cfg.py:44` | `saturation_effort` (:49); explicit model, clips to `actuator_effort_limit` |
| `ImplicitActuatorCfg` | same, :21 | PhysX PD; limits are `joint_effort_limit` / `joint_velocity_limit` |
| `G1_MINIMAL_CFG` | `unitree.py:405-406` | `clone(G1_CFG)` with `{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_minimal.usd`; `G1_CFG` (:294) is the older asset with `torso_joint` and `.*_elbow_pitch_joint` names. The flat and rough G1 velocity tasks use it (`config/g1/rough_env_cfg.py:117`), which is why #5 cannot reuse their joint regexes |
| G1 velocity task configs | `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/flat_env_cfg.py:15` (`G1FlatEnvCfg`), `rough_env_cfg.py:106` (`G1RoughEnvCfg`), base `velocity_env_cfg.py:335` (`LocomotionVelocityRoughEnvCfg`) | The stand env copies their structure, not their joint names |

**Not found: a knee-limit defect.** The issue lists hip-roll *and knee* limits
as fixes. In `v3.0.0` only the hip roll is wrong; the knee is already 139 N·m and 20 rad/s
(`unitree.py:459`, `:465`). Hip pitch and hip yaw stay at 88 N·m and 32 rad/s, which matches the
URDF.

## 3. Actions

| Symbol | Where (v3.0.0) | What the drafts rely on |
| --- | --- | --- |
| `ActionTermCfg` | `source/isaaclab/isaaclab/managers/manager_term_cfg.py:82` | `class_type` (:85), `asset_name` (:91), `debug_vis` (:98), `clip` (:101) |
| `ActionTerm` | `source/isaaclab/isaaclab/managers/action_manager.py:29` | `__init__(cfg, env)` (:42) sets `self._asset = self._env.scene[self.cfg.asset_name]`; abstract `action_dim` (:75), `raw_actions` (:81), `processed_actions` (:87), `process_actions(actions)` (:150, "called once per environment step"), `apply_actions()` (:162, "called at every simulation step") |
| `ActionManager.process_action` | same, :380 | Splits the action vector by `term.action_dim` (:398-401); a term with `action_dim == 0` receives an `(N, 0)` slice |
| `ActionManager.get_term` | same, :412 | |
| `JointActionCfg` | `source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py:30` | `joint_names` (:36), `scale` (:38), `offset` (:40), `preserve_order` (:42) |
| `JointPositionActionCfg` | same, :74 | `use_default_offset: bool = True` (:82) |
| `JointAction.__init__` | `source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:59-68` | `joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names, preserve_order=self.cfg.preserve_order, as_proxy=True)`; `self._joint_ids = joint_ids.torch` |
| `JointAction.process_actions` | same, :169 | `processed = raw * scale + offset`, then clip |
| `JointPositionAction.apply_actions` | same, :196-198 | `self._asset.set_joint_position_target_index(target=self.processed_actions, joint_ids=self._joint_ids)`, i.e. Isaac Lab's own term still calls the deprecated setter |
| `Articulation.find_joints` | `source/isaaclab/isaaclab/assets/articulation/base_articulation.py:555` | `find_joints(name_keys, joint_subset=None, preserve_order=False, *, as_proxy=False)` |
| `AgileBasedLowerBodyActionCfg` | `source/isaaclab_tasks/isaaclab_tasks/contrib/locomanip_pick_place/configs/action_cfg.py:17` | Fields `joint_names` (:25), `obs_group_name` (:28), `policy_path` (:31), `policy_output_offset` (:34), `policy_output_scale` (:37) |
| `AgileBasedLowerBodyAction` | `source/isaaclab_tasks/isaaclab_tasks/contrib/locomanip_pick_place/mdp/actions.py:25` | `action_dim` is 4, "[vx, vy, wz, hip_height]" (:63-65); `process_actions` runs the TorchScript policy on `self._env.obs_buf["lower_body_policy"]` (:108, the group name is hard-coded, `cfg.obs_group_name` is used only for the history length at :86); output scaled and offset by the default pose (:118); `apply_actions` uses the deprecated `set_joint_position_target_index` (:129) |
| Agile usage | `source/isaaclab_tasks/isaaclab_tasks/contrib/locomanip_pick_place/locomanipulation_g1_env_cfg.py:328-337` | Joints `".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"` (12, no waist), `policy_output_scale=0.25`, `policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt"`; registered as `IsaacContrib-PickPlace-Locomanipulation-G1-Abs` (`contrib/locomanip_pick_place/__init__.py:13-21`, no RL entry point) |

**Writing targets for joints that no agent action drives.** The docs say the follow term uses
"`actuators.target_command.position`". In the source `position` is a **read-only** property; the
writer is a method on the same object:

- `ActuatorTargetCommand` (`source/isaaclab/isaaclab/actuators/actuator_collection.py:681`):
  `position` property (:700-703) returns the target buffer; `set_position_index(*, value, joint_ids=None, env_ids=None, full_data=False)`
  (:715-733) writes it; `set_position_mask` (:775) is the mask variant. Its docstring: "Index
  selectors must contain unique environment and joint indices. Repeated indices dispatch concurrent
  writes to the same destination and produce an undefined result."
- Reached as `articulation.actuators.target_command` (`ActuatorCollection.target_command`,
  `actuator_collection.py:138-140`; `BaseArticulation.actuators`, `base_articulation.py:120-127`:
  "set runtime commands through `articulation.actuators.target_command`").
- The deprecation, `source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py:363-371`:

  ```python
  def joint_pos_target(self) -> ProxyArray:
      """Joint position targets commanded by the user [m or rad, depending on joint type].

      .. deprecated:: 3.0.0
          Use ``articulation.actuators.target_command.position`` instead.
  ```

  and the setter, `base_articulation.py:1669-1677`:
  `warnings.warn("Articulation.set_joint_position_target_index is deprecated. Use articulation.actuators.target_command.set_position_index instead.", DeprecationWarning, ...)`
  (same message for `set_joint_position_target`, `:3238-3256`).

The draft `UpperBodyFollowAction.apply_actions` therefore calls
`self._asset.actuators.target_command.set_position_index(value=targets, joint_ids=self._joint_ids)`.
The 15-joint `JointPositionAction` keeps Isaac Lab's deprecated path (it is reused unchanged), so a
`DeprecationWarning` from it in the #5 log is expected.

**Not found: a built-in "hold un-actuated joints" event** (as the docs already say). There is
`reset_scene_to_default(env, env_ids, reset_joint_targets=False)` (`envs/mdp/events.py:1678`) for
resets only.

## 4. Commands

| Symbol | Where (v3.0.0) | What the drafts rely on |
| --- | --- | --- |
| `CommandTermCfg` | `source/isaaclab/isaaclab/managers/manager_term_cfg.py:111` | `class_type` (:114), `resampling_time_range: tuple[float, float]` (:120), `debug_vis` (:122) |
| `CommandTerm` | `source/isaaclab/isaaclab/managers/command_manager.py:26` | `__init__` creates `time_left` and `command_counter` (:39-59); abstract `command` property (:73-77, "Shape is (num_envs, command_dim)"), `_update_metrics()` (:190), `_resample_command(env_ids)` (:195), `_update_command()` (:200) |
| `CommandTerm.reset` | same, :115-144 | Zeroes `command_counter[env_ids]`, then `_resample(env_ids)`; `env_ids` is `slice(None)` when `None` |
| `CommandTerm.compute(dt)` | same, :146-161 | `_update_metrics()`, `time_left -= dt`, `_resample` where `time_left <= 0`, then `_update_command()` |
| `CommandTerm._resample` | same, :167-183 | Sets `time_left[env_ids]` uniform in `resampling_time_range` **before** calling `_resample_command(env_ids)`, then `command_counter[env_ids] += 1`. `env_ids` may be a slice or device indices |
| `CommandManager.get_command` / `get_term` | same, :370 / :381 | |
| `UniformVelocityCommandCfg` | `source/isaaclab/isaaclab/envs/mdp/commands/commands_cfg.py:35` | `heading_command` (:43), `rel_standing_envs` (:54, "The sampled probability of environments that should be standing still"), `rel_heading_envs` (:57), `Ranges` (:65-77), `ranges` (:83) |
| `UniformVelocityCommand` | `source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py` | `command` is `(num_envs, 3)` (:115-117); `is_standing_env` is re-drawn at every resample (:174) and zeroes the command (:192) |

**"60 percent of episodes at exactly zero" needs one resample per episode.** `rel_standing_envs`
is a per-resample probability, not per episode. The draft sets the velocity command's
`resampling_time_range` to the episode length (20 s), so the reset resample is the only one and
the fraction is per episode.

**Manager construction order.** `ManagerBasedRLEnv.load_managers` builds the `CommandManager`
first (`manager_based_rl_env.py:112`), then the parent builds observations and actions
(`manager_based_env.py:363`), then terminations, rewards and curricula
(`manager_based_rl_env.py:120-126`). The follow action can therefore look up the command term in
its `__init__`.

**Step order** (`manager_based_rl_env.py:174-279`): `action_manager.process_action` (:205), then
per physics step `apply_action` (:217), then terminations (:231), rewards (:234), resets (:249),
`command_manager.compute(dt=self.step_dt)` (:271), interval events (:273), observations (:279).
The follow action reads the command once per env step in `process_actions`, i.e. the target the
policy observed on the previous step.

## 5. Observations

| Symbol | Where (v3.0.0) | What the drafts rely on |
| --- | --- | --- |
| `ObservationGroupCfg` | `source/isaaclab/isaaclab/managers/manager_term_cfg.py:244` | `concatenate_terms` (:247), `enable_corruption` (:266), `history_length: int \| None = None` (:273, overrides every term's), `flatten_history_dim: bool = True` (:281), `history_order: Literal["term", "time"] = "term"` (:289) |
| `ObservationTermCfg` | same, :156 | `noise` (:178), `history_length: int = 0` (:220), `flatten_history_dim: bool = True` (:229) |
| history assembly | `source/isaaclab/isaaclab/managers/observation_manager.py:469-500`, time-major flag `:612-618` | With `history_order="time"` (and history, flatten, concatenate, `concatenate_dim == -1`) the group is concatenated per frame, so the flat vector reshapes to `(num_envs, history_length, frame_dim)` |
| frame order | `source/isaaclab/isaaclab/utils/buffers/circular_buffer.py:105-110` | `CircularBuffer.buffer`: "most recent entry at the end and oldest entry at the beginning" |
| `base_ang_vel` | `source/isaaclab/isaaclab/envs/mdp/observations.py:76` | 3 |
| `projected_gravity` | same, :85 | 3 |
| `joint_pos_rel` | same, :214 | `joint_pos[:, ids] - default_joint_pos[:, ids]` |
| `joint_vel_rel` | same, :258 | |
| `last_action` | same, :891 | `action_name=None` returns the whole `action_manager.action` (15 here, the follow term adds 0) |
| `generated_commands` | same, :909 | `env.command_manager.get_command(command_name)` |
| `UniformNoiseCfg` | `source/isaaclab/isaaclab/utils/noise/noise_cfg.py:41` | Noise levels copied from `velocity_env_cfg.py:167-176` |
| `SceneEntityCfg` | `source/isaaclab/isaaclab/managers/scene_entity_cfg.py:23` | `joint_names` (:43), `body_names` (:77), `preserve_order: bool = False` (:110) |

The drafts set `history_length=5`, `history_order="time"` on the policy group so the exported
layout is "five frames of the 138-value frame, oldest first" (`locomotion/obs_layout.py`). Joint
order inside each term follows `preserve_order=True` and `joint_maps`.

## 6. Rewards, terminations, events, curriculum

All of these are in `source/isaaclab/isaaclab/envs/mdp/` unless noted.

| Function | Where (v3.0.0) | Signature (after `env`) |
| --- | --- | --- |
| `track_lin_vel_xy_exp` | `rewards.py:343` | `std, command_name, asset_cfg` |
| `track_ang_vel_z_exp` | `rewards.py:355` | `std, command_name, asset_cfg` |
| `base_height_l2` | `rewards.py:123` | `target_height, asset_cfg, sensor_cfg=None` ("For flat terrain, target height is in the world frame") |
| `flat_orientation_l2` | `rewards.py:114` | `asset_cfg` |
| `joint_deviation_l1` | `rewards.py:207` | `asset_cfg`; deviation from `default_joint_pos` |
| `joint_torques_l2` | `rewards.py:168` | `asset_cfg` |
| `joint_acc_l2` | `rewards.py:196` | `asset_cfg` |
| `action_rate_l2` | `rewards.py:288` | — |
| `undesired_contacts` | `rewards.py:303` | `threshold, sensor_cfg`; reads `net_normal_forces_w_history` |
| `is_terminated` | `rewards.py:38` | — |
| `feet_slide` | `source/isaaclab_tasks/isaaclab_tasks/core/velocity/mdp/rewards.py:103` | `sensor_cfg, asset_cfg`. **Not in `isaaclab.envs.mdp`**; it lives in the velocity task's `mdp` package, which re-exports `isaaclab.envs.mdp` (`core/velocity/mdp/__init__.pyi`) |
| `time_out` | `terminations.py:32` | — |
| `bad_orientation` | `terminations.py:66` | `limit_angle` (rad, angle between projected gravity and -z) |
| `illegal_contact` | `terminations.py:175` | `threshold, sensor_cfg` |
| `randomize_rigid_body_material` | `events.py:152`, `__call__` `:214` | `static_friction_range, dynamic_friction_range, restitution_range, num_buckets, asset_cfg, make_consistent=False` |
| `randomize_rigid_body_mass` | `events.py:237`, `__call__` `:281` | `asset_cfg, mass_distribution_params, operation ("add" \| "scale" \| "abs"), distribution="uniform", recompute_inertia=True, min_mass=1e-6` |
| `randomize_actuator_gains` | `events.py:629`, `__call__` `:704` | `asset_cfg, stiffness_distribution_params=None, damping_distribution_params=None, operation="abs", distribution="uniform"`; tip: "For implicit actuators, this function uses CPU tensors ... recommended to use this function only during the initialization" |
| `apply_external_force_torque` | `events.py:1177` | `force_range, torque_range, asset_cfg`; samples each axis uniformly and writes `asset.permanent_wrench_composer` (:1209), so the force **stays applied** until the next sample |
| `push_by_setting_velocity` | `events.py:1217` | `velocity_range` dict (m/s, rad/s), adds a sampled velocity to the root |
| `reset_root_state_uniform` | `events.py:1244` (class), `__call__` `:1259` | `pose_range`, `velocity_range` |
| `reset_joints_by_scale` | `events.py:1407` | `position_range, velocity_range` |
| `EventTermCfg` | `managers/manager_term_cfg.py:305` | `mode` (:315), `interval_range_s` (:323) |
| `RewardTermCfg` / `TerminationTermCfg` | `managers/manager_term_cfg.py:384` / `:412` | `weight` (:395) / `time_out` (:423) |
| `CurriculumTermCfg` | `managers/manager_term_cfg.py:137` | `func(env, env_ids, **params)`; the return value is logged |
| `CurriculumManager.compute` | `managers/curriculum_manager.py:124-139` | Calls `term_cfg.func(self._env, env_ids, **term_cfg.params)` |
| curriculum call site | `envs/manager_based_rl_env.py:386` | First line of `_reset_idx`, before managers reset and before `episode_length_buf` is zeroed (:423), so `termination_manager.terminated[env_ids]` (`managers/termination_manager.py:117`) still describes the episode that just ended |
| class-based terms | `managers/manager_base.py:348-353`, `:411-412` | A `ManagerTermBase` subclass as `func` is instantiated with `(cfg=term_cfg, env=env)`; example `DifficultyScheduler` (`envs/mdp/curriculums.py:294`) |
| `ManagerTermBase` | `managers/manager_base.py:27` | `__init__(cfg, env)` (:57), `__call__` (:106) |

**Not found: a push event in newtons.** The docs ask for "random pushes 10 to 30 N every 3 s".
The source offers `push_by_setting_velocity` (a velocity change in m/s) and
`apply_external_force_torque` (a force in N sampled per axis from one `(min, max)` range, held
until the next sample). Neither has a minimum magnitude. The draft uses
`apply_external_force_torque` on `torso_link` with `force_range=(-30.0, 30.0)` every 3 s, so units
match the docs, the 10 N floor is dropped, and the "push" is a 3 s constant force. #5 decides
whether that is the intended disturbance.

**Not found: velocity tracking reward names as in the G1 task.** The G1 tasks use
`track_lin_vel_xy_yaw_frame_exp` and `track_ang_vel_z_world_exp` from the velocity `mdp`
(`core/velocity/mdp/rewards.py:122`, `:134`); `isaaclab.envs.mdp` has the base-frame variants used
by the drafts.

## 7. Agents, registration and the CLI

| Symbol | Where (v3.0.0) | What the drafts rely on |
| --- | --- | --- |
| `RslRlOnPolicyRunnerCfg` | `source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:340` | `actor: RslRlMLPModelCfg` (:346), `critic` (:349), `algorithm: RslRlPpoAlgorithmCfg` (:352); base fields `seed` (:239), `num_steps_per_env` (:245), `max_iterations` (:255), `obs_groups` (:265), `save_interval` (:297), `experiment_name` (:300) |
| `RslRlMLPModelCfg` | same, :22 | `hidden_dims` (:28), `activation` (:31), `obs_normalization` (:34), `distribution_cfg` (:37), `GaussianDistributionCfg(init_std=...)` (:48) |
| `RslRlPpoAlgorithmCfg` | same, :165 | `num_learning_epochs`, `num_mini_batches`, `learning_rate`, `schedule`, `gamma`, `lam`, `entropy_coef`, `desired_kl`, `max_grad_norm`, `value_loss_coef`, `use_clipped_value_loss`, `clip_param` (:171-207) |
| `RslRlPpoActorCriticCfg` | same, :372 | **Deprecated**: "For rsl-rl >= 4.0.0, this configuration is deprecated. Please use `RslRlMLPModelCfg` instead." The runner's `policy` field says the same (:358-362). We pin `rsl-rl-lib` 5.5.1, so the draft uses `actor`/`critic` `RslRlMLPModelCfg`, as `G1FlatPPORunnerCfg` does |
| G1 reference agent | `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/agents/rsl_rl_ppo_cfg.py:12-53` | `num_steps_per_env = 24`, flat: `max_iterations = 1500`, `hidden_dims = [256, 128, 128]` (:50-53); PPO values copied into the draft |
| gym registration | `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/__init__.py:16-26` | `gym.register(id=..., entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True, kwargs={"env_cfg_entry_point": "<module>:<Class>", "rsl_rl_cfg_entry_point": "<module>:<Class>", "default_agent": "rsl_rl"})` |
| task discovery | `source/isaaclab/isaaclab/cli/__init__.py:39`, `:48-55` | `_TASK_ENTRY_POINT_GROUP = "isaaclab.tasks"`; `load_external_tasks()` imports every entry point in that group before `train`/`play`/`zero_agent`/`random_agent` (:221). Documented as `[project.entry-points."isaaclab.tasks"] isaaclab_tutorial = "isaaclab_tutorial.tasks"` (`docs/source/isaaclab/tutorial.rst:111-112`) |
| `isaaclab list_envs` | `source/isaaclab/isaaclab/cli/commands/list_envs.py:19` | Loads the same entry points (:49-50); `--show_presets` lists preset combinations (:34-37); from a project directory it shows that project's tasks by default (:52) |
| `isaaclab train` / `play` | `cli/__init__.py:58-76` → `isaaclab_rl.entrypoints.run_train_cli` / `run_play_cli` (`source/isaaclab_rl/isaaclab_rl/entrypoints/dispatch.py:48`, `:53`) | Flags include `--task`, `--num_envs`, `--max_iterations`, `--seed`, `--rl_library` |
| `isaaclab zero_agent` / `random_agent` | `cli/__init__.py:103-114` → `dispatch.py:69`, `:74` → `entrypoints/simple_agents.py:58` | Flags `--task`, `--num_envs`, `--max_steps` (`simple_agents.py:257-261`); zero agent sends zeros per term (:149-169), random agent sends `U[-1, 1]` (:136-146) |
| `--viz none` | `source/isaaclab/isaaclab/app/sim_launcher.py:190`, `:204`; flag defined as `--visualizer`/`--viz` in `source/isaaclab_physx/isaaclab_physx/app/kit_launcher.py:365-371` | `none` disables visualizers |

**Not found: registration by calling a function.** The issue asks for `register_tasks()` that "scripts
call". The `isaaclab` CLI never calls a function: it imports the modules named in the
`isaaclab.tasks` entry-point group and expects the import to register the IDs. The draft
`tasks/registry.py` therefore registers on import *and* exposes an idempotent `register_tasks()`
for our own scripts. Hooking it into the CLI is one line in `pyproject.toml`
(`[project.entry-points."isaaclab.tasks"] kitchenbot = "kitchenbot.tasks.registry"`), left to #5
because only #5 can check it with `isaaclab list_envs`. Registration uses string entry points, so it
imports only `gymnasium`; the drafts load when a CLI command resolves the config.

## 8. Other 3.0 changes the drafts depend on

- **Data buffers are `ProxyArray`s.** `Articulation.data.*` properties return
  `ProxyArray` (`source/isaaclab/isaaclab/utils/warp/proxy_array.py:28`); tensors are read with
  `.torch` (:122), e.g. `asset.data.default_joint_pos.torch[:, ids]` (`joint_actions.py:194`).
  Arithmetic on a `ProxyArray` still works through a deprecation bridge (:37-38) but warns.
- `ArticulationData.soft_joint_pos_limits` (`base_articulation_data.py:518`): "In torch this
  resolves to (num_instances, num_joints, 2)", order `[lower, upper]`.
- `ArticulationData.default_joint_pos` (`base_articulation_data.py:336`): from `init_state`.
- `ArticulationData.joint_pos` / `joint_vel` (`base_articulation_data.py:964` / `:975`):
  `(num_instances, num_joints)` in the articulation's public joint order.
- `index_fill_` (`source/isaaclab/isaaclab/utils/array.py:47`) is how Isaac Lab fills by
  slice-or-indices; the drafts index with `torch.arange(num_envs)[env_ids]` instead.
- The contrib `assemble_trocar` G1 + Dex3 task lists the Dex3 joints as
  `{left,right}_hand_{thumb_0..2,middle_0..1,index_0..1}_joint`
  (`source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/g129_dex3_env_cfg.py:61-74`), the
  same names as `robots/joint_maps.py`. Independent support for #28's map; #3 still checks the USD.
