# UNTESTED DRAFT (#46 and #68): written against Isaac Lab v3.0.0 source without a simulator; #5 runs and fixes it.  # noqa: E501
"""``KitchenBot-Stand-G1-Dex3``: the balance-policy training environment (architecture s. 6).

Structure follows Isaac Lab's velocity task (``core/velocity/velocity_env_cfg.py`` and the G1 flat
config), with the differences section 6 lists: G1 + Dex3 asset, flat plane, 15 policy actions over
legs and waist, the upper body driven by ``UpperBodyCommand`` through ``UpperBodyFollowAction``,
a 5-frame time-major observation history laid out by ``obs_layout``, a standing-heavy velocity
command, a commanded forward waist pitch that the policy observes and the waist reward tracks
(``WaistLeanCommand``, D012, #68), and the section 6 randomization. Every Isaac Lab symbol used
here is listed with its v3.0.0 file and line in docs/isaaclab_3_api_notes.md. Reward weights are
copied from the G1 flat and rough configs where the term exists there and are provisional
otherwise; #5 tunes them.

Physics: ``StandPhysicsCfg`` declares only ``isaacsim_physx`` and makes it the default, so
``physics=isaacsim_physx`` on the command line resolves and forgetting it still runs PhysX.
200 Hz physics (``dt = 0.005``), decimation 4, 50 Hz policy, 20 s episodes.
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab_tasks.core.velocity.mdp as velocity_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
from isaaclab_tasks.utils import PresetCfg

from kitchenbot.locomotion import obs_layout
from kitchenbot.locomotion.commands import UpperBodyCommandCfg, WaistLeanCommandCfg
from kitchenbot.locomotion.curriculums import UpperBodyRangeCurriculum
from kitchenbot.locomotion.rewards import feet_planted_when_standing, waist_tracking_l1
from kitchenbot.locomotion.upper_body_action import UpperBodyFollowActionCfg
from kitchenbot.robots.g1_dex3 import G1_DEX3_CFG
from kitchenbot.robots.joint_maps import WAIST

EPISODE_LENGTH_S = 20.0
LOWER = list(obs_layout.LOWER_BODY_JOINTS)
UPPER = list(obs_layout.UPPER_BODY_JOINTS)
FEET = ".*_ankle_roll_link"
KNEES = ".*_knee_link"


def _joints(names: list[str]) -> SceneEntityCfg:
    """The robot restricted to ``names``, in that order."""
    return SceneEntityCfg("robot", joint_names=names, preserve_order=True)


def _contacts(bodies: str | list[str]) -> SceneEntityCfg:
    return SceneEntityCfg("contact_forces", body_names=bodies)


@configclass
class StandPhysicsCfg(PresetCfg):
    """Physics presets: Isaac Sim PhysX only (D003); also the default."""

    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    default = isaacsim_physx


@configclass
class StandSceneCfg(InteractiveSceneCfg):
    """Flat floor, one G1 + Dex3, a contact sensor on every robot link, a dome light."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=PhysxRigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot: ArticulationCfg = G1_DEX3_CFG
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3, track_air_time=True
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0)
    )


@configclass
class CommandsCfg:
    """Planar velocity (60 percent of episodes exactly zero), the upper-body targets and the
    forward waist pitch (D012: zero for half of the targets, a 1 rad/s ramp to each)."""

    # One resample per episode, so ``rel_standing_envs`` is a per-episode fraction.
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S),
        rel_standing_envs=0.6,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.3, 0.3)
        ),
    )
    upper_body = UpperBodyCommandCfg(asset_name="robot", joint_names=UPPER)
    waist_lean = WaistLeanCommandCfg(asset_name="robot")


@configclass
class ActionsCfg:
    """15 policy actions around the default pose; the upper body follows the command."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LOWER,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )
    upper_body = UpperBodyFollowActionCfg(
        asset_name="robot", joint_names=UPPER, command_name="upper_body"
    )


@configclass
class ObservationsCfg:
    """One policy group; term order and widths are ``obs_layout.FRAME_TERMS``."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        waist_lean_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "waist_lean"}
        )
        lower_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _joints(LOWER)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        lower_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _joints(LOWER)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=mdp.last_action)
        upper_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _joints(UPPER)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        upper_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _joints(UPPER)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        upper_targets = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "upper_body"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = obs_layout.HISTORY_LENGTH
            self.flatten_history_dim = True
            self.history_order = "time"

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Section 6 rewards. Weights: G1 flat/rough values where they exist, else provisional."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.75})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # D012: tracks the waist lean command (pitch) and holds yaw and roll at zero; provisional weight
    waist_deviation = RewTerm(
        func=waist_tracking_l1,
        weight=-0.1,
        params={"command_name": "waist_lean", "asset_cfg": _joints(list(WAIST))},
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-2.0e-6, params={"asset_cfg": _joints(LOWER)}
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-1.0e-7, params={"asset_cfg": _joints(LOWER)}
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    feet_slide = RewTerm(
        func=velocity_mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": _contacts(FEET),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": _contacts([KNEES, "torso_link"]), "threshold": 1.0},
    )
    feet_planted = RewTerm(
        func=feet_planted_when_standing,
        weight=0.25,
        params={"command_name": "base_velocity", "sensor_cfg": _contacts(FEET)},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
    """Torso or pelvis on the ground, tilt over 60 degrees, 20 s time-out."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": _contacts(["pelvis", "torso_link"]), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}
    )


@configclass
class EventsCfg:
    """Section 6 randomization plus the usual resets."""

    # startup: friction 0.6 to 1.0, base mass -1 to +3 kg, gains x0.9 to x1.1
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    # interval: a force of up to 30 N per axis on the torso, redrawn every 3 s (no 10 N floor in
    # Isaac Lab; see the notes file, section 6)
    push_robot = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(3.0, 3.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (-30.0, 30.0),
            "torque_range": (0.0, 0.0),
        },
    )


@configclass
class CurriculumCfg:
    """Upper-body range factor 0.2 to 1.0 as the fall rate drops."""

    upper_body_range = CurrTerm(
        func=UpperBodyRangeCurriculum, params={"command_name": "upper_body"}
    )


@configclass
class StandEnvCfg(ManagerBasedRLEnvCfg):
    """``KitchenBot-Stand-G1-Dex3`` training configuration."""

    sim: SimulationCfg = SimulationCfg(physics=StandPhysicsCfg())
    scene: StandSceneCfg = StandSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt

    def play_mode(self):
        """Isaac Lab 3.0 play hook (``manager_based_rl_env_cfg.py:101``): no pushes, no
        curriculum, the full upper-body range from the start."""
        super().play_mode()
        self.events.push_robot = None
        self.curriculum.upper_body_range = None
        self.commands.upper_body.initial_range_factor = 1.0


@configclass
class StandEnvCfg_PLAY(StandEnvCfg):
    """``StandEnvCfg`` with ``play_mode()`` applied, for scripts that want a class."""

    def __post_init__(self):
        super().__post_init__()
        self.play_mode()
