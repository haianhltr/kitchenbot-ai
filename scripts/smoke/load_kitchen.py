# UNTESTED DRAFT (#59): written against Isaac Lab v3.0.0 source without a simulator; #10 and #11 run and fix it.  # noqa: E501
"""Load the kitchen scene layers, step them with scripted door and rack targets, print FPS.

For #10 (``--layers counter``, FPS at 1 and 64 envs, both base modes) and #11 (``--layers
counter,dishwasher``: door and rack follow scripted targets, slot poses against the constants)::

    isaaclab -p scripts/smoke/load_kitchen.py --layers counter --num_envs 1 --headless
    isaaclab -p scripts/smoke/load_kitchen.py --layers counter,dishwasher --num_envs 64 --headless
    isaaclab -p scripts/smoke/load_kitchen.py --layers counter,dishwasher,plates --fixed-base

What it does: builds ``kitchen_scene_cfg.make_kitchen_scene_cfg`` with PhysX (``PhysxCfg``, the
``isaacsim_physx`` backend; this script uses the standalone launcher, not the ``isaaclab`` task CLI,
so the physics is set in ``SimulationCfg`` here), writes every articulation's default root and
joint state, then steps ``--steps`` frames at 200 Hz. When the dishwasher is loaded the door target
swings between 90 and 60 deg and the lower-rack target between -0.2 and 0 m (both cosines, period
``SWING_PERIOD_S``), written through ``actuators.target_command.set_position_index`` (the 3.0 path,
docs/isaaclab_3_api_notes.md section 3). It prints the wall-clock frame rate and environment-steps
per second (after ``WARMUP_STEPS``), the last door and rack positions against their targets, and the
largest distance between the slot poses composed from the simulated rack link and the
``kitchen_dims`` constants at the measured rack position (#11's acceptance: 1 cm).

Without ``--fixed-base`` the G1 has no balance policy here and falls within a second or two; that
does not affect the scene check, and ``--fixed-base`` gives the configuration a stable frame rate.
``--headless`` is forwarded to Kit's ``SimulationApp`` through the launcher arguments (3.0.0 has no
``--headless`` flag of its own; ``kit_launcher.py:48``, ``:498-504``).
"""

import argparse
import math
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Load kitchen scene layers and report FPS (#59).")
parser.add_argument("--layers", default="counter,dishwasher,plates", help="comma-separated")
parser.add_argument("--num_envs", type=int, default=1, help="number of environments")
parser.add_argument("--steps", type=int, default=500, help="physics frames to step")
parser.add_argument("--fixed-base", action="store_true", help="pin the G1 pelvis")
parser.add_argument("--headless", action="store_true", help="run Kit without a window")
parser.add_argument("--seed", type=int, default=0, help="torch seed (nothing is random yet)")
add_launcher_args(parser)
args_cli = parser.parse_args()

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import instantiate  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402

from kitchenbot.scene import dishwasher_cfg as dwc  # noqa: E402
from kitchenbot.scene import dishwasher_frames as df  # noqa: E402
from kitchenbot.scene import kitchen_dims as kd  # noqa: E402
from kitchenbot.scene import kitchen_scene_cfg as ksc  # noqa: E402

PHYSICS_DT = 1.0 / 200.0  # docs/architecture.md section 5: 200 Hz physics
WARMUP_STEPS = 20
SWING_PERIOD_S = 2.5
DOOR_SWING_DEG = (kd.DISHWASHER_DOOR_OPEN_DEG, 60.0)
RACK_SWING_M = (kd.DISHWASHER_RACK_LIMITS_M[0], 0.0)


def swing(t: float, ends: tuple[float, float]) -> float:
    """Cosine between ``ends[0]`` (at t = 0) and ``ends[1]`` (half a period later)."""
    w = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / SWING_PERIOD_S))
    return ends[0] + w * (ends[1] - ends[0])


def reset_articulations(scene) -> None:
    """Write every articulation's default root pose (plus the env origins) and joint state."""
    for articulation in scene.articulations.values():
        root_pose = articulation.data.default_root_pose.torch.clone()
        root_pose[:, :3] += scene.env_origins
        articulation.write_root_pose_to_sim_index(root_pose=root_pose)
        articulation.write_root_velocity_to_sim_index(
            root_velocity=articulation.data.default_root_vel.torch.clone()
        )
        articulation.write_joint_position_to_sim_index(
            position=articulation.data.default_joint_pos.torch.clone()
        )
        articulation.write_joint_velocity_to_sim_index(
            velocity=articulation.data.default_joint_vel.torch.clone()
        )
    scene.reset()


def main() -> None:
    torch.manual_seed(args_cli.seed)
    layers = ksc.parse_layers(args_cli.layers)
    sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device, physics=PhysxCfg())
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view([2.5, -2.5, 2.0], [0.3, -0.2, 0.8])
        scene_cfg = ksc.make_kitchen_scene_cfg(
            num_envs=args_cli.num_envs, layers=layers, fixed_base=args_cli.fixed_base
        )
        scene = instantiate(scene_cfg)
        sim.reset()
        reset_articulations(scene)
        print(
            f"[load_kitchen] layers={','.join(layers)} num_envs={scene.num_envs} "
            f"fixed_base={args_cli.fixed_base} dt={PHYSICS_DT}"
        )

        dishwasher = scene["dishwasher"] if "dishwasher" in layers else None
        if dishwasher is not None:
            door_ids, _ = dishwasher.find_joints(df.DOOR_JOINT)
            rack_ids, _ = dishwasher.find_joints(df.LOWER_RACK_JOINT)
            print(
                f"[load_kitchen] dishwasher joints {dishwasher.joint_names}, "
                f"bodies {dishwasher.body_names}"
            )

        n = scene.num_envs
        start = None
        door_target = rack_target = 0.0
        for step in range(args_cli.steps):
            if step == WARMUP_STEPS:
                start = time.perf_counter()
            if dishwasher is not None:
                t = step * PHYSICS_DT
                door_target = math.radians(swing(t, DOOR_SWING_DEG))
                rack_target = swing(t, RACK_SWING_M)
                command = dishwasher.actuators.target_command
                command.set_position_index(
                    value=torch.full((n, 1), door_target, device=sim.device), joint_ids=door_ids
                )
                command.set_position_index(
                    value=torch.full((n, 1), rack_target, device=sim.device), joint_ids=rack_ids
                )
            scene.write_data_to_sim()
            sim.step(render=not args_cli.headless)
            scene.update(PHYSICS_DT)
        elapsed = time.perf_counter() - start if start is not None else float("nan")
        timed = max(args_cli.steps - WARMUP_STEPS, 0)
        fps = timed / elapsed if elapsed > 0 else float("nan")
        print(
            f"[load_kitchen] {timed} frames in {elapsed:.2f} s: {fps:.1f} FPS, "
            f"{fps * n:.0f} env-steps/s at num_envs={n}"
        )

        if dishwasher is not None:
            q = dishwasher.data.joint_pos.torch
            door = q[:, door_ids[0]]
            rack = q[:, rack_ids[0]]
            print(
                f"[load_kitchen] door {math.degrees(door.mean()):.1f} deg (target "
                f"{math.degrees(door_target):.1f}), lower rack {rack.mean():.3f} m (target "
                f"{rack_target:.3f}), env spread door {math.degrees(door.max() - door.min()):.2f} "
                "deg"
            )
            measured = dwc.slot_poses_from_sim(dishwasher)  # (n, 5, 3)
            expected = torch.stack(
                [
                    dwc.slot_poses_w(scene.env_origins[i : i + 1], float(rack[i]))[0, :, :3]
                    for i in range(n)
                ]
            )
            error = (measured - expected).norm(dim=-1)
            print(
                f"[load_kitchen] slot poses: sim vs kitchen_dims max {error.max() * 100:.2f} cm "
                f"(acceptance 1 cm), slot 3 env 0 at {measured[0, 2].tolist()}"
            )


if __name__ == "__main__":
    main()
