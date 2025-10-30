from __future__ import annotations

import logging
import os
import time
from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import imageio as iio
import numpy as np
import torch
import tyro
from loguru import logger as log
from numpy.typing import NDArray
from rich.logging import RichHandler
from torchvision.utils import make_grid, save_image
from tyro import MISSING

from metasim.cfg.randomization import RandomizationCfg
from metasim.cfg.render import RenderCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.sim import HybridSimEnv
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_env_class
from metasim.utils.state import TensorState, state_tensor_to_nested
from motion_lib import MotionLib

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@configclass
class Args:
    motion_file: str = MISSING
    robot: str = "g1"
    scene: str | None = None
    render: RenderCfg = RenderCfg(mode="raytracing")
    random: RandomizationCfg = RandomizationCfg()

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "sapien3"
    renderer: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] = "sapien3"

    ## Others
    num_envs: int = 1
    try_add_table: bool = True
    headless: bool = False

    ## Only in args
    save_image_dir: str | None = "tmp"
    save_video_path: str | None = None
    loop: bool = False

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


###########################################################
## Utils
###########################################################
class ObsSaver:
    """Save the observations to images or videos."""

    def __init__(self, image_dir: str | None = None, video_path: str | None = None):
        """Initialize the ObsSaver."""
        self.image_dir = image_dir
        self.video_path = video_path
        self.images: list[NDArray] = []

        self.image_idx = 0

    def add(self, state: TensorState):
        """Add the observation to the list."""
        if self.image_dir is None and self.video_path is None:
            return

        try:
            rgb_data = next(iter(state.cameras.values())).rgb
            image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))  # (C, H, W)
        except Exception as e:
            log.error(f"Error adding observation: {e}")
            return

        if self.image_dir is not None:
            os.makedirs(self.image_dir, exist_ok=True)
            save_image(image, os.path.join(self.image_dir, f"rgb_{self.image_idx:04d}.png"))
            self.image_idx += 1

        image = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        image = (image * 255).astype(np.uint8)
        self.images.append(image)

    def save(self):
        """Save the images or videos."""
        if self.video_path is not None and self.images:
            log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(self.video_path, self.images, fps=30)


###########################################################
## Main
###########################################################
def main():
    camera = PinholeCameraCfg(pos=(3.0, -3.0, 1.5), look_at=(0.0, 0.0, 1.0))
    scenario = ScenarioCfg(
        task=None,
        robots=[args.robot],
        scene=args.scene,
        cameras=[camera],
        random=args.random,
        render=args.render,
        sim=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        try_add_table=args.try_add_table,
        headless=args.headless,
    )

    num_envs: int = scenario.num_envs

    tic = time.time()
    if scenario.renderer is None:
        log.info(f"Using simulator: {scenario.sim}")
        env_class = get_sim_env_class(SimType(scenario.sim))
        env = env_class(scenario)
    else:
        log.info(f"Using simulator: {scenario.sim}, renderer: {scenario.renderer}")
        env_class_render = get_sim_env_class(SimType(scenario.renderer))
        env_render = env_class_render(scenario)  # Isaaclab must launch right after import
        env_class_physics = get_sim_env_class(SimType(scenario.sim))
        env_physics = env_class_physics(scenario)  # Isaaclab must launch right after import
        env = HybridSimEnv(env_physics, env_render)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    ## Data
    tic = time.time()
    # Use MotionLib to load and process the motion file
    log.info(f"Loading motion file: {args.motion_file}")
    motion_lib = MotionLib(motion_file=args.motion_file, device=env.handler.device)
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    total_len = motion_lib.get_total_length()
    log.info(f"Motion total length: {total_len:.3f}s")

    # Calculate z-offset to place the robot on the ground
    all_root_pos = motion_lib._motion_root_pos
    min_z = all_root_pos[:, 2].min()
    z_offset = min_z.clone() - 0.01  # Add a small epsilon to avoid ground penetration
    log.info(f"Applying z-offset: {z_offset.item():.3f} to root position.")

    ########################################################
    ## Joint Mapping
    ########################################################
    # Use simulator joint order (URDF order) for mapping to ensure consistency with DOF indices
    sim_joint_names = env.handler.get_joint_names(args.robot, sort=False)
    log.info(f"Simulator '{args.robot}' joint names ({len(sim_joint_names)}): {sim_joint_names}")

    # This is the assumed joint order from the motion file (humanml3d)
    motion_joint_names = [
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "waist_yaw", "waist_roll", "waist_pitch",
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"
    ]
    log.info(f"Motion file joint names ({len(motion_joint_names)}): {motion_joint_names}")

    # Create mapping from motion file order to simulator order
    motion_name_to_idx = {name: i for i, name in enumerate(motion_joint_names)}
    joint_mapping = [motion_name_to_idx[name.replace("_joint", "")] for name in sim_joint_names]
    joint_mapping = torch.tensor(joint_mapping, device=env.handler.device, dtype=torch.long)

    ########################################################
    ## Main
    ########################################################

    obs_saver = ObsSaver(image_dir=args.save_image_dir, video_path=args.save_video_path)

    ## Reset before first step
    tic = time.time()
    obs, extras = env.reset()
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")

    robot_name = args.robot
    motion_id = 0  # Assuming only one motion in the lib for now

    motion_time = 0.0
    fps = motion_lib._motion_fps[motion_id].item()
    dt = 1.0 / fps

    ## Main loop
    step = 0
    while motion_time < total_len:
        log.debug(f"Step {step}, time: {motion_time:.3f}s")
        tic = time.time()

        motion_ids_tensor = torch.full((num_envs,), motion_id, device=env.handler.device, dtype=torch.long)
        motion_times_tensor = torch.full((num_envs,), motion_time, device=env.handler.device)

        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = motion_lib.calc_motion_frame(
            motion_ids=motion_ids_tensor, motion_times=motion_times_tensor
        )

        # Apply z-offset to root position
        root_pos[:, 2] -= z_offset

        # Reorder dof tensors to match the simulator's expected joint order
        dof_pos_mapped = dof_pos[:, joint_mapping]
        dof_vel_mapped = dof_vel[:, joint_mapping]

        robot_state = obs.robots[robot_name]
        robot_state.joint_pos = dof_pos_mapped
        robot_state.joint_vel = dof_vel_mapped
        robot_state.root_state = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1).to(
            robot_state.root_state.device
        )

        nested_states = state_tensor_to_nested(env.handler, obs)
        env.handler.set_states(nested_states)

        # Explicitly set camera pose from config to make user changes effective
        if scenario.cameras:
            camera_cfg = scenario.cameras[0]
            env.handler.set_camera_look_at(camera_cfg.name, pos=camera_cfg.pos, look_at=camera_cfg.look_at)

        env.handler.refresh_render()
        obs = env.handler.get_states()
        toc = time.time()
        log.trace(f"Time to step: {toc - tic:.2f}s")

        tic = time.time()
        obs_saver.add(obs)
        toc = time.time()
        log.trace(f"Time to save obs: {toc - tic:.2f}s")

        step += 1
        motion_time += 1 / fps
        if motion_time >= total_len:
            if args.loop:
                motion_time = 0.0
            else:
                log.info("Motion finished.")
                break

    obs_saver.save()
    env.close()


if __name__ == "__main__":
    main()
