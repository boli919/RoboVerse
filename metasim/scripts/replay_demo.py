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
import tyro
from loguru import logger as log
from numpy.typing import NDArray
from rich.logging import RichHandler
from torchvision.utils import make_grid, save_image
from tyro import MISSING
from PIL import Image
import torch
import sapien.core as sapien_core

from metasim.cfg.randomization import RandomizationCfg
from metasim.cfg.render import RenderCfg
from metasim.cfg.robots.base_robot_cfg import BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.sim import HybridSimEnv
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_sim_env_class
from metasim.utils.state import TensorState

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@configclass
class Args:
    task: str = MISSING
    robot: str = "franka"
    scene: str | None = None
    render: RenderCfg = RenderCfg()
    random: RandomizationCfg = RandomizationCfg()

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "sapien3"
    renderer: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] | None = None

    ## Others
    num_envs: int = 1
    try_add_table: bool = True
    object_states: bool = False
    split: Literal["train", "val", "test", "all"] = "all"
    headless: bool = False

    ## Only in args
    save_image_dir: str | None = "tmp"
    save_video_path: str | None = None
    stop_on_runout: bool = False
    
    ## New: Image quality parameters
    camera_width: int = 1920  # Increase camera resolution
    camera_height: int = 1080
    render_mode: Literal["rasterization", "raytracing", "pathtracing"] = "pathtracing"  # Use the highest quality rendering mode
    save_quality: int = 95  # Image save quality (0-100)
    
    ## New: Robot position control
    robot_height_offset: float = 0.0  # Robot initial height offset (meters)
    
    ## New: First-person camera settings
    first_person_view: bool = False  # Enable first-person view
    head_link_name: str = "head_link"  # Robot head link name
    camera_offset: tuple[float, float, float] = (0.1, 0.0, 0.5)  # Camera offset relative to the head (x, y, z)
    camera_direction: tuple[float, float, float] = (1.0, 0.0, -0.577)  # Camera direction vector (forward, right, up). (1, 0, -1) is 45 degrees down.

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


###########################################################
## Utils
###########################################################
def get_actions(all_actions, action_idx: int, num_envs: int, robot: BaseRobotCfg):
    envs_actions = all_actions[:num_envs]
    actions = [
        env_actions[action_idx] if action_idx < len(env_actions) else env_actions[-1] for env_actions in envs_actions
    ]
    return actions


def get_states(all_states, action_idx: int, num_envs: int):
    envs_states = all_states[:num_envs]
    states = [env_states[action_idx] if action_idx < len(env_states) else env_states[-1] for env_states in envs_states]
    return states


def get_runout(all_actions, action_idx: int):
    runout = all([action_idx >= len(all_actions[i]) for i in range(len(all_actions))])
    return runout


class ObsSaver:
    """Save the observations to images or videos."""

    def __init__(self, image_dir: str | None = None, video_path: str | None = None, save_quality: int = 95):
        """Initialize the ObsSaver."""
        self.image_dir = image_dir
        self.video_path = video_path
        self.save_quality = save_quality
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
            # Use PIL to save high-quality images
            image_np = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            pil_image.save(
                os.path.join(self.image_dir, f"rgb_{self.image_idx:04d}.png"),
                quality=self.save_quality,
                optimize=False
            )
            self.image_idx += 1

        image = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        image = (image * 255).astype(np.uint8)
        self.images.append(image)

    def save(self):
        """Save the images or videos."""
        if self.video_path is not None and self.images:
            log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            # Use high-quality video saving parameters
            iio.mimsave(
                self.video_path, 
                self.images, 
                fps=30,
                quality=8,  # High quality setting (0-10)
                codec='libx264' if self.video_path.endswith('.mp4') else None
            )


###########################################################
## Camera Utilities
###########################################################
def update_camera_poses(env, args):
    """
    Update camera poses to follow the robot's head.
    
    Args:
        env: The simulation environment.
        args: Command line arguments.
    """
    if not args.first_person_view:
        return
        
    # Ensure handler exists
    if not hasattr(env, 'handler'):
        log.warning("Environment does not have a handler attribute, cannot update camera poses")
        return
        
    handler = env.handler
    
    # Ensure robot and cameras exist
    if not hasattr(handler, 'robot') or not handler.robot:
        log.warning("Robot not found, cannot update camera poses")
        return
        
    if not hasattr(handler, 'camera_ids') or not handler.camera_ids:
        log.warning("Cameras not found, cannot update camera poses")
        return
    
    # Get the robot head link pose
    robot_name = handler.robot.name
    try:
        # Find the head link
        head_link = None
        for link in handler.link_ids.get(robot_name, []):
            if link.get_name() == args.head_link_name:
                head_link = link
                break
                
        if head_link is None:
            log.warning(f"Head link '{args.head_link_name}' not found, cannot update camera poses")
            # Print all available link names for reference
            available_links = [link.get_name() for link in handler.link_ids.get(robot_name, [])]
            log.info(f"Available links: {available_links}")
            return
            
        # Get the position and orientation of the head link
        head_pose = head_link.get_pose()
        head_pos = head_pose.p
        head_rot = head_pose.q
        
        # Calculate camera position (relative to the head)
        offset = np.array(args.camera_offset)
        
        # Convert the offset from the head's local coordinate system to the world coordinate system
        from scipy.spatial.transform import Rotation as R
        rot = R.from_quat([head_rot[1], head_rot[2], head_rot[3], head_rot[0]])  # Sapien uses w,x,y,z order
        offset_world = rot.apply(offset)
        
        camera_pos = head_pos + offset_world
        
        # Calculate camera direction (relative to the head)
        direction = np.array(args.camera_direction)
        direction_world = rot.apply(direction)
        look_at = camera_pos + direction_world
        
        # Update the poses of all cameras
        for camera_name, camera_id in handler.camera_ids.items():
            handler.set_camera_look_at(camera_name, camera_pos, look_at)
            
        log.debug(f"Updated camera pose: pos={camera_pos}, look_at={look_at}")
        
    except Exception as e:
        log.error(f"Error updating camera poses: {e}")


###########################################################
## Main
###########################################################
def replay_single_trajectory(env, scenario, traj_path, args):
    """Replays a single trajectory file."""
    log.info(f"Replaying trajectory: {traj_path}")
    ## Data
    tic = time.time()
    assert os.path.exists(traj_path), f"Trajectory file: {traj_path} does not exist."
    # Temporarily set the scenario's traj_filepath to the current file for get_traj
    original_traj_filepath = scenario.task.traj_filepath
    scenario.task.traj_filepath = traj_path
    init_states, all_actions, all_states = get_traj(
        scenario.task, scenario.robots[0], env.handler
    )
    scenario.task.traj_filepath = original_traj_filepath  # Restore it
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    obs_saver = ObsSaver(image_dir=args.save_image_dir, video_path=args.save_video_path, save_quality=args.save_quality)

    ## Reset before first step
    tic = time.time()
    
    # Apply robot height offset
    if args.robot_height_offset != 0.0:
        for state in init_states[:args.num_envs]:
            # Assume robot position is in state.root_pos
            if hasattr(state, 'root_pos'):
                state.root_pos[2] += args.robot_height_offset
            # For some robot models, other position parameters may need to be adjusted
            
    obs, extras = env.reset(states=init_states[:args.num_envs])
    
    # If in first-person view mode, update camera poses immediately after reset
    if args.first_person_view:
        update_camera_poses(env, args)
        env.handler.refresh_render()  # Refresh the render to apply camera updates
        
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")
    obs_saver.add(obs)

    ## Main loop
    step = 0
    while True:
        # log.debug(f"Step {step}")
        tic = time.time()
        if args.object_states:
            if all_states is None:
                raise ValueError("All states are None, please check the trajectory file")
            states = get_states(all_states, step, args.num_envs)
            env.handler.set_states(states)
            
            # If in first-person view mode, ensure camera poses are updated
            if args.first_person_view:
                update_camera_poses(env, args)
                
            env.handler.refresh_render()
            obs = env.handler.get_states()

            success = env.handler.task.checker.check(env.handler)
            if success.any():
                log.info(f"Env {success.nonzero().squeeze(-1).tolist()} succeeded!")
            if success.all():
                break

        else:
            actions = get_actions(all_actions, step, args.num_envs, scenario.robots[0])
            
            # Debugging info
            log.info(f"Step {step}: episode_length_buf={env.episode_length_buf}, episode_length={env.handler.scenario.episode_length}")
            
            obs, reward, success, time_out, extras = env.step(actions)
            
            # If in first-person view mode, ensure camera poses are updated
            if args.first_person_view:
                update_camera_poses(env, args)

            if success.any():
                log.info(f"Env {success.nonzero().squeeze(-1).tolist()} succeeded!")

            if time_out.any():
                log.info(f"Env {time_out.nonzero().squeeze(-1).tolist()} timed out!")
                log.info(f"After step: episode_length_buf={env.episode_length_buf}, episode_length={env.handler.scenario.episode_length}")

            if success.all() or time_out.all():
                break

        toc = time.time()
        log.trace(f"Time to step: {toc - tic:.2f}s")

        tic = time.time()
        obs_saver.add(obs)
        toc = time.time()
        log.trace(f"Time to save obs: {toc - tic:.2f}s")
        step += 1

        if args.stop_on_runout and get_runout(all_actions, step):
            log.info("Run out of actions, stopping")
            break

    obs_saver.save()


def main():
    # Set high-quality rendering mode
    render_cfg = RenderCfg(mode=args.render_mode)
    
    # Create camera configurations
    if args.first_person_view:
        # First-person camera - will be dynamically updated in the simulation
        camera = PinholeCameraCfg(
            pos=(0.0, 0.0, 0.0),  # Initial position will be updated in the simulation
            look_at=(1.0, 0.0, 0.0),  # Initial look_at will be updated in the simulation
            width=args.camera_width,
            height=args.camera_height,
            focal_length=10.0,
            horizontal_aperture=20.955
        )
        log.info(f"Enabling first-person view mode, attaching to link: {args.head_link_name}")
        log.info(f"Camera offset: {args.camera_offset}, direction: {args.camera_direction}")
    else:
        # Standard fixed camera
        # To make the camera face the ground at a 45-degree angle, the z-coordinate (height) of the camera's position
        # must be equal to the distance between the camera and the look_at point in the xy-plane.
        # Example: pos=(3, 0, 3) and look_at=(0, 0, 0)
        # Height = pos.z - look_at.z = 3 - 0 = 3
        # XY-plane distance = sqrt((pos.x-look_at.x)^2 + (pos.y-look_at.y)^2) = sqrt((3-0)^2 + (0-0)^2) = 3
        # Since Height == XY-plane distance, the angle with the horizontal plane is 45 degrees.
        #
        # You can modify the pos value below, just keep the z value equal to the xy distance to maintain the angle.
        # For example, let's move the camera a bit closer while keeping the 45-degree angle:
        camera_pos = (2.5, 0.0, 2.5)
        look_at_pos = (-3.0, 0.0, 0.0)
        
        camera = PinholeCameraCfg(
            pos=camera_pos, 
            look_at=look_at_pos,
            width=args.camera_width,
            height=args.camera_height,
            focal_length=10.0,
            horizontal_aperture=20.955
        )
    
    # Display image quality settings
    log.info(f"Image Quality Settings:")
    log.info(f"  Camera Resolution: {args.camera_width}x{args.camera_height}")
    log.info(f"  Render Mode: {args.render_mode}")
    log.info(f"  Save Quality: {args.save_quality}/100")
    
    scenario = ScenarioCfg(
        task=args.task,
        robots=[args.robot],
        scene=args.scene,
        cameras=[camera],
        random=args.random,
        render=render_cfg,  # Use high-quality render config
        sim=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        try_add_table=args.try_add_table,
        object_states=args.object_states,
        split=args.split,
        headless=args.headless,
    )

    tic = time.time()
    if scenario.renderer is None:
        log.info(f"Using simulator: {scenario.sim}")
        env_class = get_sim_env_class(SimType(scenario.sim))
        env = env_class(scenario)
    else:
        log.info(f"Using simulator: {scenario.sim}, renderer: {scenario.renderer}")
        env_class_render = get_sim_env_class(SimType(scenario.renderer))
        env_render = env_class_render(scenario)
        env_class_physics = get_sim_env_class(SimType(scenario.sim))
        env_physics = env_class_physics(scenario)
        env = HybridSimEnv(env_physics, env_render)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    traj_filepaths = scenario.task.traj_filepath
    if not isinstance(traj_filepaths, list):
        traj_filepaths = [traj_filepaths]

    for traj_path in traj_filepaths:
        replay_single_trajectory(env, scenario, traj_path, args)

    env.close()


if __name__ == "__main__":
    main()
