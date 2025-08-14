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
    
    ## 新增：图片质量相关参数
    camera_width: int = 1024  # 提升相机分辨率
    camera_height: int = 1024
    render_mode: Literal["rasterization", "raytracing", "pathtracing"] = "pathtracing"  # 使用最高质量渲染模式
    save_quality: int = 95  # 图片保存质量 (0-100)

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
            # 使用PIL保存高质量图片
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
            # 使用高质量视频保存参数
            iio.mimsave(
                self.video_path, 
                self.images, 
                fps=30,
                quality=8,  # 高质量设置 (0-10)
                codec='libx264' if self.video_path.endswith('.mp4') else None
            )


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
    obs, extras = env.reset(states=init_states[:args.num_envs])
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
            env.handler.refresh_render()
            obs = env.handler.get_states()

            success = env.handler.task.checker.check(env.handler)
            if success.any():
                log.info(f"Env {success.nonzero().squeeze(-1).tolist()} succeeded!")
            if success.all():
                break

        else:
            actions = get_actions(all_actions, step, args.num_envs, scenario.robots[0])
            
            # 调试信息
            log.info(f"Step {step}: episode_length_buf={env.episode_length_buf}, episode_length={env.handler.scenario.episode_length}")
            
            obs, reward, success, time_out, extras = env.step(actions)

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
    # 设置高质量渲染模式
    render_cfg = RenderCfg(mode=args.render_mode)
    
    # 创建高分辨率相机配置
    camera = PinholeCameraCfg(
        pos=(3, 0, 3), 
        look_at=(0.0, 0.0, 0.0),
        width=args.camera_width,
        height=args.camera_height,
        focal_length=24.0,  # 保持默认焦距
        horizontal_aperture=20.955  # 保持默认光圈
    )
    
    # 显示图片质量设置信息
    log.info(f"图片质量设置:")
    log.info(f"  相机分辨率: {args.camera_width}x{args.camera_height}")
    log.info(f"  渲染模式: {args.render_mode}")
    log.info(f"  保存质量: {args.save_quality}/100")
    
    scenario = ScenarioCfg(
        task=args.task,
        robots=[args.robot],
        scene=args.scene,
        cameras=[camera],
        random=args.random,
        render=render_cfg,  # 使用高质量渲染配置
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
