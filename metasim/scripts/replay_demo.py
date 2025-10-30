from __future__ import annotations

import logging
import os
import sys
import pickle
import time
from typing import Literal, List
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
motion_tools_path = project_root / "motion_tools"
if motion_tools_path.exists():
    sys.path.insert(0, str(project_root))

try:
    import isaacgym
except ImportError:
    pass

import imageio as iio
import numpy as np
import tyro
import yaml
from loguru import logger as log
from numpy.typing import NDArray
from rich.logging import RichHandler
from torchvision.utils import make_grid
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

if motion_tools_path.exists():
    log.info(f"Added project root to sys.path: {project_root}")
    log.info(f"motion_tools found at: {motion_tools_path}")
else:
    log.warning(f"motion_tools not found at {motion_tools_path}")
    log.warning("dof_rot computation will be skipped if motion_tools is needed")


@configclass
class Args:
    task: str = MISSING
    robot: str = "franka"
    scene: str | None = None
    render: RenderCfg = RenderCfg()
    random: RandomizationCfg = RandomizationCfg()

    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "sapien3"
    renderer: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] | None = None

    num_envs: int = 1
    try_add_table: bool = True
    object_states: bool = False
    split: Literal["train", "val", "test", "all"] = "all"
    headless: bool = False

    save_image_dir: str | None = "tmp"
    save_video_path: str | None = None
    stop_on_runout: bool = False
    
    camera_width: int = 1920
    camera_height: int = 1080
    render_mode: Literal["rasterization", "raytracing", "pathtracing"] = "pathtracing"
    save_quality: int = 95
    
    robot_height_offset: float = 0.0
    
    first_person_view: bool = False
    head_link_name: str = "head_link"
    camera_offset: tuple[float, float, float] = (0.1, 0.0, 0.5)
    camera_direction: tuple[float, float, float] = (1.0, 0.0, -0.577)
    
    motion_config: str | None = None
    save_enhanced_pkl_dir: str | None = None
    robot_urdf: str | None = None
    image_size: int = 224

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


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


def quat_xyzw_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (xyzw format) to rotation matrix."""
    q = q / q.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-9)
    x, y, z, w = q.unbind(-1)
    
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)

    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - wx)

    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = 1 - 2 * (xx + yy)

    mat = torch.stack([
        torch.stack([m00, m01, m02], dim=-1),
        torch.stack([m10, m11, m12], dim=-1),
        torch.stack([m20, m21, m22], dim=-1),
    ], dim=-2)
    return mat


def get_actuated_joint_indices(kin) -> List[int]:
    """Get indices of actuated joints."""
    idxs: List[int] = []
    for j in range(1, kin.num_joint):
        if kin.joint_dof_idx[j] != -1:
            idxs.append(j)
    return idxs


class EnhancedObsSaver:
    """Save observations and prepare data for enhanced pkl export."""

    def __init__(self, image_dir: str | None = None, video_path: str | None = None, 
                 save_quality: int = 95, image_size: int = 224):
        self.image_dir = image_dir
        self.video_path = video_path
        self.save_quality = save_quality
        self.image_size = image_size
        self.images: list[NDArray] = []
        self.resized_images: list[NDArray] = []
        self.image_idx = 0

    def add(self, state: TensorState):
        """Add observation to the list."""
        if self.image_dir is None and self.video_path is None:
            return

        try:
            rgb_data = next(iter(state.cameras.values())).rgb
            image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))
        except Exception as e:
            log.error(f"Error adding observation: {e}")
            return

        if self.image_dir is not None:
            os.makedirs(self.image_dir, exist_ok=True)
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            pil_image.save(
                os.path.join(self.image_dir, f"rgb_{self.image_idx:04d}.png"),
                quality=self.save_quality,
                optimize=False
            )
            self.image_idx += 1

        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        self.images.append(image)
        
        resized = Image.fromarray(image).resize((self.image_size, self.image_size), Image.LANCZOS)
        self.resized_images.append(np.asarray(resized, dtype=np.uint8))

    def save(self):
        """Save video if path is specified."""
        if self.video_path is not None and self.images:
            log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(
                self.video_path, 
                self.images, 
                fps=30,
                quality=8,
                codec='libx264' if self.video_path.endswith('.mp4') else None
            )


def update_camera_poses(env, args):
    """Update camera poses to follow the robot's head."""
    if not args.first_person_view:
        return
        
    if not hasattr(env, 'handler'):
        log.warning("Environment does not have a handler attribute, cannot update camera poses")
        return
        
    handler = env.handler
    
    if not hasattr(handler, 'robot') or not handler.robot:
        log.warning("Robot not found, cannot update camera poses")
        return
        
    if not hasattr(handler, 'camera_ids') or not handler.camera_ids:
        log.warning("Cameras not found, cannot update camera poses")
        return
    
    robot_name = handler.robot.name
    try:
        head_link = None
        for link in handler.link_ids.get(robot_name, []):
            if link.get_name() == args.head_link_name:
                head_link = link
                break
                
        if head_link is None:
            log.warning(f"Head link '{args.head_link_name}' not found, cannot update camera poses")
            available_links = [link.get_name() for link in handler.link_ids.get(robot_name, [])]
            log.info(f"Available links: {available_links}")
            return
            
        head_pose = head_link.get_pose()
        head_pos = head_pose.p
        head_rot = head_pose.q
        
        offset = np.array(args.camera_offset)
        
        from scipy.spatial.transform import Rotation as R
        rot = R.from_quat([head_rot[1], head_rot[2], head_rot[3], head_rot[0]])
        offset_world = rot.apply(offset)
        
        camera_pos = head_pos + offset_world
        
        direction = np.array(args.camera_direction)
        direction_world = rot.apply(direction)
        look_at = camera_pos + direction_world
        
        for camera_name, camera_id in handler.camera_ids.items():
            handler.set_camera_look_at(camera_name, camera_pos, look_at)
            
        log.debug(f"Updated camera pose: pos={camera_pos}, look_at={look_at}")
        
    except Exception as e:
        log.error(f"Error updating camera poses: {e}")


def load_motion_data(motion_config_path: str):
    """Load motion data from yaml config."""
    with open(motion_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    motion_file = config.get('motion_file')
    if motion_file:
        log.info(f"Loading motion data from {motion_file}")
        with open(motion_file, 'rb') as f:
            motion_data = pickle.load(f)
        return motion_data
    
    root_path = config.get('root_path', '')
    motions = config.get('motions', [])
    
    if not motions:
        raise ValueError(f"No motion_file or motions specified in {motion_config_path}")
    
    motion_file_relative = motions[0].get('file')
    if not motion_file_relative:
        raise ValueError(f"No file specified in first motion entry")
    
    motion_file = os.path.join(root_path, motion_file_relative)
    
    log.info(f"Loading motion data from {motion_file}")
    with open(motion_file, 'rb') as f:
        motion_data = pickle.load(f)
    
    return motion_data


def save_enhanced_pkl(output_path: str, motion_data: dict, resized_images: list, 
                     urdf_path: str, device: str = "cpu"):
    """Save enhanced pkl with dof_rot and resized rgb images."""
    log.info(f"Preparing enhanced pkl data...")
    
    if urdf_path and not os.path.isabs(urdf_path):
        urdf_path_abs = os.path.join(project_root, urdf_path)
        if os.path.exists(urdf_path_abs):
            urdf_path = urdf_path_abs
            log.info(f"Resolved relative URDF path to: {urdf_path}")
        else:
            urdf_path_cwd = os.path.join(os.getcwd(), urdf_path)
            if os.path.exists(urdf_path_cwd):
                urdf_path = urdf_path_cwd
                log.info(f"Resolved URDF path relative to cwd: {urdf_path}")
    
    log.info(f"URDF path: {urdf_path}")
    log.info(f"URDF exists: {os.path.exists(urdf_path) if urdf_path else False}")
    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Project root: {project_root}")
    
    required_keys = ["root_pos", "root_rot", "dof_pos"]
    for key in required_keys:
        if key not in motion_data:
            raise ValueError(f"Missing required key '{key}' in motion data")

    root_pos = torch.tensor(np.asarray(motion_data["root_pos"], dtype=np.float32), device=device)
    root_rot_wxyz = torch.tensor(np.asarray(motion_data["root_rot"], dtype=np.float32), device=device)
    dof_pos = torch.tensor(np.asarray(motion_data["dof_pos"], dtype=np.float32), device=device)

    T = dof_pos.shape[0]
    N = dof_pos.shape[1]
    log.info(f"Motion data shape: T={T}, N={N}")

    out = dict(motion_data)

    if not urdf_path:
        log.warning("No URDF path provided, skipping dof_rot computation")
    elif not os.path.exists(urdf_path):
        log.warning(f"URDF path does not exist: {urdf_path}, skipping dof_rot computation")
        log.info(f"Please check if the file exists at: {os.path.abspath(urdf_path)}")
    else:
        try:
            log.info("Attempting to import motion_tools...")
            from motion_tools.utils.kinematics_model import KinematicsModel
            log.info("motion_tools imported successfully")
            
            log.info(f"Loading kinematics model from {urdf_path}...")
            kin = KinematicsModel(file_path=urdf_path, device=device)
            log.info(f"Kinematics model loaded successfully, num_dof={kin.num_dof}")
            
            if kin.num_dof != N:
                log.warning(f"URDF num_dof={kin.num_dof} does not match dof_pos dim={N}")
                N = min(kin.num_dof, N)
                dof_pos = dof_pos[:, :N]

            actuated_body_indices = get_actuated_joint_indices(kin)
            num_act = len(actuated_body_indices)
            log.info(f"Found {num_act} actuated joints")

            dof_rot = torch.zeros((T, num_act, 3, 3), dtype=torch.float32, device=device)

            rr_xyzw = torch.stack([
                root_rot_wxyz[:, 1], root_rot_wxyz[:, 2], root_rot_wxyz[:, 3], root_rot_wxyz[:, 0]
            ], dim=-1)

            batch_size = 512
            log.info(f"Computing forward kinematics for {T} frames...")
            
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                
                if start % (batch_size * 4) == 0:
                    log.info(f"Processing batch [{start+1}-{end}/{T}]...")
                
                rp = root_pos[start:end]
                rr = rr_xyzw[start:end]
                dq = dof_pos[start:end]

                body_pos_w, body_rot_xyzw = kin.forward_kinematics(rp, rr, dq)
                body_rot_sel = body_rot_xyzw[:, actuated_body_indices, :]
                rotmat = quat_xyzw_to_rotmat(body_rot_sel.reshape(-1, 4)).reshape(-1, num_act, 3, 3)
                dof_rot[start:end] = rotmat

            log.info("Forward kinematics computation completed.")
            out["dof_rot"] = dof_rot.cpu().numpy()
            log.info(f"Successfully added dof_rot with shape {dof_rot.shape}")
            
        except ImportError as e:
            log.error(f"Failed to import motion_tools: {e}")
            log.error("Please ensure motion_tools is installed")
            import traceback
            traceback.print_exc()
        except Exception as e:
            log.error(f"Error computing dof_rot: {e}")
            import traceback
            traceback.print_exc()

    if len(resized_images) > 0:
        if len(resized_images) != T:
            log.warning(f"Image count ({len(resized_images)}) != frame count ({T})")
            resized_images = resized_images[:min(T, len(resized_images))]
        out["rgb_images"] = resized_images
        log.info(f"Added {len(resized_images)} resized RGB images ({resized_images[0].shape})")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    log.info(f"Saving enhanced data to {output_path}...")
    
    with open(output_path, "wb") as f:
        pickle.dump(out, f)
    
    log.info("Enhanced pkl saved successfully!")
    log.info(f"Final pkl keys: {list(out.keys())}")
    if "dof_rot" in out:
        log.info(f"dof_rot shape: {out['dof_rot'].shape}")
    else:
        log.warning("WARNING: dof_rot was NOT added to the pkl file!")


def replay_single_trajectory(env, scenario, traj_path, args, obs_saver, motion_data=None):
    """Replay a single trajectory file."""
    log.info(f"Replaying trajectory: {traj_path}")
    
    motion_length = None
    if motion_data is not None and "dof_pos" in motion_data:
        motion_length = len(motion_data["dof_pos"])
        log.info(f"Motion data length: {motion_length} frames")
    
    tic = time.time()
    assert os.path.exists(traj_path), f"Trajectory file: {traj_path} does not exist."
    original_traj_filepath = scenario.task.traj_filepath
    scenario.task.traj_filepath = traj_path
    init_states, all_actions, all_states = get_traj(
        scenario.task, scenario.robots[0], env.handler
    )
    scenario.task.traj_filepath = original_traj_filepath
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    tic = time.time()
    
    if args.robot_height_offset != 0.0:
        for state in init_states[:args.num_envs]:
            if hasattr(state, 'root_pos'):
                state.root_pos[2] += args.robot_height_offset
            
    obs, extras = env.reset(states=init_states[:args.num_envs])
    
    if args.first_person_view:
        update_camera_poses(env, args)
        env.handler.refresh_render()
        
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")
    obs_saver.add(obs)

    step = 0
    while True:
        if motion_length is not None and step >= motion_length:
            log.info(f"Reached motion length limit ({motion_length} frames), stopping")
            break
        
        tic = time.time()
        if args.object_states:
            if all_states is None:
                raise ValueError("All states are None, please check the trajectory file")
            states = get_states(all_states, step, args.num_envs)
            env.handler.set_states(states)
            
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
            
            log.info(f"Step {step}: episode_length_buf={env.episode_length_buf}, episode_length={env.handler.scenario.episode_length}")
            
            obs, reward, success, time_out, extras = env.step(actions)
            
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
    return obs_saver


def main():
    render_cfg = RenderCfg(mode=args.render_mode)
    
    sensor_width_mm = 36.0
    fx_pixels = 1386.4
    fy_pixels = 1388.6
    width_pixels = 1920
    height_pixels = 1080
    cx_pixels = 960
    cy_pixels = 540
    
    horizontal_aperture = (width_pixels / fx_pixels) * sensor_width_mm
    focal_length = fx_pixels * sensor_width_mm / width_pixels
    
    if args.first_person_view:
        camera = PinholeCameraCfg(
            pos=(0.0, 0.0, 0.0),
            look_at=(1.0, 0.0, 0.0),
            width=args.camera_width,
            height=args.camera_height,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture
        )
        fovx = camera.horizontal_fov
        fovy = camera.vertical_fov
        log.info(f"fovx: {fovx}, fovy: {fovy}")
        log.info(f"Camera offset: {args.camera_offset}, direction: {args.camera_direction}")
        log.info(f"Camera intrinsics: focal_length={focal_length:.3f}mm, horizontal_aperture={horizontal_aperture:.3f}mm")
        log.info(f"Original params: fx={fx_pixels}, fy={fy_pixels}, resolution={width_pixels}x{height_pixels}")
    else:
        camera_pos = (2.5, 0.0, 2.5)
        look_at_pos = (-3.0, 0.0, 0.0)
        
        camera = PinholeCameraCfg(
            pos=camera_pos, 
            look_at=look_at_pos,
            width=args.camera_width,
            height=args.camera_height,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture
        )
        log.info(f"Using standard fixed camera")
        log.info(f"Camera intrinsics: focal_length={focal_length:.3f}mm, horizontal_aperture={horizontal_aperture:.3f}mm")
        log.info(f"Original params: fx={fx_pixels}, fy={fy_pixels}, resolution={width_pixels}x{height_pixels}")
    
    log.info(f"Image Quality Settings:")
    log.info(f"  Camera Resolution: {args.camera_width}x{args.camera_height}")
    log.info(f"  Render Mode: {args.render_mode}")
    log.info(f"  Save Quality: {args.save_quality}/100")
    log.info(f"  Resized Image Size: {args.image_size}x{args.image_size}")
    
    scenario = ScenarioCfg(
        task=args.task,
        robots=[args.robot],
        scene=args.scene,
        cameras=[camera],
        random=args.random,
        render=render_cfg,
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

    motion_data = None
    if args.motion_config:
        motion_data = load_motion_data(args.motion_config)

    traj_filepaths = scenario.task.traj_filepath
    if not isinstance(traj_filepaths, list):
        traj_filepaths = [traj_filepaths]

    if args.save_enhanced_pkl_dir:
        os.makedirs(args.save_enhanced_pkl_dir, exist_ok=True)

    for idx, traj_path in enumerate(traj_filepaths):
        traj_basename = os.path.splitext(os.path.basename(traj_path))[0]
        
        current_image_dir = None
        current_video_path = None
        if args.save_image_dir:
            current_image_dir = os.path.join(args.save_image_dir, f"{traj_basename}")
        if args.save_video_path:
            video_ext = os.path.splitext(args.save_video_path)[1]
            current_video_path = os.path.join(
                os.path.dirname(args.save_video_path),
                f"{traj_basename}{video_ext}"
            )
        
        obs_saver = EnhancedObsSaver(
            image_dir=current_image_dir, 
            video_path=current_video_path, 
            save_quality=args.save_quality,
            image_size=args.image_size
        )
        
        obs_saver = replay_single_trajectory(env, scenario, traj_path, args, obs_saver, motion_data)

        if args.save_enhanced_pkl_dir and motion_data is not None:
            output_pkl_path = os.path.join(args.save_enhanced_pkl_dir, f"{traj_basename}_enhanced.pkl")
            save_enhanced_pkl(
                output_pkl_path, 
                motion_data, 
                obs_saver.resized_images,
                args.robot_urdf if args.robot_urdf else "",
                device="cpu"
            )

    env.close()


if __name__ == "__main__":
    main()