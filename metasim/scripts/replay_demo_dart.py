from __future__ import annotations

import logging
import os
import sys
import pickle
import time
from typing import Literal, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

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

# DART imports
sys.path.insert(0, os.path.join(project_root, 'DART'))
from DART.model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from DART.model.mld_vae import AutoMldVae
from DART.data_loaders.humanml.data.dataset_hml3d import SinglePrimitiveDatasetRetarget
from DART.utils.misc_util import encode_text, compose_texts_with_and
from DART.mld.train_mvae import Args as MVAEArgs
from DART.mld.train_mld import MLDArgs, create_gaussian_diffusion
from DART.data_scripts.retarget_utils.torch_utils import quat_from_euler_xyz
from DART.data_scripts.retarget_utils.retarget_data_utils import convert_relative_to_absolute_retarget_motion
import torch.nn as nn

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
    stop_on_runout: bool = True
    
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
    
    # DART integration options
    use_dart: bool = True
    dart_prefix_frames: int = 2
    motion_pkl_path: str = "motion_data/test1_zero_male/walk_stand_v2_enhanced.pkl"
    denoiser_checkpoint: str = 'DART/mld_denoiser/mld_hml3d_retarget_collected/checkpoint_200000.pt'
    text_prompt: str = 'walks in the scene'
    guidance_param: float = 5.0
    respacing: str = ''
    replan_freq: int = 8  # frames @ 20fps = 0.5s
    replan_chunk_size: int = 10  # frames @ 20fps = 1.25s
    debug_print: bool = False

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


class ClassifierFreeWrapper(nn.Module):
    """Wrapper for classifier-free guidance in diffusion model."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        assert self.model.cond_mask_prob > 0

    def forward(self, x, timesteps, y=None):
        y['uncond'] = False
        out = self.model(x, timesteps, y)
        y_uncond = y.copy()
        y_uncond['uncond'] = True
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'] * (out - out_uncond))


class DARTMotionGenerator:
    """DART motion generator for autoregressive generation."""
    
    def __init__(self, args, device='cpu'):
        from DART.utils.misc_util import load_and_freeze_clip
        self.clip_model = load_and_freeze_clip('ViT-B/32', device=device)
        self.args = args
        self.device = device
        self.segment_count = 0
        
        # ✅ 新增：收集所有生成的运动数据
        self.collected_motion_data = {
            'root_pos': [],
            'root_rot': [],
            'dof_pos': [],
            'local_body_pos': [],
        }
        self.prefix_data = None  # 保存prefix数据
        
        log.info("Loading DART models...")
        proj_root = project_root
        
        # Load denoiser
        denoiser_path = Path(proj_root) / args.denoiser_checkpoint
        denoiser_dir = denoiser_path.parent
        with open(denoiser_dir / "args.yaml", "r") as f:
            self.denoiser_args = tyro.extras.from_yaml(MLDArgs, yaml.safe_load(f)).denoiser_args
        
        denoiser_class = DenoiserMLP if self.denoiser_args.model_type == 'mlp' else DenoiserTransformer
        self.denoiser_model = denoiser_class(**asdict(self.denoiser_args.model_args)).to(device)
        checkpoint = torch.load(denoiser_path, map_location=device)
        self.denoiser_model.load_state_dict(checkpoint['model_state_dict'])
        self.denoiser_model.eval()
        for p in self.denoiser_model.parameters():
            p.requires_grad = False
        self.denoiser_model = ClassifierFreeWrapper(self.denoiser_model)
        
        # Load VAE
        vae_path = Path(proj_root) / 'DART' / self.denoiser_args.mvae_path.lstrip('./')
        vae_dir = vae_path.parent
        with open(vae_dir / "args.yaml", "r") as f:
            vae_args = tyro.extras.from_yaml(MVAEArgs, yaml.safe_load(f))
        self.vae_model = AutoMldVae(**asdict(vae_args.model_args)).to(device)
        checkpoint = torch.load(vae_path, map_location=device)
        model_state_dict = checkpoint['model_state_dict']
        if 'latent_mean' not in model_state_dict:
            model_state_dict['latent_mean'] = torch.tensor(0)
        if 'latent_std' not in model_state_dict:
            model_state_dict['latent_std'] = torch.tensor(1)
        self.vae_model.load_state_dict(model_state_dict)
        self.vae_model.latent_mean = model_state_dict['latent_mean']
        self.vae_model.latent_std = model_state_dict['latent_std']
        self.vae_model.eval()
        for p in self.vae_model.parameters():
            p.requires_grad = False
        
        # Create diffusion
        diffusion_args = self.denoiser_args.diffusion_args
        diffusion_args.respacing = args.respacing
        self.diffusion = create_gaussian_diffusion(diffusion_args)
        
        # Load dataset for normalization/denormalization
        cfg_path = str(Path(proj_root) / 'DART' / vae_args.data_args.cfg_path.lstrip('./'))
        dataset_path = str(Path(proj_root) / 'DART' / vae_args.data_args.data_dir.lstrip('./'))
        # Use motion_pkl_path from args instead of hardcoded path
        sample_stand_path = str(Path(proj_root) / args.motion_pkl_path)
        self.dataset = SinglePrimitiveDatasetRetarget(
            cfg_path=cfg_path,
            dataset_path=dataset_path,
            sequence_path=sample_stand_path,
            body_type=vae_args.data_args.body_type,
            batch_size=1,
            device=device,
            enforce_gender='male',
            enforce_zero_beta=1,
        )
        
        self.history_length = self.dataset.history_length
        self.future_length = self.dataset.future_length
        
        # ✅ 修复：Encode text prompt，不要额外unsqueeze
        action = compose_texts_with_and(args.text_prompt.split(' and '))
        self.text_embedding = encode_text(
            self.clip_model, 
            [action], 
            force_empty_zero=True
        ).to(device=device, dtype=torch.float32)
        
        # Initialize history motion (will be set from prefix)
        self.history_motion = None
        
        # Store robot joint names (will be set later)
        self.joint_names = None
        
        # ✅ Global accumulated position and rotation for continuous motion
        self.global_root_pos = None  # [3] - accumulated position
        self.global_root_rot = None  # [4] - accumulated rotation (wxyz quaternion)
        
        log.info(f"DART Generator initialized (history={self.history_length}, future={self.future_length})")
    
    def initialize_from_prefix(self, prefix_data: dict):
        """Initialize DART with first few frames as prefix.
        
        Args:
            prefix_data: dict with keys like 'dof_pos', 'root_pos', 'root_rot', etc.
                         Should have shape [T, D] where T >= self.history_length
        """
        log.info(f"Initializing DART with prefix data (need {self.history_length} frames)")
        
        # ✅ 保存prefix数据，用于最终保存
        self.prefix_data = prefix_data
        
        # Convert prefix data to DART format
        dart_format = self._convert_to_dart_format(prefix_data)
        
        # Ensure we have enough frames
        T = dart_format.shape[0]
        if T < self.history_length:
            # Repeat last frame to fill history
            repeat_count = self.history_length - T
            last_frame = dart_format[-1:].repeat(repeat_count, 1)
            dart_format = torch.cat([dart_format, last_frame], dim=0)
            log.warning(f"Prefix only has {T} frames, repeated last frame to get {self.history_length}")
        
        # Take last history_length frames
        history_frames = dart_format[-self.history_length:].unsqueeze(0)  # [1, history_length, D]
        
        # Normalize
        self.history_motion = self.dataset.normalize(history_frames)
        
        # ✅ Only initialize global position if not already set (e.g., from init_states)
        if self.global_root_pos is None:
            log.info("Initializing global position from prefix data (fallback)")
            
            if 'root_pos' in prefix_data:
                # Use last frame of prefix as initial global position
                last_pos = prefix_data['root_pos'][-1]
                self.global_root_pos = torch.tensor(
                    last_pos, 
                    dtype=torch.float32
                )
            else:
                self.global_root_pos = torch.zeros(3, dtype=torch.float32)
            
            # Initialize global rotation - handle different formats
            if 'root_rot' in prefix_data:
                last_rot = prefix_data['root_rot'][-1]
                
                # Check format: quaternion (4 elements) or euler angles (3 elements)
                if len(last_rot) == 4:
                    # Quaternion format (assume wxyz)
                    self.global_root_rot = torch.tensor(
                        last_rot, 
                        dtype=torch.float32
                    )
                elif len(last_rot) == 3:
                    # Euler angles format - convert to quaternion
                    euler_tensor = torch.tensor(last_rot, dtype=torch.float32)
                    quat_xyzw = quat_from_euler_xyz(
                        euler_tensor[0], 
                        euler_tensor[1], 
                        euler_tensor[2]
                    )
                    # Convert xyzw to wxyz
                    self.global_root_rot = torch.tensor([
                        quat_xyzw[3],  # w
                        quat_xyzw[0],  # x
                        quat_xyzw[1],  # y
                        quat_xyzw[2]   # z
                    ], dtype=torch.float32)
                else:
                    log.warning(f"Unknown root_rot format with {len(last_rot)} elements, using identity")
                    self.global_root_rot = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
            else:
                # Default: identity quaternion (no rotation)
                self.global_root_rot = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
        else:
            log.info("Global position already set (from init_states), skipping prefix override")
        
        log.info(f"✅ DART initialized successfully")
        log.info(f"   Final global pos: {self.global_root_pos.cpu().numpy()}")
        log.info(f"   Final global rot: {self.global_root_rot.cpu().numpy()}")
    
    def _convert_to_dart_format(self, data: dict) -> torch.Tensor:
        """Convert trajectory data format to DART motion format."""
        device = self.device
        
        # Convert all numpy arrays to tensors
        motion_dict = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                motion_dict[key] = torch.tensor(value, dtype=torch.float32, device=device)
            elif not isinstance(value, (str, dict)):
                motion_dict[key] = value
        
        if self.args.debug_print:
            log.debug(f"Converted keys: {list(motion_dict.keys())}")
            log.debug(f"Dataset expects: {self.dataset.motion_repr}")
        
        # Use dataset's dict_to_tensor to convert to tensor format
        motion_tensor = self.dataset.dict_to_tensor(motion_dict)
        
        return motion_tensor
    
    def generate_next_frames(self, num_frames: int = None) -> dict:
        """Generate next frames using DART.
        
        Args:
            num_frames: Number of frames to generate (default: future_length)
            
        Returns:
            dict with motion data in DART format (relative coordinates)
        """
        if self.history_motion is None:
            raise RuntimeError("DART not initialized! Call initialize_from_prefix first.")
        
        if num_frames is None:
            num_frames = self.future_length
        
        # Prepare conditioning
        guidance_param = torch.ones(1, *self.denoiser_args.model_args.noise_shape).to(
            self.device) * self.args.guidance_param
        y = {
            'text_embedding': self.text_embedding,
            'history_motion_normalized': self.history_motion,
            'scale': guidance_param,
        }
        
        # Sample from diffusion model
        sample_fn = (self.diffusion.p_sample_loop if self.args.respacing == ''
                     else self.diffusion.ddim_sample_loop)
        
        x_start_pred = sample_fn(
            self.denoiser_model,
            (1, *self.denoiser_args.model_args.noise_shape),
            clip_denoised=False,
            model_kwargs={'y': y},
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        
        # Decode
        latent_pred = x_start_pred.permute(1, 0, 2)
        future_motion_pred = self.vae_model.decode(
            latent_pred, self.history_motion,
            nfuture=self.future_length,
            scale_latent=self.denoiser_args.rescale_latent
        )
        
        # Denormalize
        future_frames = self.dataset.denormalize(future_motion_pred)  # [1, future_length, D]
        
        # Convert to dict format
        motion_dict = self.dataset.tensor_to_dict(future_frames)
        
        # Ensure proper shape [B, T, D]
        for k, v in motion_dict.items():
            if v.dim() == 4:  # [B, D, 1, T]
                motion_dict[k] = v.squeeze(2).permute(0, 2, 1)
            elif v.dim() == 2:  # [T, D]
                motion_dict[k] = v.unsqueeze(0)
        
        # Update history for next generation (autoregressive)
        new_history = future_frames[:, -self.history_length:, :]
        self.history_motion = self.dataset.normalize(new_history)
        
        self.segment_count += 1
        
        if self.args.debug_print:
            log.debug(f"Generated segment {self.segment_count} with {future_frames.shape[1]} frames")
        
        log.info(f"Generated segment {self.segment_count} with {future_frames.shape[1]} frames")
        
        return motion_dict
    
    def convert_dart_to_states(self, dart_dict: dict, robot_cfg: BaseRobotCfg) -> List:
        """✅ Convert DART motion format to states format for object_states mode.
        
        Args:
            dart_dict: Motion dict from DART with relative coordinates
            robot_cfg: Robot configuration
            
        Returns:
            List of states [[state_dict_t0, state_dict_t1, ...]] compatible with env.handler.set_states()
        """
        # Extract data [B, T, D] -> [T, D]
        dof_pos = dart_dict['dof_pos'][0].cpu().numpy()  # [T, num_dof]
        root_pos_rel = dart_dict['root_pos_relative'][0].cpu()  # ✅ [T, 3] torch tensor on CPU
        root_rot_rel = dart_dict['root_rot_relative'][0].cpu()  # ✅ [T, 3] torch tensor on CPU (euler: [0, 0, delta_yaw])
        
        T = dof_pos.shape[0]
        
        # Convert relative to absolute (within this segment)
        root_pos_segment, root_rot_segment = convert_relative_to_absolute_retarget_motion(
            root_pos_rel, 
            root_rot_rel
        )
        # root_pos_segment: [T, 3] - absolute within segment (but segment starts at origin)
        # root_rot_segment: [T, 3] - euler angles [0, 0, yaw_absolute_in_segment]
        
        # ✅ Initialize global position and rotation if not set
        if self.global_root_pos is None:
            # Fallback: initialize from first frame of segment
            self.global_root_pos = root_pos_segment[0].clone()
            # Initialize with identity rotation (will be set properly from init_states)
            self.global_root_rot = torch.tensor([1, 0, 0, 0], dtype=torch.float32)  # wxyz
            log.warning("Global position not pre-set, using first frame as fallback")
        
        # ✅ Extract initial euler angles from global rotation
        from scipy.spatial.transform import Rotation as R
        global_rot_wxyz = self.global_root_rot.numpy()
        global_rot_xyzw = np.array([global_rot_wxyz[1], global_rot_wxyz[2], 
                                     global_rot_wxyz[3], global_rot_wxyz[0]])
        global_rot = R.from_quat(global_rot_xyzw)
        global_euler = global_rot.as_euler('xyz', degrees=False)
        global_pitch, global_roll, global_yaw = global_euler[0], global_euler[1], global_euler[2]
        
        # ✅ CRITICAL: Only yaw changes in DART! Keep pitch/roll from init_states
        # Get yaw change in this segment
        segment_start_yaw = root_rot_segment[0, 2].item()
        segment_yaw_changes = root_rot_segment[:, 2] - segment_start_yaw  # [T] - yaw change from segment start
        
        # Apply to global yaw (keep pitch/roll constant)
        global_yaws = global_yaw + segment_yaw_changes.numpy()  # [T]
        
        # Build quaternions with accumulated yaw but keeping original pitch/roll
        root_rot_global_quat_xyzw = []
        for t in range(T):
            rot = R.from_euler('xyz', [global_pitch, global_roll, global_yaws[t]], degrees=False)
            root_rot_global_quat_xyzw.append(rot.as_quat())  # xyzw
        root_rot_global_quat_xyzw = np.array(root_rot_global_quat_xyzw)  # [T, 4] xyzw
        
        # ✅ Handle xy and z position separately
        # Compute xy displacement in this segment (in local frame)
        segment_xy_displacement = root_pos_segment[:, :2] - root_pos_segment[0, :2]  # [T, 2]
        
        # Transform displacement to world frame using rotation AT SEGMENT START
        # (because displacement is measured from segment start in that frame)
        root_pos_global = torch.zeros_like(root_pos_segment)
        rot_at_segment_start = R.from_euler('xyz', [global_pitch, global_roll, global_yaw], degrees=False)
        
        for t in range(T):
            # Get displacement in local frame
            local_disp_xy = segment_xy_displacement[t].numpy()
            local_disp_3d = np.array([local_disp_xy[0], local_disp_xy[1], 0.0])
            
            # Transform to world frame using rotation at segment start
            world_disp = rot_at_segment_start.apply(local_disp_3d)
            
            # Add to global position
            root_pos_global[t, 0] = self.global_root_pos[0] + world_disp[0]
            root_pos_global[t, 1] = self.global_root_pos[1] + world_disp[1]
            root_pos_global[t, 2] = root_pos_segment[t, 2]  # z: use DART's absolute height
        
        # Convert quaternion from xyzw to wxyz for Sapien/v2 format
        root_rot_quat_wxyz = torch.zeros(T, 4)
        root_rot_quat_wxyz[:, 0] = torch.from_numpy(root_rot_global_quat_xyzw[:, 3])  # w
        root_rot_quat_wxyz[:, 1] = torch.from_numpy(root_rot_global_quat_xyzw[:, 0])  # x
        root_rot_quat_wxyz[:, 2] = torch.from_numpy(root_rot_global_quat_xyzw[:, 1])  # y
        root_rot_quat_wxyz[:, 3] = torch.from_numpy(root_rot_global_quat_xyzw[:, 2])  # z
        
        # Get joint names
        if self.joint_names is None:
            self.joint_names = list(robot_cfg.actuators.keys())
        
        robot_name = robot_cfg.name
        
        # Build states for each timestep
        states_sequence = []
        for t in range(T):
            # Create dof_pos dict
            dof_dict = {}
            for i, joint_name in enumerate(self.joint_names):
                if i < dof_pos.shape[1]:
                    dof_dict[joint_name] = float(dof_pos[t, i])
            
            # Create state dict for this timestep
            state_t = {
                robot_name: {
                    'pos': root_pos_global[t],
                    'rot': root_rot_quat_wxyz[t],
                    'dof_pos': dof_dict,
                }
            }
            states_sequence.append(state_t)
        
        # ✅ Update global position and rotation to last frame of this segment
        self.global_root_pos[:2] = root_pos_global[-1, :2].clone()
        # Note: z is not accumulated
        
        self.global_root_rot = root_rot_quat_wxyz[-1].clone()
        
        if self.args.debug_print:
            log.debug(f"Segment end global pos: {self.global_root_pos.cpu().numpy()}")
            log.debug(f"Segment end global yaw: {global_yaws[-1]:.3f} rad ({np.degrees(global_yaws[-1]):.1f} deg)")
            log.debug(f"Last frame z height: {root_pos_global[-1, 2].item()}")
        
        # ✅ 收集生成的运动数据
        for t in range(T):
            self.collected_motion_data['root_pos'].append(root_pos_global[t].cpu().numpy())
            self.collected_motion_data['root_rot'].append(root_rot_quat_wxyz[t].cpu().numpy())
            self.collected_motion_data['dof_pos'].append(dof_pos[t])
            # local_body_pos如果存在的话也收集
            if 'local_body_pos' in dart_dict:
                local_body_pos = dart_dict['local_body_pos'][0].cpu().numpy()
                self.collected_motion_data['local_body_pos'].append(local_body_pos[t])
        
        log.info(f"Converted DART motion to {len(states_sequence)} states")
        
        # Return in format expected by get_states: [[states]]
        return [states_sequence]
    
    def convert_dart_to_actions(self, dart_dict: dict, robot_cfg: BaseRobotCfg) -> List:
        """Convert DART motion format to action format for non-object_states mode.
        
        Args:
            dart_dict: Motion dict from DART with relative coordinates
            robot_cfg: Robot configuration
            
        Returns:
            List of actions compatible with the environment
        """
        # Extract data [B, T, D] -> [T, D]
        dof_pos = dart_dict['dof_pos'][0].cpu().numpy()  # [T, num_dof]
        
        T = dof_pos.shape[0]
        
        # Get joint names
        if self.joint_names is None:
            self.joint_names = list(robot_cfg.actuators.keys())
        
        # Build actions for each timestep
        actions = []
        for t in range(T):
            # Create action dict matching robot's joint names
            action = dof_pos[t, :len(self.joint_names)]
            actions.append(action)
        
        log.info(f"Converted DART motion to {len(actions)} actions")
        
        # Return in format expected by get_actions: [[actions]]
        return [actions]
    
    def export_complete_motion_pkl(self, robot_cfg: BaseRobotCfg) -> dict:
        """✅ 导出完整的运动数据（prefix + 生成的部分）
        
        Returns:
            dict with keys: ['fps', 'root_pos', 'root_rot', 'dof_pos', 'local_body_pos', 'link_body_list']
        """
        log.info("Exporting complete motion data (prefix + generated)...")
        
        # 初始化输出字典
        output_dict = {
            'fps': 20,  # 默认20fps
            'link_body_list': list(robot_cfg.actuators.keys()) if robot_cfg else []
        }
        
        # 合并prefix数据和生成的数据
        if self.prefix_data is not None:
            prefix_length = len(self.prefix_data.get('dof_pos', []))
            log.info(f"Prefix frames: {prefix_length}")
        else:
            prefix_length = 0
            log.warning("No prefix data found")
        
        generated_length = len(self.collected_motion_data['dof_pos'])
        log.info(f"Generated frames: {generated_length}")
        
        # 组合数据
        if self.prefix_data is not None and prefix_length > 0:
            # 合并prefix和生成的数据
            output_dict['root_pos'] = np.concatenate([
                self.prefix_data['root_pos'],
                np.array(self.collected_motion_data['root_pos'])
            ], axis=0)
            
            output_dict['root_rot'] = np.concatenate([
                self.prefix_data['root_rot'],
                np.array(self.collected_motion_data['root_rot'])
            ], axis=0)
            
            output_dict['dof_pos'] = np.concatenate([
                self.prefix_data['dof_pos'],
                np.array(self.collected_motion_data['dof_pos'])
            ], axis=0)
            
            # local_body_pos可能不存在
            if 'local_body_pos' in self.prefix_data and len(self.collected_motion_data['local_body_pos']) > 0:
                output_dict['local_body_pos'] = np.concatenate([
                    self.prefix_data['local_body_pos'],
                    np.array(self.collected_motion_data['local_body_pos'])
                ], axis=0)
            elif 'local_body_pos' in self.prefix_data:
                output_dict['local_body_pos'] = self.prefix_data['local_body_pos']
        else:
            # 仅使用生成的数据
            output_dict['root_pos'] = np.array(self.collected_motion_data['root_pos'])
            output_dict['root_rot'] = np.array(self.collected_motion_data['root_rot'])
            output_dict['dof_pos'] = np.array(self.collected_motion_data['dof_pos'])
            if len(self.collected_motion_data['local_body_pos']) > 0:
                output_dict['local_body_pos'] = np.array(self.collected_motion_data['local_body_pos'])
        
        total_frames = len(output_dict['dof_pos'])
        log.success(f"✅ Exported complete motion: {total_frames} frames total")
        log.success(f"   Keys: {list(output_dict.keys())}")
        log.success(f"   root_pos shape: {output_dict['root_pos'].shape}")
        log.success(f"   root_rot shape: {output_dict['root_rot'].shape}")
        log.success(f"   dof_pos shape: {output_dict['dof_pos'].shape}")
        if 'local_body_pos' in output_dict:
            log.success(f"   local_body_pos shape: {output_dict['local_body_pos'].shape}")
        
        return output_dict


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
    xw, yw, zw = x * w, y * w, z * w
    
    rotmat = torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)], dim=-1),
        torch.stack([2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)], dim=-1),
        torch.stack([2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)], dim=-1)
    ], dim=-2)
    
    return rotmat


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


def extract_motion_from_pkl(pkl_data: dict) -> dict:
    """Extract motion data from new nested pkl format.
    
    Handles both formats:
    - Old format: flat dict with motion keys at root level
    - New format: nested dict with structure:
        {
            '<dynamic_name>': {
                'motion': {
                    'gender': ...,
                    'betas': ...,
                    'root_pos_relative': ...,
                    'root_rot_relative': ...,
                    'dof_pos': ...,
                    'root_vel': ...,
                    'root_ang_vel': ...,
                    'dof_vel': ...,
                    'local_body_pos': ...,
                    'rgb_images': ...
                }
            }
        }
    
    Args:
        pkl_data: Loaded pkl data (possibly nested)
        
    Returns:
        Flat motion dict with all motion keys at root level
    """
    # Check if it's the new nested format
    if len(pkl_data) == 1:
        # Get the only key (dynamic name)
        root_key = list(pkl_data.keys())[0]
        
        # Check if it has 'motion' sub-key
        if isinstance(pkl_data[root_key], dict) and 'motion' in pkl_data[root_key]:
            log.info(f"✅ Detected new pkl format with root key: '{root_key}'")
            motion_data = pkl_data[root_key]['motion']
            log.info(f"   Motion keys: {list(motion_data.keys())}")
            return motion_data
    
    # If not nested format, assume it's old format (flat dict)
    log.info("Detected old pkl format (flat dict)")
    return pkl_data


def load_motion_data(config_path: str) -> dict:
    """Load motion data from config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_enhanced_pkl(output_path: str, motion_data: dict, resized_images, 
                      robot_urdf: str, device="cpu"):
    """Save enhanced pkl with additional computed data.
    
    Args:
        resized_images: Can be list or numpy array of RGB images
    """
    log.info("Creating enhanced pkl file...")
    
    out = motion_data.copy()
    
    # Get frame count from motion data
    T = len(out.get('root_pos', []))
    
    # Compute dof_rot if motion_tools is available
    if robot_urdf and motion_tools_path.exists():
        try:
            from motion_tools.kinematics.urdf_kinematics import URDFKinematics
            
            log.info("Computing dof_rot using motion_tools...")
            kin = URDFKinematics(robot_urdf, device=device)
            
            root_pos = torch.from_numpy(out['root_pos']).to(device)
            root_rot_xyzw = torch.from_numpy(out['root_rot']).to(device)
            dof_pos = torch.from_numpy(out['dof_pos']).to(device)
            
            T = root_pos.shape[0]
            num_act = len(kin.actuated_joint_names)
            dof_rot = torch.zeros(T, num_act, 3, 3, device=device)
            
            # Get indices of actuated bodies
            actuated_body_indices = [
                i for i, name in enumerate(kin.link_names)
                if any(joint in name for joint in kin.actuated_joint_names)
            ]
            
            # Compute in batches
            batch_size = 100
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                
                rp = root_pos[start:end]
                rr = root_rot_xyzw[start:end]
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
        except Exception as e:
            log.error(f"Error computing dof_rot: {e}")

    # Handle rgb_images (can be list or numpy array)
    if resized_images is not None:
        img_count = len(resized_images) if isinstance(resized_images, (list, np.ndarray)) else 0
        if img_count > 0:
            if img_count != T:
                log.warning(f"Image count ({img_count}) != frame count ({T})")
                if isinstance(resized_images, list):
                    resized_images = resized_images[:min(T, img_count)]
                else:  # numpy array
                    resized_images = resized_images[:min(T, img_count)]
            out["rgb_images"] = resized_images
            log.info(f"Added {img_count} RGB images (type: {type(resized_images).__name__})")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    log.info(f"Saving enhanced data to {output_path}...")
    
    with open(output_path, "wb") as f:
        pickle.dump(out, f)
    
    log.info("✅ Enhanced pkl saved successfully!")
    log.info(f"Final pkl keys: {list(out.keys())}")


def replay_single_trajectory_with_dart(env, scenario, traj_path, args, obs_saver, 
                                       motion_data=None, dart_generator=None):
    """Replay trajectory with optional DART generation after prefix frames.
    
    Args:
        env: Simulation environment
        scenario: Scenario configuration
        traj_path: Path to trajectory file
        args: Arguments
        obs_saver: Observation saver
        motion_data: Optional motion data for replay
        dart_generator: Optional DART generator for autoregressive generation
    """
    log.info(f"Replaying trajectory: {traj_path}")
    
    if dart_generator is not None:
        log.info(f"🎯 DART mode enabled: Using first {args.dart_prefix_frames} frames as prefix, then generating")
    
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

    # Load pkl data for rgb_images (and DART if enabled)
    full_motion_data = None
    if args.motion_pkl_path and os.path.exists(args.motion_pkl_path):
        log.info(f"Loading motion data from {args.motion_pkl_path}")
        
        # Load complete motion data
        with open(args.motion_pkl_path, 'rb') as f:
            raw_pkl_data = pickle.load(f)
        
        # ✅ Extract motion data from nested format (handles both old and new formats)
        full_motion_data = extract_motion_from_pkl(raw_pkl_data)
        
        # Check if rgb_images exist in pkl
        if 'rgb_images' in full_motion_data:
            rgb_imgs = full_motion_data['rgb_images']
            if isinstance(rgb_imgs, list):
                log.info(f"Found rgb_images in pkl (list) with {len(rgb_imgs)} frames")
                # Convert list to numpy array if needed
                if len(rgb_imgs) > 0:
                    if isinstance(rgb_imgs[0], np.ndarray):
                        log.info(f"  First frame shape: {rgb_imgs[0].shape}")
                    else:
                        log.info(f"  First frame type: {type(rgb_imgs[0])}")
            elif hasattr(rgb_imgs, 'shape'):
                log.info(f"Found rgb_images in pkl (array) with shape: {rgb_imgs.shape}")
            else:
                log.info(f"Found rgb_images in pkl with type: {type(rgb_imgs)}")
            log.info("Will use pkl rgb_images instead of sapien-rendered images")
        else:
            log.warning("No rgb_images found in pkl, will use sapien-rendered images")
    
    # Initialize DART if enabled
    if dart_generator is not None and args.dart_prefix_frames > 0 and full_motion_data is not None:
        log.info(f"Initializing DART with prefix frames")
        
        # Extract prefix frames
        prefix_frames = min(args.dart_prefix_frames, len(full_motion_data['dof_pos']))
        
        prefix_data = {}
        for key, value in full_motion_data.items():
            if isinstance(value, np.ndarray):
                if len(value.shape) > 1 and value.shape[0] >= prefix_frames:
                    prefix_data[key] = value[:prefix_frames]
                else:
                    prefix_data[key] = value
            else:
                prefix_data[key] = value
        
        log.info(f"Extracted {prefix_frames} prefix frames from pkl")
        log.info(f"Available keys: {list(prefix_data.keys())}")
        
        # Set joint names for format conversion
        dart_generator.joint_names = list(scenario.robots[0].actuators.keys())
        
        # ✅ CRITICAL: Get initial position from trajectory's first state
        robot_name = scenario.robots[0].name
        if all_states and len(all_states) > 0 and len(all_states[0]) > 0:
            first_state = all_states[0][0]
            
            log.info(f"First state keys: {first_state.keys()}")
            log.info(f"First state type: {type(first_state)}")
            
            # Try to extract robot state
            robot_state = None
            if robot_name in first_state:
                robot_state = first_state[robot_name]
            elif 'robots' in first_state and robot_name in first_state['robots']:
                robot_state = first_state['robots'][robot_name]
            
            if robot_state is not None:
                log.info(f"Robot state keys: {robot_state.keys() if hasattr(robot_state, 'keys') else 'no keys'}")
                log.info(f"Robot state type: {type(robot_state)}")
                
                # Extract position
                if 'pos' in robot_state:
                    pos_value = robot_state['pos']
                    if isinstance(pos_value, torch.Tensor):
                        dart_generator.global_root_pos = pos_value.float().cpu()
                    else:
                        dart_generator.global_root_pos = torch.tensor(pos_value, dtype=torch.float32)
                    log.success(f"✅ Set DART global_root_pos from trajectory: {dart_generator.global_root_pos.numpy()}")
                
                # Extract rotation
                if 'rot' in robot_state:
                    rot_value = robot_state['rot']
                    if isinstance(rot_value, torch.Tensor):
                        dart_generator.global_root_rot = rot_value.float().cpu()
                    else:
                        dart_generator.global_root_rot = torch.tensor(rot_value, dtype=torch.float32)
                    log.success(f"✅ Set DART global_root_rot from trajectory: {dart_generator.global_root_rot.numpy()}")
            else:
                log.warning(f"Could not find robot '{robot_name}' in first_state")
        else:
            log.warning("No states available to initialize DART position")
        
        # Initialize DART with prefix (this sets up history_motion for generation)
        dart_generator.initialize_from_prefix(prefix_data)
        log.success(f"✅ DART initialized with {prefix_frames} frames from pkl file")

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
    dart_generated_actions = None
    dart_generated_states = None
    dart_action_idx = 0
    dart_state_idx = 0
    total_dart_frames_generated = 0
    
    while True:
        # Check if we should use DART generation
        use_dart_this_step = (dart_generator is not None and 
                              step >= args.dart_prefix_frames)
        
        if use_dart_this_step:
            # ✅ Periodic replanning: generate new batch when needed
            if args.object_states:
                # Generate states for kinematic mode
                if dart_generated_states is None or dart_state_idx >= len(dart_generated_states[0]):
                    log.info(f"📍 Step {step}: Generating new DART motion chunk (kinematic mode)...")
                    
                    # Generate enough segments to fill the chunk
                    segments_needed = (args.replan_chunk_size + dart_generator.future_length - 1) // dart_generator.future_length
                    dart_generated_states = [[]]
                    
                    for _ in range(segments_needed):
                        dart_motion_dict = dart_generator.generate_next_frames()
                        segment_states = dart_generator.convert_dart_to_states(
                            dart_motion_dict, scenario.robots[0]
                        )
                        dart_generated_states[0].extend(segment_states[0])
                    
                    # Trim to exact chunk size
                    dart_generated_states[0] = dart_generated_states[0][:args.replan_chunk_size]
                    total_dart_frames_generated += len(dart_generated_states[0])
                    dart_state_idx = 0
                    
                    log.success(f"✅ Generated {len(dart_generated_states[0])} state frames (total: {total_dart_frames_generated})")
                
                # Get DART-generated robot states
                dart_robot_states = get_states(dart_generated_states, dart_state_idx, args.num_envs)
                
                # Merge with original states to preserve objects
                # Get the last available state as template for objects
                template_step = min(step, len(all_states[0]) - 1) if all_states and len(all_states[0]) > 0 else 0
                base_states = get_states(all_states, template_step, args.num_envs) if all_states else None
                
                # Build complete states with DART robot + original objects
                states = []
                for env_idx in range(args.num_envs):
                    state = {}
                    
                    # Add objects from original states if available
                    if base_states and env_idx < len(base_states):
                        if 'objects' in base_states[env_idx]:
                            state['objects'] = base_states[env_idx]['objects']
                        else:
                            state['objects'] = {}
                    else:
                        state['objects'] = {}
                    
                    # Add DART-generated robot states
                    if env_idx < len(dart_robot_states):
                        if 'robots' in dart_robot_states[env_idx]:
                            state['robots'] = dart_robot_states[env_idx]['robots']
                        else:
                            # dart_robot_states might have robot name as key directly
                            robot_name = scenario.robots[0].name
                            if robot_name in dart_robot_states[env_idx]:
                                state['robots'] = {robot_name: dart_robot_states[env_idx][robot_name]}
                            else:
                                state['robots'] = dart_robot_states[env_idx]
                    
                    states.append(state)
                
                dart_state_idx += 1
            else:
                # Generate actions for dynamic mode
                if dart_generated_actions is None or dart_action_idx >= len(dart_generated_actions[0]):
                    log.info(f"📍 Step {step}: Generating new DART motion chunk (dynamic mode)...")
                    
                    # Generate enough segments to fill the chunk
                    segments_needed = (args.replan_chunk_size + dart_generator.future_length - 1) // dart_generator.future_length
                    dart_generated_actions = [[]]
                    
                    for _ in range(segments_needed):
                        dart_motion_dict = dart_generator.generate_next_frames()
                        segment_actions = dart_generator.convert_dart_to_actions(
                            dart_motion_dict, scenario.robots[0]
                        )
                        dart_generated_actions[0].extend(segment_actions[0])
                    
                    # Trim to exact chunk size
                    dart_generated_actions[0] = dart_generated_actions[0][:args.replan_chunk_size]
                    total_dart_frames_generated += len(dart_generated_actions[0])
                    dart_action_idx = 0
                    
                    log.success(f"✅ Generated {len(dart_generated_actions[0])} action frames (total: {total_dart_frames_generated})")
                
                # Use DART-generated actions
                actions = get_actions(dart_generated_actions, dart_action_idx, args.num_envs, scenario.robots[0])
                dart_action_idx += 1
        else:
            # Use original trajectory (prefix phase)
            if args.object_states:
                states = get_states(all_states, step, args.num_envs)
            else:
                actions = get_actions(all_actions, step, args.num_envs, scenario.robots[0])

        tic = time.time()
        if args.object_states:
            if all_states is None and not use_dart_this_step:
                raise ValueError("All states are None, please check the trajectory file")
            
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
            if args.debug_print:
                log.debug(f"Step {step}: episode_length_buf={env.episode_length_buf}")
            
            obs, reward, success, time_out, extras = env.step(actions)
            
            if args.first_person_view:
                update_camera_poses(env, args)

            if success.any():
                log.info(f"Env {success.nonzero().squeeze(-1).tolist()} succeeded!")

            if time_out.any():
                log.info(f"Env {time_out.nonzero().squeeze(-1).tolist()} timed out!")

            if success.all() or time_out.all():
                break

        toc = time.time()
        log.trace(f"Time to step: {toc - tic:.2f}s")

        tic = time.time()
        obs_saver.add(obs)
        toc = time.time()
        log.trace(f"Time to save obs: {toc - tic:.2f}s")
        step += 1

        # Check if we ran out of original actions (only relevant when not using DART)
        if args.stop_on_runout and not use_dart_this_step:
            if args.object_states:
                if step >= len(all_states[0]):
                    log.info("Run out of original states")
                    if dart_generator is None:
                        log.info("No DART generator, stopping")
                        break
            else:
                if get_runout(all_actions, step):
                    log.info("Run out of original actions")
                    if dart_generator is None:
                        log.info("No DART generator, stopping")
                        break

    obs_saver.save()
    
    if dart_generator is not None:
        log.success(f"🎉 DART Generation Summary:")
        log.success(f"  - Total segments generated: {dart_generator.segment_count}")
        log.success(f"  - Total frames generated: {total_dart_frames_generated}")
    
    return obs_saver


def main():
    render_cfg = RenderCfg(mode=args.render_mode)
    
    # Camera configuration
    sensor_width_mm = 36.0
    fx_pixels = 1386.4
    fy_pixels = 1388.6
    width_pixels = 1920
    height_pixels = 1080
    
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
        log.info(f"First-person camera mode enabled")
        log.info(f"Camera offset: {args.camera_offset}, direction: {args.camera_direction}")
    else:
        camera = PinholeCameraCfg(
            pos=(2.5, 0.0, 2.5), 
            look_at=(-3.0, 0.0, 0.0),
            width=args.camera_width,
            height=args.camera_height,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture
        )
        log.info(f"Using standard fixed camera")
    
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

    # Load motion data if provided
    motion_data = None
    if args.motion_config:
        motion_data = load_motion_data(args.motion_config)

    # Initialize DART generator if enabled
    dart_generator = None
    if args.use_dart:
        log.info("="*80)
        log.info("🚀 DART MOTION GENERATION ENABLED")
        log.info("="*80)
        log.info(f"Text prompt: {args.text_prompt}")
        log.info(f"Prefix frames: {args.dart_prefix_frames}")
        log.info(f"Replan frequency: {args.replan_freq} frames ({args.replan_freq/20:.2f}s @ 20fps)")
        log.info(f"Chunk size: {args.replan_chunk_size} frames ({args.replan_chunk_size/20:.2f}s @ 20fps)")
        log.info(f"Motion pkl: {args.motion_pkl_path}")
        log.info("="*80)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dart_generator = DARTMotionGenerator(args, device=device)

    # Get trajectory paths
    traj_filepaths = scenario.task.traj_filepath
    if not isinstance(traj_filepaths, list):
        traj_filepaths = [traj_filepaths]

    if args.save_enhanced_pkl_dir:
        os.makedirs(args.save_enhanced_pkl_dir, exist_ok=True)

    # Process each trajectory
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
        
        # Replay with optional DART generation
        obs_saver = replay_single_trajectory_with_dart(
            env, scenario, traj_path, args, obs_saver, motion_data, dart_generator
        )

        # ✅ 保存完整的运动pkl（prefix + 生成的部分）
        if dart_generator is not None and args.save_enhanced_pkl_dir:
            complete_motion_dict = dart_generator.export_complete_motion_pkl(scenario.robots[0])
            complete_pkl_path = os.path.join(args.save_enhanced_pkl_dir, f"{traj_basename}_complete_motion.pkl")
            os.makedirs(os.path.dirname(complete_pkl_path), exist_ok=True)
            
            log.info(f"Saving complete motion pkl to {complete_pkl_path}")
            with open(complete_pkl_path, 'wb') as f:
                pickle.dump(complete_motion_dict, f)
            log.success(f"✅ Complete motion pkl saved successfully!")
            log.success(f"   Total frames: {len(complete_motion_dict['dof_pos'])}")
            log.success(f"   Dict keys: {list(complete_motion_dict.keys())}")

        # Save enhanced pkl if requested
        if args.save_enhanced_pkl_dir and motion_data is not None:
            output_pkl_path = os.path.join(args.save_enhanced_pkl_dir, f"{traj_basename}_enhanced.pkl")
            # Use rgb_images from pkl if available, otherwise use sapien-rendered images
            rgb_images_to_save = None
            if full_motion_data is not None and 'rgb_images' in full_motion_data:
                rgb_images_to_save = full_motion_data['rgb_images']
                if isinstance(rgb_images_to_save, list):
                    log.info(f"Using rgb_images from pkl for saving: {len(rgb_images_to_save)} frames (list)")
                elif hasattr(rgb_images_to_save, 'shape'):
                    log.info(f"Using rgb_images from pkl for saving: {rgb_images_to_save.shape}")
                else:
                    log.info(f"Using rgb_images from pkl for saving: type {type(rgb_images_to_save)}")
            else:
                rgb_images_to_save = obs_saver.resized_images
                log.info(f"Using sapien-rendered images for saving: {len(rgb_images_to_save)} frames")
            
            save_enhanced_pkl(
                output_pkl_path, 
                motion_data, 
                rgb_images_to_save,
                args.robot_urdf if args.robot_urdf else "",
                device="cpu"
            )

    env.close()
    log.success("🎉 All trajectories processed successfully!")


if __name__ == "__main__":
    main()