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
    debug_print: bool = False
    
    # Policy integration options
    use_policy: bool = False
    policy_checkpoint: str | None = None
    policy_type: Literal["rl", "vla", "custom"] = "rl"
    policy_config: str | None = None

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


class PolicyGenerator:
    """Policy-based action generator."""
    
    def __init__(self, args, device='cpu'):
        self.args = args
        self.device = device
        self.policy = None
        self.obs_history = []
        
        log.info("Loading Policy...")
        
        if args.policy_checkpoint is None:
            raise ValueError("policy_checkpoint must be specified when use_policy=True")
        
        # Load policy based on policy_type
        if args.policy_type == "rl":
            self._load_rl_policy()
        elif args.policy_type == "vla":
            self._load_vla_policy()
        elif args.policy_type == "custom":
            self._load_custom_policy()
        else:
            raise ValueError(f"Unknown policy_type: {args.policy_type}")
        
        log.info(f"Policy loaded successfully (type: {args.policy_type})")
    
    def _load_rl_policy(self):
        """Load RL policy (e.g., PPO, SAC)."""
        import torch
        
        checkpoint = torch.load(self.args.policy_checkpoint, map_location=self.device)
        
        # Example: load policy network
        # You need to adapt this to your actual policy architecture
        # self.policy = YourPolicyClass(...)
        # self.policy.load_state_dict(checkpoint['policy_state_dict'])
        # self.policy.eval()
        
        log.warning("RL policy loading not fully implemented - please customize _load_rl_policy()")
        self.policy = None
    
    
    def predict_action(self, obs: TensorState, robot_cfg: BaseRobotCfg):
        """Predict action from observation.
        
        Args:
            obs: Current observation state
            robot_cfg: Robot configuration
            
        Returns:
            action: Action array for the robot
        """
        if self.policy is None:
            raise RuntimeError("Policy not loaded! Please implement policy loading in PolicyGenerator.")
        
        # Extract relevant observation data
        # This is an example - adapt to your observation structure
        obs_dict = {}
        
        # Extract camera observations
        if hasattr(obs, 'cameras') and obs.cameras:
            camera_data = next(iter(obs.cameras.values()))
            if hasattr(camera_data, 'rgb'):
                obs_dict['rgb'] = camera_data.rgb
        
        # Extract proprioceptive observations
        if hasattr(obs, 'robots') and obs.robots:
            robot_data = next(iter(obs.robots.values()))
            if hasattr(robot_data, 'dof_pos'):
                obs_dict['dof_pos'] = robot_data.dof_pos
            if hasattr(robot_data, 'dof_vel'):
                obs_dict['dof_vel'] = robot_data.dof_vel
        
        # Call policy to get action
        with torch.no_grad():
            # Example: action = self.policy(obs_dict)
            # You need to implement this based on your policy interface
            pass
        
        # Placeholder - return zero action
        num_dofs = len(robot_cfg.actuators)
        action = np.zeros(num_dofs, dtype=np.float32)
        
        log.warning("predict_action() returning zero action - please implement actual policy inference")
        return action


class DARTMotionGenerator:
    """DART motion generator for autoregressive generation."""
    
    def __init__(self, args, device='cpu'):
        from DART.utils.misc_util import load_and_freeze_clip
        self.clip_model = load_and_freeze_clip('ViT-B/32', device=device)
        self.args = args
        self.device = device
        self.segment_count = 0
        
        self.collected_motion_data = {
            'root_pos': [],
            'root_rot': [],
            'dof_pos': [],
            'local_body_pos': [],
        }
        self.prefix_data = None
        
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
        
        action = compose_texts_with_and(args.text_prompt.split(' and '))
        self.text_embedding = encode_text(
            self.clip_model, 
            [action], 
            force_empty_zero=True
        ).to(device=device, dtype=torch.float32)
        
        self.history_motion = None
        self.joint_names = None
        
        self.global_root_pos = None
        self.global_root_rot = None
        
        log.info(f"DART Generator initialized (history={self.history_length}, future={self.future_length})")
    
    def initialize_from_prefix(self, prefix_data: dict):
        """Initialize DART with first few frames as prefix.
        
        Args:
            prefix_data: dict with keys like 'dof_pos', 'root_pos', 'root_rot', etc.
                         Should have shape [T, D] where T >= self.history_length
        """
        log.info(f"Initializing DART with prefix data (need {self.history_length} frames)")
        
        self.prefix_data = prefix_data
        
        # Convert prefix data to DART format
        dart_format = self._convert_to_dart_format(prefix_data)
        
        T = dart_format.shape[0]
        if T < self.history_length:
            # Repeat last frame to fill history
            repeat_count = self.history_length - T
            last_frame = dart_format[-1:].repeat(repeat_count, 1)
            dart_format = torch.cat([dart_format, last_frame], dim=0)
            log.warning(f"Prefix only has {T} frames, repeated last frame to get {self.history_length}")
        
        history_frames = dart_format[-self.history_length:].unsqueeze(0)  # [1, history_length, D]
        
        self.history_motion = self.dataset.normalize(history_frames)
        
        if self.global_root_pos is None:
            log.info("Initializing global position from prefix data (fallback)")
            
            if 'root_pos' in prefix_data:
                last_pos = prefix_data['root_pos'][-1]
                self.global_root_pos = torch.tensor(
                    last_pos, 
                    dtype=torch.float32
                )
            else:
                self.global_root_pos = torch.zeros(3, dtype=torch.float32)
            
            if 'root_rot' in prefix_data:
                last_rot = prefix_data['root_rot'][-1]
                
                if len(last_rot) == 4:
                    self.global_root_rot = torch.tensor(
                        last_rot, 
                        dtype=torch.float32
                    )
                elif len(last_rot) == 3:
                    euler_tensor = torch.tensor(last_rot, dtype=torch.float32)
                    quat_xyzw = quat_from_euler_xyz(
                        euler_tensor[0], 
                        euler_tensor[1], 
                        euler_tensor[2]
                    )
                    self.global_root_rot = torch.tensor([
                        quat_xyzw[3],
                        quat_xyzw[0],
                        quat_xyzw[1],
                        quat_xyzw[2]
                    ], dtype=torch.float32)
                else:
                    log.warning(f"Unknown root_rot format with {len(last_rot)} elements, using identity")
                    self.global_root_rot = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
            else:
                self.global_root_rot = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
        else:
            log.info("Global position already set (from init_states), skipping prefix override")
        
        log.info(f"DART initialized successfully")
        log.info(f"   Final global pos: {self.global_root_pos.cpu().numpy()}")
        log.info(f"   Final global rot: {self.global_root_rot.cpu().numpy()}")
    
    def _convert_to_dart_format(self, data: dict) -> torch.Tensor:
        """Convert trajectory data format to DART motion format."""
        device = self.device
        
        motion_dict = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                motion_dict[key] = torch.tensor(value, dtype=torch.float32, device=device)
            elif torch.is_tensor(value):
                motion_dict[key] = value.to(device=device, dtype=torch.float32)
            elif not isinstance(value, (str, dict)):
                motion_dict[key] = value
        
        if self.args.debug_print:
            log.debug(f"Converted keys: {list(motion_dict.keys())}")
            log.debug(f"Dataset expects: {self.dataset.motion_repr}")
        
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
        
        guidance_param = torch.ones(1, *self.denoiser_args.model_args.noise_shape).to(
            self.device) * self.args.guidance_param
        y = {
            'text_embedding': self.text_embedding,
            'history_motion_normalized': self.history_motion,
            'scale': guidance_param,
        }
        
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
        
        latent_pred = x_start_pred.permute(1, 0, 2)
        future_motion_pred = self.vae_model.decode(
            latent_pred, self.history_motion,
            nfuture=self.future_length,
            scale_latent=self.denoiser_args.rescale_latent
        )
        
        future_frames = self.dataset.denormalize(future_motion_pred)
        
        motion_dict = self.dataset.tensor_to_dict(future_frames)
        
        for k, v in motion_dict.items():
            if v.dim() == 4:
                motion_dict[k] = v.squeeze(2).permute(0, 2, 1)
            elif v.dim() == 2:
                motion_dict[k] = v.unsqueeze(0)
        
        new_history = future_frames[:, -self.history_length:, :]
        self.history_motion = self.dataset.normalize(new_history)
        
        self.segment_count += 1
        
        if self.args.debug_print:
            log.debug(f"Generated segment {self.segment_count} with {future_frames.shape[1]} frames")
        
        return motion_dict
    
    def convert_dart_to_states(self, dart_dict: dict, robot_cfg: BaseRobotCfg) -> List:
        """Convert DART motion format to states format for object_states mode.
        
        Args:
            dart_dict: Motion dict from DART with relative coordinates
            robot_cfg: Robot configuration
            
        Returns:
            List of states [[state_dict_t0, state_dict_t1, ...]] compatible with env.handler.set_states()
        """
        dof_pos = dart_dict['dof_pos'][0].cpu().numpy()
        root_pos_rel = dart_dict['root_pos_relative'][0].cpu()
        root_rot_rel = dart_dict['root_rot_relative'][0].cpu()
        
        T = dof_pos.shape[0]
        
        root_pos_segment, root_rot_segment = convert_relative_to_absolute_retarget_motion(
            root_pos_rel, 
            root_rot_rel
        )
        
        if self.global_root_pos is None:
            self.global_root_pos = root_pos_segment[0].clone()
            self.global_root_rot = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
            log.warning("Global position not pre-set, using first frame as fallback")
        
        from scipy.spatial.transform import Rotation as R
        global_rot_wxyz = self.global_root_rot.numpy()
        global_rot_xyzw = np.array([global_rot_wxyz[1], global_rot_wxyz[2], 
                                     global_rot_wxyz[3], global_rot_wxyz[0]])
        global_rot = R.from_quat(global_rot_xyzw)
        global_euler = global_rot.as_euler('xyz', degrees=False)
        global_pitch, global_roll, global_yaw = global_euler[0], global_euler[1], global_euler[2]
        
        segment_start_yaw = root_rot_segment[0, 2].item()
        segment_yaw_changes = root_rot_segment[:, 2] - segment_start_yaw
        
        global_yaws = global_yaw + segment_yaw_changes.numpy()
        
        root_rot_global_quat_xyzw = []
        for t in range(T):
            rot = R.from_euler('xyz', [global_pitch, global_roll, global_yaws[t]], degrees=False)
            root_rot_global_quat_xyzw.append(rot.as_quat())
        root_rot_global_quat_xyzw = np.array(root_rot_global_quat_xyzw)
        
        segment_xy_displacement = root_pos_segment[:, :2] - root_pos_segment[0, :2]
        
        root_pos_global = torch.zeros_like(root_pos_segment)
        rot_at_segment_start = R.from_euler('xyz', [global_pitch, global_roll, global_yaw], degrees=False)
        
        for t in range(T):
            local_disp_xy = segment_xy_displacement[t].numpy()
            local_disp_3d = np.array([local_disp_xy[0], local_disp_xy[1], 0.0])
            
            world_disp = rot_at_segment_start.apply(local_disp_3d)
            
            root_pos_global[t, 0] = self.global_root_pos[0] + world_disp[0]
            root_pos_global[t, 1] = self.global_root_pos[1] + world_disp[1]
            root_pos_global[t, 2] = root_pos_segment[t, 2]
        
        root_rot_quat_wxyz = torch.zeros(T, 4)
        root_rot_quat_wxyz[:, 0] = torch.from_numpy(root_rot_global_quat_xyzw[:, 3])
        root_rot_quat_wxyz[:, 1] = torch.from_numpy(root_rot_global_quat_xyzw[:, 0])
        root_rot_quat_wxyz[:, 2] = torch.from_numpy(root_rot_global_quat_xyzw[:, 1])
        root_rot_quat_wxyz[:, 3] = torch.from_numpy(root_rot_global_quat_xyzw[:, 2])
        
        if self.joint_names is None:
            self.joint_names = list(robot_cfg.actuators.keys())
        
        robot_name = robot_cfg.name
        
        states_sequence = []
        for t in range(T):
            dof_dict = {}
            for i, joint_name in enumerate(self.joint_names):
                if i < dof_pos.shape[1]:
                    dof_dict[joint_name] = float(dof_pos[t, i])
            
            state_t = {
                robot_name: {
                    'pos': root_pos_global[t],
                    'rot': root_rot_quat_wxyz[t],
                    'dof_pos': dof_dict,
                }
            }
            states_sequence.append(state_t)
        
        self.global_root_pos[:2] = root_pos_global[-1, :2].clone()
        self.global_root_rot = root_rot_quat_wxyz[-1].clone()
        
        if self.args.debug_print:
            log.debug(f"Segment end global pos: {self.global_root_pos.cpu().numpy()}")
            log.debug(f"Segment end global yaw: {global_yaws[-1]:.3f} rad ({np.degrees(global_yaws[-1]):.1f} deg)")
            log.debug(f"Last frame z height: {root_pos_global[-1, 2].item()}")
        
        for t in range(T):
            self.collected_motion_data['root_pos'].append(root_pos_global[t].cpu().numpy())
            self.collected_motion_data['root_rot'].append(root_rot_quat_wxyz[t].cpu().numpy())
            self.collected_motion_data['dof_pos'].append(dof_pos[t])
            if 'local_body_pos' in dart_dict:
                local_body_pos = dart_dict['local_body_pos'][0].cpu().numpy()
                self.collected_motion_data['local_body_pos'].append(local_body_pos[t])
        
        log.info(f"Converted DART motion to {len(states_sequence)} states")
        
        return [states_sequence]
    
    def convert_dart_to_actions(self, dart_dict: dict, robot_cfg: BaseRobotCfg) -> List:
        """Convert DART motion format to action format for non-object_states mode.
        
        Args:
            dart_dict: Motion dict from DART with relative coordinates
            robot_cfg: Robot configuration
            
        Returns:
            List of actions compatible with the environment
        """
        dof_pos = dart_dict['dof_pos'][0].cpu().numpy()
        
        T = dof_pos.shape[0]
        
        if self.joint_names is None:
            self.joint_names = list(robot_cfg.actuators.keys())
        
        actions = []
        for t in range(T):
            action = dof_pos[t, :len(self.joint_names)]
            actions.append(action)
        
        log.info(f"Converted DART motion to {len(actions)} actions")
        
        return [actions]
    
    def export_complete_motion_pkl(self, robot_cfg: BaseRobotCfg) -> dict:
        """Export complete motion data (prefix + generated).
        
        Returns:
            dict with keys: ['fps', 'root_pos', 'root_rot', 'dof_pos', 'local_body_pos', 'link_body_list']
        """
        log.info("Exporting complete motion data (prefix + generated)...")
        
        output_dict = {
            'fps': 20,
            'link_body_list': list(robot_cfg.actuators.keys()) if robot_cfg else []
        }
        
        if self.prefix_data is not None:
            prefix_length = len(self.prefix_data.get('dof_pos', []))
        else:
            prefix_length = 0
            log.warning("No prefix data found")
        
        generated_length = len(self.collected_motion_data['dof_pos'])
        
        if self.prefix_data is not None and prefix_length > 0:
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
            
            if 'local_body_pos' in self.prefix_data and len(self.collected_motion_data['local_body_pos']) > 0:
                output_dict['local_body_pos'] = np.concatenate([
                    self.prefix_data['local_body_pos'],
                    np.array(self.collected_motion_data['local_body_pos'])
                ], axis=0)
            elif 'local_body_pos' in self.prefix_data:
                output_dict['local_body_pos'] = self.prefix_data['local_body_pos']
        else:
            output_dict['root_pos'] = np.array(self.collected_motion_data['root_pos'])
            output_dict['root_rot'] = np.array(self.collected_motion_data['root_rot'])
            output_dict['dof_pos'] = np.array(self.collected_motion_data['dof_pos'])
            if len(self.collected_motion_data['local_body_pos']) > 0:
                output_dict['local_body_pos'] = np.array(self.collected_motion_data['local_body_pos'])
        
        total_frames = len(output_dict['dof_pos'])
        log.info(f"Exported complete motion: {total_frames} frames total")
        
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
    - New format: nested dict
    
    Args:
        pkl_data: Loaded pkl data (possibly nested)
        
    Returns:
        Flat motion dict with all motion keys at root level
    """
    if len(pkl_data) == 1:
        root_key = list(pkl_data.keys())[0]
        
        if isinstance(pkl_data[root_key], dict) and 'motion' in pkl_data[root_key]:
            motion_data = pkl_data[root_key]['motion']
            return motion_data
    
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
    
    T = len(out.get('root_pos', []))
    
    # MODIFICATION: Disabled dof_rot computation as requested
    if False and robot_urdf and motion_tools_path.exists():
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
            
            actuated_body_indices = [
                i for i, name in enumerate(kin.link_names)
                if any(joint in name for joint in kin.actuated_joint_names)
            ]
            
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
    else:
        # MODIFICATION: Log that dof_rot computation is skipped
        if robot_urdf and motion_tools_path.exists():
            log.warning("Skipping dof_rot computation as requested.")
        else:
            log.warning("Skipping dof_rot computation (robot_urdf or motion_tools not found).")

    if resized_images is not None:
        img_count = len(resized_images) if isinstance(resized_images, (list, np.ndarray)) else 0
        if img_count > 0:
            if img_count != T:
                log.warning(f"Image count ({img_count}) != frame count ({T})")
                if isinstance(resized_images, list):
                    resized_images = resized_images[:min(T, img_count)]
                else:
                    resized_images = resized_images[:min(T, img_count)]
            out["rgb_images"] = resized_images
            log.info(f"Added {img_count} RGB images (type: {type(resized_images).__name__})")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    log.info(f"Saving enhanced data to {output_path}...")
    
    with open(output_path, "wb") as f:
        pickle.dump(out, f)
    
    log.info("Enhanced pkl saved successfully!")


def replay_single_trajectory_with_dart(env, scenario, traj_path, args, obs_saver, 
                                       motion_data=None, dart_generator=None, policy_generator=None):
    """Replay trajectory with optional DART generation or policy control after prefix frames.
    
    Args:
        env: Simulation environment
        scenario: Scenario configuration
        traj_path: Path to trajectory file
        args: Arguments
        obs_saver: Observation saver
        motion_data: Optional motion data for replay
        dart_generator: Optional DART generator for autoregressive generation
        policy_generator: Optional policy generator for policy-based control
    """
    log.info(f"Replaying trajectory: {traj_path}")
    
    if dart_generator is not None:
        log.info(f"DART mode enabled: Using first {args.dart_prefix_frames} frames as prefix, then generating")
    
    if policy_generator is not None:
        log.info(f"Policy mode enabled: Using policy to generate actions")
    
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

    full_motion_data = None
    if args.motion_pkl_path and os.path.exists(args.motion_pkl_path):
        log.info(f"Loading motion data from {args.motion_pkl_path}")
        
        with open(args.motion_pkl_path, 'rb') as f:
            raw_pkl_data = pickle.load(f)
        
        full_motion_data = extract_motion_from_pkl(raw_pkl_data)
        
        if 'rgb_images' in full_motion_data:
            log.info("Found rgb_images in pkl, will be overwritten by new render")
        else:
            log.warning("No rgb_images found in pkl, will use new render")
    
    if dart_generator is not None and args.dart_prefix_frames > 0 and full_motion_data is not None:
        log.info(f"Initializing DART with prefix frames")
        
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
        
        # --- MODIFICATION: Overwrite prefix data with "real" data from all_states ---
        if all_states and len(all_states) > 0 and len(all_states[0]) >= prefix_frames:
            log.info(f"Overwriting DART prefix with 'real' data from all_states")
            robot_name = scenario.robots[0].name
            joint_names = list(scenario.robots[0].actuators.keys())
            
            real_dof_pos = []
            real_root_pos = []
            real_root_rot = []
            
            for t in range(prefix_frames):
                state_t = all_states[0][t] # Assumes env 0
                
                robot_state = None
                if robot_name in state_t:
                    robot_state = state_t[robot_name]
                elif 'robots' in state_t and robot_name in state_t['robots']:
                    robot_state = state_t['robots'][robot_name]
                
                if robot_state:
                    pos = robot_state['pos']
                    rot = robot_state['rot']
                    
                    real_root_pos.append(pos.cpu().numpy() if torch.is_tensor(pos) else pos)
                    real_root_rot.append(rot.cpu().numpy() if torch.is_tensor(rot) else rot)
                    
                    dof_pos_dict = robot_state['dof_pos']
                    dof_pos_t = [dof_pos_dict[j] for j in joint_names if j in dof_pos_dict]
                    real_dof_pos.append(dof_pos_t)
                else:
                    log.error(f"Could not find robot '{robot_name}' in all_states at frame {t}")
                    break
            
            if len(real_dof_pos) == prefix_frames:
                prefix_data['dof_pos'] = np.array(real_dof_pos)
                prefix_data['root_pos'] = np.array(real_root_pos)
                prefix_data['root_rot'] = np.array(real_root_rot)
                log.info(f"Successfully overwrote dof_pos, root_pos, root_rot in prefix_data")
            else:
                log.warning(f"Failed to extract real data from all_states, using pkl data for prefix.")
        else:
            log.warning("Not enough data in all_states to overwrite prefix, using pkl data.")
        # --- END MODIFICATION ---

        dart_generator.joint_names = list(scenario.robots[0].actuators.keys())
        
        robot_name = scenario.robots[0].name
        if all_states and len(all_states) > 0 and len(all_states[0]) > 0:
            first_state = all_states[0][0]
            
            robot_state = None
            if robot_name in first_state:
                robot_state = first_state[robot_name]
            elif 'robots' in first_state and robot_name in first_state['robots']:
                robot_state = first_state['robots'][robot_name]
            
            if robot_state is not None:
                if 'pos' in robot_state:
                    pos_value = robot_state['pos']
                    if isinstance(pos_value, torch.Tensor):
                        dart_generator.global_root_pos = pos_value.float().cpu()
                    else:
                        dart_generator.global_root_pos = torch.tensor(pos_value, dtype=torch.float32)
                    log.info(f"Set DART global_root_pos from trajectory: {dart_generator.global_root_pos.numpy()}")
                
                if 'rot' in robot_state:
                    rot_value = robot_state['rot']
                    if isinstance(rot_value, torch.Tensor):
                        dart_generator.global_root_rot = rot_value.float().cpu()
                    else:
                        dart_generator.global_root_rot = torch.tensor(rot_value, dtype=torch.float32)
                    log.info(f"Set DART global_root_rot from trajectory: {dart_generator.global_root_rot.numpy()}")
            else:
                log.warning(f"Could not find robot '{robot_name}' in first_state")
        else:
            log.warning("No states available to initialize DART position")
        
        dart_generator.initialize_from_prefix(prefix_data)
        log.info(f"DART initialized with {prefix_frames} frames from pkl file")

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
    use_policy = policy_generator is not None
    
    while True:
        # Check for 'q' key press to quit
        if (not args.headless and
            hasattr(env, 'handler') and
            hasattr(env.handler, 'renderer') and
            hasattr(env.handler.renderer, 'window') and
            env.handler.renderer.window.get_key_state(ord('Q'))):
            log.info("'Q' key pressed, stopping simulation.")
            break
        
        # Determine which control mode to use
        use_dart_this_step = (dart_generator is not None and 
                              step >= args.dart_prefix_frames and
                              not use_policy)
        use_policy_this_step = use_policy and not args.object_states
        
        if use_dart_this_step:
            if args.object_states:
                # Generate new states if buffer is empty
                if dart_generated_states is None or dart_state_idx >= len(dart_generated_states[0]):
                    log.info(f"Step {step}: Generating new DART motion segment (kinematic)...")
                    
                    dart_motion_dict = dart_generator.generate_next_frames()
                    dart_generated_states = dart_generator.convert_dart_to_states(
                        dart_motion_dict, scenario.robots[0]
                    )
                    
                    total_dart_frames_generated += len(dart_generated_states[0])
                    dart_state_idx = 0
                    
                    log.info(f"Generated {len(dart_generated_states[0])} state frames (total: {total_dart_frames_generated})")
                
                dart_robot_states = get_states(dart_generated_states, dart_state_idx, args.num_envs)
                
                template_step = min(step, len(all_states[0]) - 1) if all_states and len(all_states[0]) > 0 else 0
                base_states = get_states(all_states, template_step, args.num_envs) if all_states else None
                
                states = []
                for env_idx in range(args.num_envs):
                    state = {}
                    
                    if base_states and env_idx < len(base_states):
                        if 'objects' in base_states[env_idx]:
                            state['objects'] = base_states[env_idx]['objects']
                        else:
                            state['objects'] = {}
                    else:
                        state['objects'] = {}
                    
                    if env_idx < len(dart_robot_states):
                        if 'robots' in dart_robot_states[env_idx]:
                            state['robots'] = dart_robot_states[env_idx]['robots']
                        else:
                            robot_name = scenario.robots[0].name
                            if robot_name in dart_robot_states[env_idx]:
                                state['robots'] = {robot_name: dart_robot_states[env_idx][robot_name]}
                            else:
                                state['robots'] = dart_robot_states[env_idx]
                    
                    states.append(state)
                
                dart_state_idx += 1
            else:
                # Generate new actions if buffer is empty
                if dart_generated_actions is None or dart_action_idx >= len(dart_generated_actions[0]):
                    log.info(f"Step {step}: Generating new DART motion segment (dynamic)...")

                    dart_motion_dict = dart_generator.generate_next_frames()
                    dart_generated_actions = dart_generator.convert_dart_to_actions(
                        dart_motion_dict, scenario.robots[0]
                    )

                    total_dart_frames_generated += len(dart_generated_actions[0])
                    dart_action_idx = 0
                    
                    log.info(f"Generated {len(dart_generated_actions[0])} action frames (total: {total_dart_frames_generated})")
                
                actions = get_actions(dart_generated_actions, dart_action_idx, args.num_envs, scenario.robots[0])
                dart_action_idx += 1
        elif use_policy_this_step:
            # Use policy to generate actions
            actions = []
            for env_idx in range(args.num_envs):
                action = policy_generator.predict_action(obs, scenario.robots[0])
                actions.append(action)
            
            if step % 100 == 0:
                log.info(f"Step {step}: Using policy to generate action")
        else:
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

        if args.stop_on_runout and not use_dart_this_step and not use_policy_this_step:
            if args.object_states:
                if step >= len(all_states[0]):
                    log.info("Run out of original states")
                    if dart_generator is None and policy_generator is None:
                        log.info("No DART generator or policy, stopping")
                        break
            else:
                if get_runout(all_actions, step):
                    log.info("Run out of original actions")
                    if dart_generator is None and policy_generator is None:
                        log.info("No DART generator or policy, stopping")
                        break

    obs_saver.save()
    
    if dart_generator is not None:
        log.info(f"DART Generation Summary:")
        log.info(f"  - Total segments generated: {dart_generator.segment_count}")
        log.info(f"  - Total frames generated: {total_dart_frames_generated}")
    
    return obs_saver


def main():
    render_cfg = RenderCfg(mode=args.render_mode)
    
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

    motion_data = None
    if args.motion_config:
        motion_data = load_motion_data(args.motion_config)

    dart_generator = None
    policy_generator = None
    
    if args.use_dart:
        log.info("="*80)
        log.info("DART MOTION GENERATION ENABLED")
        log.info(f"Text prompt: {args.text_prompt}")
        log.info(f"Prefix frames: {args.dart_prefix_frames}")
        log.info(f"Motion pkl: {args.motion_pkl_path}")
        log.info("="*80)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dart_generator = DARTMotionGenerator(args, device=device)
    
    if args.use_policy:
        log.info("="*80)
        log.info("POLICY CONTROL ENABLED")
        log.info(f"Policy type: {args.policy_type}")
        log.info(f"Policy checkpoint: {args.policy_checkpoint}")
        if args.policy_config:
            log.info(f"Policy config: {args.policy_config}")
        log.info("="*80)
        
        if args.object_states:
            log.warning("Policy mode is not compatible with object_states=True")
            log.warning("Policy will be ignored. Set object_states=False to use policy.")
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            policy_generator = PolicyGenerator(args, device=device)

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
        
        obs_saver = replay_single_trajectory_with_dart(
            env, scenario, traj_path, args, obs_saver, motion_data, dart_generator, policy_generator
        )

        if dart_generator is not None and args.save_enhanced_pkl_dir:
            complete_motion_dict = dart_generator.export_complete_motion_pkl(scenario.robots[0])
            complete_pkl_path = os.path.join(args.save_enhanced_pkl_dir, f"{traj_basename}_complete_motion.pkl")
            os.makedirs(os.path.dirname(complete_pkl_path), exist_ok=True)
            
            log.info(f"Saving complete motion pkl to {complete_pkl_path}")
            with open(complete_pkl_path, 'wb') as f:
                pickle.dump(complete_motion_dict, f)
            log.info(f"Complete motion pkl saved successfully!")

        if args.save_enhanced_pkl_dir and motion_data is not None:
            output_pkl_path = os.path.join(args.save_enhanced_pkl_dir, f"{traj_basename}_enhanced.pkl")
            
            # --- MODIFICATION: Always use newly rendered images ---
            rgb_images_to_save = obs_saver.resized_images
            log.info(f"Using newly rendered sapien images for saving: {len(rgb_images_to_save)} frames")
            
            save_enhanced_pkl(
                output_pkl_path, 
                motion_data, 
                rgb_images_to_save,
                args.robot_urdf if args.robot_urdf else "",
                device="cpu"
            )
            # --- END MODIFICATION ---

    env.close()
    log.info("All trajectories processed successfully!")


if __name__ == "__main__":
    main()