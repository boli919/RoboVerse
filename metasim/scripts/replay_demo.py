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
    import isaacgym  # noqa: F401
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
import torchvision.transforms as T  # for clip transforms

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
    scenes: list[str] | None = None
    render: RenderCfg = RenderCfg()
    random: RandomizationCfg = RandomizationCfg()

    sim: Literal[
        "isaaclab",
        "isaacgym",
        "genesis",
        "pybullet",
        "sapien2",
        "sapien3",
        "mujoco",
        "mjx",
    ] = "sapien3"
    renderer: Literal[
        "isaaclab",
        "isaacgym",
        "genesis",
        "pybullet",
        "mujoco",
        "sapien2",
        "sapien3",
    ] | None = None

    num_envs: int = 1
    env_spacing: float = 5.0  # 环境之间的距离（米）
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

    first_person_view: bool = True
    head_link_name: str = "head_link"
    camera_offset: tuple[float, float, float] = (0.1, 0.0, 0.5)
    camera_direction: tuple[float, float, float] = (1.0, 0.0, -0.577)

    motion_config: str | None = None
    motion_configs: list[str] | None = None  # 多个motion配置文件（每个环境一个）
    save_enhanced_pkl_dir: str | None = None
    robot_urdf: str | None = None
    image_size: int = 224  # 你之前存 pkl 用的 224×224

    # perturbation 相关
    enable_perturbation: bool = False
    perturb_prob: float = 0.1
    perturb_force_range: tuple[float, float] = (5.0, 20.0)
    perturb_torque_range: tuple[float, float] = (0.5, 2.0)
    perturb_action_noise: float = 0.01
    perturb_interval: int = 1
    perturb_position_strength: float = 1.0
    perturb_rotation_strength: float = 0.3
    perturb_damping: float = 0.92
    save_trajectory_plot: str | None = None

    # CLIP
    use_clip: bool = False
    clip_device: str = "cuda"
    clip_model_name: str = "ViT-B/32"

    def __post_init__(self):
        log.info(f"Args: {self}")
        if self.enable_perturbation:
            log.info("🌊 perturbation enabled")


args = tyro.cli(Args)


def get_actions(all_actions, action_idx: int, num_envs: int, robot: BaseRobotCfg):
    envs_actions = all_actions[:num_envs]
    return [
        env_actions[action_idx] if action_idx < len(env_actions) else env_actions[-1]
        for env_actions in envs_actions
    ]


def get_states(all_states, action_idx: int, num_envs: int):
    envs_states = all_states[:num_envs]
    return [
        env_states[action_idx] if action_idx < len(env_states) else env_states[-1]
        for env_states in envs_states
    ]


def get_runout(all_actions, action_idx: int):
    return all(action_idx >= len(a) for a in all_actions)


_perturbation_state = {}
_trajectory_data = {}


def init_perturbation_state(num_envs: int):
    global _perturbation_state, _trajectory_data
    _perturbation_state = {
        "pos_offset": [np.zeros(3) for _ in range(num_envs)],
        "rot_offset_euler": [np.zeros(3) for _ in range(num_envs)],
        "velocity": [np.zeros(3) for _ in range(num_envs)],
        "ang_velocity": [np.zeros(3) for _ in range(num_envs)],
    }
    _trajectory_data = {
        "original_positions": [[] for _ in range(num_envs)],
        "perturbed_positions": [[] for _ in range(num_envs)],
        "original_rotations": [[] for _ in range(num_envs)],
        "perturbed_rotations": [[] for _ in range(num_envs)],
    }


def perturb_states(states, args, step: int, num_envs: int):
    if not args.enable_perturbation:
        return states

    import random
    from copy import deepcopy
    from scipy.spatial.transform import Rotation as R

    global _perturbation_state
    if not _perturbation_state:
        init_perturbation_state(num_envs)

    out_states = []
    for env_id, state in enumerate(states):
        s = deepcopy(state)

        # random push
        rand_pos = np.array(
            [
                random.gauss(0, 0.001 * args.perturb_position_strength),
                random.gauss(0, 0.001 * args.perturb_position_strength),
                random.gauss(0, 0.0003 * args.perturb_position_strength),
            ]
        )
        rand_rot = np.array(
            [
                random.gauss(0, 0.002 * args.perturb_rotation_strength),
                random.gauss(0, 0.002 * args.perturb_rotation_strength),
                random.gauss(0, 0.001 * args.perturb_rotation_strength),
            ]
        )
        # spring back
        k = 0.15
        restoring_pos = -k * _perturbation_state["pos_offset"][env_id]
        restoring_rot = -k * _perturbation_state["rot_offset_euler"][env_id]

        _perturbation_state["velocity"][env_id] += rand_pos + restoring_pos
        _perturbation_state["ang_velocity"][env_id] += rand_rot + restoring_rot

        _perturbation_state["velocity"][env_id] *= args.perturb_damping
        _perturbation_state["ang_velocity"][env_id] *= args.perturb_damping

        _perturbation_state["pos_offset"][env_id] += _perturbation_state["velocity"][env_id]
        _perturbation_state["rot_offset_euler"][env_id] += _perturbation_state["ang_velocity"][env_id]

        pos_max = 0.15 * args.perturb_position_strength
        rot_max = 0.3 * args.perturb_rotation_strength
        _perturbation_state["pos_offset"][env_id] = np.clip(
            _perturbation_state["pos_offset"][env_id], -pos_max, pos_max
        )
        _perturbation_state["rot_offset_euler"][env_id] = np.clip(
            _perturbation_state["rot_offset_euler"][env_id], -rot_max, rot_max
        )

        if "robots" in s:
            for rname, rdata in s["robots"].items():
                if "pos" in rdata:
                    orig = rdata["pos"]
                    pert = orig + _perturbation_state["pos_offset"][env_id]
                    s["robots"][rname]["pos"] = pert

                    global _trajectory_data
                    if _trajectory_data:
                        orig_np = (
                            orig.cpu().numpy() if hasattr(orig, "cpu") else np.array(orig)
                        )
                        pert_np = (
                            pert.cpu().numpy() if hasattr(pert, "cpu") else np.array(pert)
                        )
                        _trajectory_data["original_positions"][env_id].append(orig_np)
                        _trajectory_data["perturbed_positions"][env_id].append(pert_np)

                if "rot" in rdata:
                    orig = rdata["rot"]
                    orig_xyzw = np.array([orig[1], orig[2], orig[3], orig[0]])
                    R_orig = R.from_quat(orig_xyzw)
                    R_off = R.from_euler("xyz", _perturbation_state["rot_offset_euler"][env_id])
                    R_pert = R_off * R_orig
                    pert_xyzw = R_pert.as_quat()
                    s["robots"][rname]["rot"] = np.array(
                        [pert_xyzw[3], pert_xyzw[0], pert_xyzw[1], pert_xyzw[2]]
                    )
                    if _trajectory_data:
                        _trajectory_data["original_rotations"][env_id].append(
                            R_orig.as_euler("xyz")
                        )
                        _trajectory_data["perturbed_rotations"][env_id].append(
                            R_pert.as_euler("xyz")
                        )

                if args.perturb_action_noise > 0.0 and "dof_pos" in rdata:
                    for jn, pos in rdata["dof_pos"].items():
                        noise = random.gauss(0, args.perturb_action_noise * 0.1)
                        s["robots"][rname]["dof_pos"][jn] = pos + noise

        out_states.append(s)
    return out_states


def plot_trajectory_comparison(save_path: str):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    global _trajectory_data
    if not _trajectory_data or not _trajectory_data["original_positions"]:
        log.warning("No trajectory data to plot")
        return
    n = len(_trajectory_data["original_positions"])
    fig = plt.figure(figsize=(20, 10))
    for i in range(n):
        if not _trajectory_data["original_positions"][i]:
            continue
        orig = np.array(_trajectory_data["original_positions"][i])
        pert = np.array(_trajectory_data["perturbed_positions"][i])

        ax = fig.add_subplot(2, n, i + 1, projection="3d")
        ax.plot(orig[:, 0], orig[:, 1], orig[:, 2], "b-", label="orig")
        ax.plot(pert[:, 0], pert[:, 1], pert[:, 2], "r--", label="pert")
        ax.legend()
        ax.set_title(f"Env {i} 3D")

        ax2 = fig.add_subplot(2, n, n + i + 1)
        ax2.plot(orig[:, 0], orig[:, 1], "b-", label="orig")
        ax2.plot(pert[:, 0], pert[:, 1], "r--", label="pert")
        ax2.legend()
        ax2.axis("equal")
        ax2.set_title(f"Env {i} top")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info(f"plot saved to {save_path}")
    plt.close()


def add_action_noise(actions, args):
    if args.perturb_action_noise <= 0.0:
        return actions
    import random

    out = []
    for a in actions:
        if a and "dof_pos_target" in a:
            na = a.copy()
            nd = {}
            for jn, pos in a["dof_pos_target"].items():
                nd[jn] = pos + random.gauss(0, args.perturb_action_noise)
            na["dof_pos_target"] = nd
            out.append(na)
        else:
            out.append(a)
    return out


def quat_xyzw_to_rotmat(q: torch.Tensor) -> torch.Tensor:
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

    return torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )


def get_actuated_joint_indices(kin) -> List[int]:
    idxs: List[int] = []
    for j in range(1, kin.num_joint):
        if kin.joint_dof_idx[j] != -1:
            idxs.append(j)
    return idxs


class EnhancedObsSaver:
    """
    - 保存大图(可选)
    - 缩成 224x224 存着
    - 再用你截图的 CLIP transform 把“原始大图”过一遍，得到 per-frame CLIP feature
    """

    def __init__(
        self,
        image_dir: str | None = None,
        video_path: str | None = None,
        save_quality: int = 95,
        image_size: int = 224,
        clip_model=None,
        clip_img_transform=None,
        clip_device: str = "cpu",
        save_clip_feats: bool = False,
    ):
        self.image_dir = image_dir
        self.video_path = video_path
        self.save_quality = save_quality
        self.image_size = image_size
        self.images: list[NDArray] = []          # 原始大图
        self.resized_images: list[NDArray] = []  # 224x224
        self.image_idx = 0

        self.clip_model = clip_model
        self.clip_img_transform = clip_img_transform
        self.clip_device = clip_device
        self.save_clip_feats = (
            save_clip_feats
            and (clip_model is not None)
            and (clip_img_transform is not None)
        )
        self.clip_feats: list[np.ndarray] = []   # 这里会存每一帧的 CLIP feature

    def add(self, state: TensorState):
        if (
            self.image_dir is None
            and self.video_path is None
            and not self.save_clip_feats
        ):
            return

        try:
            rgb_data = next(iter(state.cameras.values())).rgb
            image = make_grid(
                rgb_data.permute(0, 3, 1, 2) / 255,
                nrow=int(rgb_data.shape[0] ** 0.5),
            )
        except Exception as e:
            log.error(f"Error adding obs: {e}")
            return

        # 1) 原始大图 (H,W,3) uint8
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)

        # 1.1 可选保存到文件
        if self.image_dir is not None:
            os.makedirs(self.image_dir, exist_ok=True)
            Image.fromarray(img_np).save(
                os.path.join(self.image_dir, f"rgb_{self.image_idx:04d}.png"),
                quality=self.save_quality,
                optimize=False,
            )

        self.images.append(img_np)
        self.image_idx += 1

        # 2) 把要写进 pkl 的做成 224×224 (你之前就是这个)
        resized_pil = Image.fromarray(img_np).resize(
            (self.image_size, self.image_size),
            Image.LANCZOS,
        )
        resized_np = np.asarray(resized_pil, dtype=np.uint8)
        self.resized_images.append(resized_np)

        # 3) CLIP 一帧一帧 encode
        if self.save_clip_feats:
            try:
                # 这里就是你截图的那一套：ToPILImage -> CenterCrop(720) -> Resize(224,224) -> ToTensor -> Normalize
                clip_tensor = self.clip_img_transform(img_np.astype(np.uint8))
                clip_tensor = clip_tensor.unsqueeze(0).to(self.clip_device)
                with torch.no_grad():
                    feat = self.clip_model.encode_image(clip_tensor).float()
                    feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                self.clip_feats.append(feat.cpu().numpy()[0])
            except Exception as e:
                log.error(f"[CLIP] frame {self.image_idx-1} encode failed: {e}")
                # 保持长度一致
                self.clip_feats.append(np.zeros((512,), dtype=np.float32))

    def save(self):
        if self.video_path is not None and self.images:
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(
                self.video_path,
                self.images,
                fps=30,
                quality=8,
                codec="libx264" if self.video_path.endswith(".mp4") else None,
            )


def update_camera_poses(env, args):
    if not args.first_person_view:
        return
    if not hasattr(env, "handler"):
        return
    handler = env.handler
    if not hasattr(handler, "robot") or not handler.robot:
        return
    if not hasattr(handler, "camera_ids") or not handler.camera_ids:
        return
    robot_name = handler.robot.name
    try:
        head_link = None
        for link in handler.link_ids.get(robot_name, []):
            if link.get_name() == args.head_link_name:
                head_link = link
                break
        if head_link is None:
            return
        head_pose = head_link.get_pose()
        head_pos = head_pose.p
        head_rot = head_pose.q

        from scipy.spatial.transform import Rotation as R

        offset = np.array(args.camera_offset)
        rot = R.from_quat([head_rot[1], head_rot[2], head_rot[3], head_rot[0]])
        offset_world = rot.apply(offset)
        cam_pos = head_pos + offset_world
        dir_local = np.array(args.camera_direction)
        dir_world = rot.apply(dir_local)
        look_at = cam_pos + dir_world
        for cname, _cid in handler.camera_ids.items():
            handler.set_camera_look_at(cname, cam_pos, look_at)
        handler.refresh_render()
    except Exception as e:
        log.error(f"update_camera_poses error: {e}")


def load_motion_data(motion_config_path: str):
    with open(motion_config_path, "r") as f:
        cfg = yaml.safe_load(f)
    motion_file = cfg.get("motion_file")
    if motion_file:
        with open(motion_file, "rb") as f:
            return pickle.load(f)
    root_path = cfg.get("root_path", "")
    motions = cfg.get("motions", [])
    if not motions:
        raise ValueError(f"No motions in {motion_config_path}")
    motion_file_rel = motions[0].get("file")
    if not motion_file_rel:
        raise ValueError("first motion has no file")
    motion_file = os.path.join(root_path, motion_file_rel)
    with open(motion_file, "rb") as f:
        return pickle.load(f)


def save_enhanced_pkl(
    output_path: str,
    motion_data: dict,
    resized_images: list,
    urdf_path: str,
    device: str = "cpu",
    clip_image_embeds: list[np.ndarray] | None = None,
):
    """
    NOTE: 你说“不要弄个新的键，直接覆盖掉原来的 rgb_images”，
    所以这里的逻辑是：

    - 如果传进来 clip_image_embeds，就用它覆盖 out["rgb_images"]
    - 否则就用原来的 224x224 resized_images 写到 out["rgb_images"]
    """
    log.info("Preparing enhanced pkl ...")

    if urdf_path and not os.path.isabs(urdf_path):
        abs1 = os.path.join(project_root, urdf_path)
        if os.path.exists(abs1):
            urdf_path = abs1
        else:
            abs2 = os.path.join(os.getcwd(), urdf_path)
            if os.path.exists(abs2):
                urdf_path = abs2

    required = ["root_pos", "root_rot", "dof_pos"]
    for k in required:
        if k not in motion_data:
            raise ValueError(f"Missing required key {k}")

    root_pos = torch.tensor(np.asarray(motion_data["root_pos"], np.float32), device=device)
    root_rot_wxyz = torch.tensor(np.asarray(motion_data["root_rot"], np.float32), device=device)
    dof_pos = torch.tensor(np.asarray(motion_data["dof_pos"], np.float32), device=device)

    T_len = dof_pos.shape[0]
    N = dof_pos.shape[1]

    out = dict(motion_data)

    # dof_rot 部分跟你原来的保持一致
    if not urdf_path or not os.path.exists(urdf_path):
        log.warning("URDF missing, skip dof_rot")
    else:
        try:
            from motion_tools.utils.kinematics_model import KinematicsModel

            kin = KinematicsModel(file_path=urdf_path, device=device)
            if kin.num_dof != N:
                log.warning(f"URDF dof {kin.num_dof} != data dof {N}, will clip")
                N = min(kin.num_dof, N)
                dof_pos = dof_pos[:, :N]
            act_idxs = get_actuated_joint_indices(kin)
            num_act = len(act_idxs)
            dof_rot = torch.zeros((T_len, num_act, 3, 3), dtype=torch.float32, device=device)

            rr_xyzw = torch.stack(
                [
                    root_rot_wxyz[:, 1],
                    root_rot_wxyz[:, 2],
                    root_rot_wxyz[:, 3],
                    root_rot_wxyz[:, 0],
                ],
                dim=-1,
            )

            bs = 512
            for st in range(0, T_len, bs):
                ed = min(st + bs, T_len)
                rp = root_pos[st:ed]
                rr = rr_xyzw[st:ed]
                dq = dof_pos[st:ed]
                bpos, brot = kin.forward_kinematics(rp, rr, dq)
                brot_sel = brot[:, act_idxs, :]
                rotmat = quat_xyzw_to_rotmat(brot_sel.reshape(-1, 4)).reshape(
                    -1, num_act, 3, 3
                )
                dof_rot[st:ed] = rotmat
            out["dof_rot"] = dof_rot.cpu().numpy()
        except Exception as e:
            log.error(f"dof_rot failed: {e}")

    # ====== 这里是关键 ======
    # 如果有 clip 的结果，就直接覆盖 rgb_images
    if clip_image_embeds is not None and len(clip_image_embeds) > 0:
        if len(clip_image_embeds) != T_len:
            log.warning(
                f"CLIP frames {len(clip_image_embeds)} != motion frames {T_len}, will trim"
            )
        clip_arr = np.asarray(clip_image_embeds[:T_len], dtype=np.float32)
        out["rgb_images"] = clip_arr
        log.info(f"rgb_images OVERWRITTEN by CLIP features, shape={clip_arr.shape}")
    else:
        # 否则就还是你原来那种 224x224 图
        if resized_images:
            if len(resized_images) != T_len:
                log.warning(
                    f"image count {len(resized_images)} != motion frames {T_len}, trim"
                )
            out["rgb_images"] = resized_images[:T_len]
            log.info(
                f"rgb_images saved as 224x224 images, shape={(len(resized_images[:T_len]),)}"
            )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(out, f)
    log.info(f"Enhanced pkl saved to {output_path}, keys={list(out.keys())}")


def replay_single_trajectory(env, scenario, traj_path, args, obs_saver, motion_data=None):
    log.info(f"Replaying trajectory: {traj_path}")

    motion_len = None
    if motion_data is not None and "dof_pos" in motion_data:
        motion_len = len(motion_data["dof_pos"])

    assert os.path.exists(traj_path), f"{traj_path} not exist"
    orig_traj = scenario.task.traj_filepath
    scenario.task.traj_filepath = traj_path
    init_states, all_actions, all_states = get_traj(
        scenario.task, scenario.robots[0], env.handler
    )
    scenario.task.traj_filepath = orig_traj

    if args.robot_height_offset != 0.0:
        for s in init_states[: args.num_envs]:
            if hasattr(s, "root_pos"):
                s.root_pos[2] += args.robot_height_offset

    obs, _ = env.reset(states=init_states[: args.num_envs])

    if args.first_person_view:
        update_camera_poses(env, args)
    obs_saver.add(obs)

    step = 0
    while True:
        if motion_len is not None and step >= motion_len:
            log.info("reached motion length, stop")
            break

        if args.object_states:
            if all_states is None:
                raise ValueError("all_states is None")
            states = get_states(all_states, step, args.num_envs)
            states = perturb_states(states, args, step, args.num_envs)
            env.handler.set_states(states)
            if args.first_person_view:
                update_camera_poses(env, args)
            env.handler.refresh_render()
            obs = env.handler.get_states()
            success = env.handler.task.checker.check(env.handler)
            if success.all():
                break
        else:
            actions = get_actions(all_actions, step, args.num_envs, scenario.robots[0])
            actions = add_action_noise(actions, args)
            obs, reward, success, time_out, extras = env.step(actions)
            if args.first_person_view:
                update_camera_poses(env, args)
            if success.all() or time_out.all():
                break

        obs_saver.add(obs)
        step += 1

        if args.stop_on_runout and get_runout(all_actions, step):
            log.info("actions run out, stop")
            break

    obs_saver.save()
    if args.save_trajectory_plot:
        plot_trajectory_comparison(args.save_trajectory_plot)
    return obs_saver


def main():
    render_cfg = RenderCfg(mode=args.render_mode)

    sensor_width_mm = 36.0
    fx_pixels = 1386.4
    width_pixels = 1920
    horizontal_aperture = (width_pixels / fx_pixels) * sensor_width_mm
    focal_length = fx_pixels * sensor_width_mm / width_pixels

    if args.first_person_view:
        camera = PinholeCameraCfg(
            pos=(0.0, 0.0, 0.0),
            look_at=(1.0, 0.0, 0.0),
            width=args.camera_width,
            height=args.camera_height,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
        )
    else:
        camera = PinholeCameraCfg(
            pos=(2.5, 0.0, 2.5),
            look_at=(-3.0, 0.0, 0.0),
            width=args.camera_width,
            height=args.camera_height,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
        )

    scenario = ScenarioCfg(
        task=args.task,
        robots=[args.robot],
        scene=args.scene,
        scenes=args.scenes,
        cameras=[camera],
        random=args.random,
        render=render_cfg,
        sim=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        env_spacing=args.env_spacing,  # 添加环境间距配置
        try_add_table=args.try_add_table,
        object_states=args.object_states,
        split=args.split,
        headless=args.headless,
    )

    # create env
    if scenario.renderer is None:
        env_class = get_sim_env_class(SimType(scenario.sim))
        env = env_class(scenario)
    else:
        env_render = get_sim_env_class(SimType(scenario.renderer))(scenario)
        env_phys = get_sim_env_class(SimType(scenario.sim))(scenario)
        env = HybridSimEnv(env_phys, env_render)

    # ===== load CLIP (按你截图的）=====
    clip_model = None
    clip_img_transform = None
    clip_device = "cpu"
    if args.use_clip:
        clip_device = args.clip_device
        try:
            import clip as _clip
            # 先试你工程里的 load_and_freeze_clip
            load_and_freeze_clip = None
            try:
                from clip_utils import load_and_freeze_clip  # 换成你工程里的路径
            except Exception:
                try:
                    from motion_tools.utils.clip_utils import load_and_freeze_clip
                except Exception:
                    load_and_freeze_clip = None

            if load_and_freeze_clip is not None:
                clip_model = load_and_freeze_clip(
                    clip_version=args.clip_model_name, device=clip_device
                )
                log.info(f"[CLIP] loaded via load_and_freeze_clip on {clip_device}")
            else:
                clip_model, _ = _clip.load(args.clip_model_name, device=clip_device)
                clip_model.eval()
                log.info(f"[CLIP] loaded via clip.load on {clip_device}")

            # 这里完全复刻你截图里的 transform
            clip_mean = (0.48145466, 0.4578275, 0.40821073)
            clip_std = (0.26862954, 0.26130258, 0.27577711)
            clip_img_transform = T.Compose(
                [
                    T.ToPILImage(),
                    T.CenterCrop(720),
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=clip_mean, std=clip_std),
                ]
            )
        except Exception as e:
            log.error(f"[CLIP] load failed: {e}")
            args.use_clip = False

    # 加载 motion 数据（支持多个）
    motion_data_list = []
    if args.motion_configs:
        # 多个 motion configs（每个环境一个）
        log.info(f"Loading {len(args.motion_configs)} motion configs for {args.num_envs} envs")
        for i, cfg_path in enumerate(args.motion_configs):
            motion_data = load_motion_data(cfg_path)
            motion_data_list.append(motion_data)
            log.info(f"  Env {i}: {cfg_path}")
        
        # 如果 motion_configs 数量少于 num_envs，循环使用
        if len(motion_data_list) < args.num_envs:
            log.warning(f"Only {len(motion_data_list)} motions for {args.num_envs} envs, will cycle")
            while len(motion_data_list) < args.num_envs:
                motion_data_list.append(motion_data_list[len(motion_data_list) % len(args.motion_configs)])
    elif args.motion_config:
        # 单个 motion config（所有环境共享）
        motion_data = load_motion_data(args.motion_config)
        motion_data_list = [motion_data] * args.num_envs
        log.info(f"Using single motion config for all {args.num_envs} envs")
    else:
        motion_data_list = None

    trajs = scenario.task.traj_filepath
    if not isinstance(trajs, list):
        trajs = [trajs]

    if args.save_enhanced_pkl_dir:
        os.makedirs(args.save_enhanced_pkl_dir, exist_ok=True)

    for traj_path in trajs:
        base = os.path.splitext(os.path.basename(traj_path))[0]

        img_dir = os.path.join(args.save_image_dir, base) if args.save_image_dir else None
        vid_path = None
        if args.save_video_path:
            ext = os.path.splitext(args.save_video_path)[1]
            vid_path = os.path.join(os.path.dirname(args.save_video_path), f"{base}{ext}")

        obs_saver = EnhancedObsSaver(
            image_dir=img_dir,
            video_path=vid_path,
            save_quality=args.save_quality,
            image_size=args.image_size,
            clip_model=clip_model,
            clip_img_transform=clip_img_transform,
            clip_device=clip_device,
            save_clip_feats=args.use_clip,
        )

        obs_saver = replay_single_trajectory(
            env, scenario, traj_path, args, obs_saver, motion_data_list
        )

        if args.save_enhanced_pkl_dir and motion_data is not None:
            out_pkl = os.path.join(
                args.save_enhanced_pkl_dir, f"{base}_enhanced.pkl"
            )
            save_enhanced_pkl(
                out_pkl,
                motion_data,
                obs_saver.resized_images,
                args.robot_urdf if args.robot_urdf else "",
                device="cpu",
                clip_image_embeds=obs_saver.clip_feats,  # 👈 这里传进去，但在函数里会覆盖 rgb_images
            )

    env.close()


if __name__ == "__main__":
    main()
