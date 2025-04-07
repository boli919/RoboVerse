from __future__ import annotations

import math

import numpy as np
import torch
from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401
from loguru import logger as log

from metasim.cfg.objects import ArticulationObjCfg, BaseObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.robots.base_robot_cfg import BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler, EnvWrapper, GymEnvWrapper
from metasim.types import Action, EnvState


class IsaacgymHandler(BaseSimHandler):
    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self._robot: BaseRobotCfg = scenario.robot
        self._robot_names = {self._robot.name}
        self._robot_init_pose = (0, 0, 0) if not self.robot.default_position else self.robot.default_position
        self._cameras = scenario.cameras

        self.gym = None
        self.sim = None
        self.viewer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._num_envs: int = scenario.num_envs
        self._episode_length_buf = [0 for _ in range(self.num_envs)]

        # asset related
        self._asset_dict_dict: dict = {}  # dict of object link index dict
        self._articulated_asset_dict_dict: dict = {}  # dict of articulated object link index dict
        self._articulated_joint_dict_dict: dict = {}  # dict of articulated object joint index dict
        self._robot_link_dict: dict = {}  # dict of robot link index dict
        self._robot_joint_dict: dict = {}  # dict of robot joint index dict
        self._joint_info: dict = {}  # dict of joint names of each env
        self._num_joints: int = 0
        self._body_info: dict = {}  # dict of body names of each env
        self._num_bodies: int = 0

        # environment related pointers
        self._envs: list = []
        self._obj_handles: list = []  # 2 dim list: list in list, each list contains object handles of each env
        self._articulated_obj_handles: list = []  # 2 dim list: list in list, each list contains articulated object handles of each env
        self._robot_handles: list = []  # 2 dim list: list of robot handles of each env

        # environment related tensor indices
        self._env_rigid_body_global_indices: list = []  # 2 dim list: list in list, each list contains global indices of each env

        self._root_states: torch.Tensor | None = None  # will update after refresh
        self._dof_states: torch.Tensor | None = None  # will update after refresh
        self._rigid_body_states: torch.Tensor | None = None  # will update after refresh

    def launch(self) -> None:
        ## IsaacGym Initialization
        self._init_gym()
        self._make_envs()
        self._set_up_camera()
        # ==== prepare tensors =====
        # from now on, we will use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim)
        self._root_states = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self._dof_states = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self._rigid_body_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))

    def _init_gym(self) -> None:
        physics_engine = gymapi.SIM_PHYSX
        self.gym = gymapi.acquire_gym()
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.num_threads = 0
        sim_params.physx.use_gpu = True

        compute_device_id = 0
        graphics_device_id = 0
        self.sim = self.gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")

    def _set_up_camera(self) -> None:
        self._depth_tensors = []
        self._rgb_tensors = []
        self._seg_tensors = []
        self._vinv_mats = []
        self._proj_mats = []
        self._camera_handles = []
        self._env_origin = []
        for i_env in range(self.num_envs):
            self._depth_tensors.append([])
            self._rgb_tensors.append([])
            self._seg_tensors.append([])
            self._vinv_mats.append([])
            self._proj_mats.append([])
            self._env_origin.append([])

            origin = self.gym.get_env_origin(self._envs[i_env])
            self._env_origin[i_env] = [origin.x, origin.y, origin.z]
            for cam_cfg in self.cameras:
                camera_props = gymapi.CameraProperties()
                camera_props.width = cam_cfg.width
                camera_props.height = cam_cfg.height
                camera_props.horizontal_fov = cam_cfg.horizontal_fov
                camera_props.near_plane = cam_cfg.clipping_range[0]
                camera_props.far_plane = cam_cfg.clipping_range[1]
                camera_props.enable_tensors = True
                camera_handle = self.gym.create_camera_sensor(self._envs[i_env], camera_props)
                self._camera_handles.append(camera_handle)

                camera_eye = gymapi.Vec3(*cam_cfg.pos)
                camera_lookat = gymapi.Vec3(*cam_cfg.look_at)
                self.gym.set_camera_location(camera_handle, self._envs[i_env], camera_eye, camera_lookat)

                camera_tensor_depth = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self._envs[i_env], camera_handle, gymapi.IMAGE_DEPTH
                )
                camera_tensor_rgb = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self._envs[i_env], camera_handle, gymapi.IMAGE_COLOR
                )
                camera_tensor_rgb_seg = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self._envs[i_env], camera_handle, gymapi.IMAGE_SEGMENTATION
                )
                torch_cam_depth_tensor = gymtorch.wrap_tensor(camera_tensor_depth)
                torch_cam_rgb_tensor = gymtorch.wrap_tensor(camera_tensor_rgb)
                torch_cam_rgb_seg_tensor = gymtorch.wrap_tensor(camera_tensor_rgb_seg)

                cam_vinv = torch.inverse(
                    torch.tensor(self.gym.get_camera_view_matrix(self.sim, self._envs[i_env], camera_handle))
                ).to(self.device)
                cam_proj = torch.tensor(
                    self.gym.get_camera_proj_matrix(self.sim, self._envs[i_env], camera_handle),
                    device=self.device,
                )

                self._depth_tensors[i_env].append(torch_cam_depth_tensor)
                self._rgb_tensors[i_env].append(torch_cam_rgb_tensor)
                self._seg_tensors[i_env].append(torch_cam_rgb_seg_tensor)
                self._vinv_mats[i_env].append(cam_vinv)
                self._proj_mats[i_env].append(cam_proj)

    def _load_object_asset(self, object: BaseObjCfg) -> None:
        asset_root = "."
        if isinstance(object, PrimitiveCubeCfg):
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = False
            asset_options.disable_gravity = False
            asset_options.flip_visual_attachments = False
            asset = self.gym.create_box(self.sim, object.size[0], object.size[1], object.size[2], asset_options)
        elif isinstance(object, PrimitiveSphereCfg):
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = False
            asset_options.disable_gravity = False
            asset_options.flip_visual_attachments = False
            asset = self.gym.create_sphere(self.sim, object.radius, asset_options)

        elif isinstance(object, ArticulationObjCfg):
            asset_path = object.urdf_path
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = True
            asset_options.disable_gravity = False
            asset_options.flip_visual_attachments = False
            asset = self.gym.load_asset(self.sim, asset_root, asset_path, asset_options)
            self._articulated_asset_dict_dict[object.name] = self.gym.get_asset_rigid_body_dict(asset)
            self._articulated_joint_dict_dict[object.name] = self.gym.get_asset_dof_dict(asset)
        elif isinstance(object, RigidObjCfg):
            asset_path = object.urdf_path
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = False
            asset_options.disable_gravity = False
            asset_options.flip_visual_attachments = False
            asset = self.gym.load_asset(self.sim, asset_root, asset_path, asset_options)

        asset_link_dict = self.gym.get_asset_rigid_body_dict(asset)
        self._asset_dict_dict[object.name] = asset_link_dict
        return asset

    def _load_robot_assets(self) -> None:
        asset_root = "."
        robot_asset_file = self.robot.urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = self.robot.fix_base_link
        asset_options.disable_gravity = not self.robot.enabled_gravity
        asset_options.flip_visual_attachments = self.robot.isaacgym_flip_visual_attachments
        asset_options.collapse_fixed_joints = self.robot.collapse_fixed_joints
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)
        # configure robot dofs
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        robot_lower_limits = robot_dof_props["lower"]
        robot_upper_limits = robot_dof_props["upper"]
        robot_ranges = robot_upper_limits - robot_lower_limits
        robot_mids = 0.3 * (robot_upper_limits + robot_lower_limits)
        robot_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
        robot_dof_props["stiffness"][:7].fill(400.0)
        robot_dof_props["damping"][:7].fill(40.0)

        # grippers
        robot_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        robot_dof_props["stiffness"][7:].fill(800.0)
        robot_dof_props["damping"][7:].fill(40.0)

        robot_num_dofs = self.gym.get_asset_dof_count(robot_asset)
        default_dof_pos = np.zeros(robot_num_dofs, dtype=np.float32)
        default_dof_pos[:7] = robot_mids[:7]
        # grippers open
        default_dof_pos[7:] = robot_upper_limits[7:]

        default_dof_state = np.zeros(robot_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        # # get link index of panda hand, which we will use as end effector
        self._robot_link_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
        self._robot_joint_dict = self.gym.get_asset_dof_dict(robot_asset)

        return robot_asset, robot_dof_props, default_dof_state, default_dof_pos

    def _make_envs(
        self,
    ) -> None:
        # configure env grid
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        log.info("Creating %d environments" % self.num_envs)

        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(*self._robot_init_pose)

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # get object and robot asset
        obj_assets_list = [self._load_object_asset(obj) for obj in self.objects]
        robot_asset, robot_dof_props, default_dof_state, default_dof_pos = self._load_robot_assets()

        #### Joint Info ####
        for art_obj_name, art_obj_joint_dict in self._articulated_joint_dict_dict.items():
            num_joints = len(art_obj_joint_dict)
            joint_names_ = []
            for joint_i in range(num_joints):
                for joint_name, joint_idx in art_obj_joint_dict.items():
                    if joint_idx == joint_i:
                        joint_names_.append(joint_name)
            assert len(joint_names_) == num_joints
            joint_info_ = {}
            joint_info_["names"] = joint_names_
            joint_info_["local_indices"] = art_obj_joint_dict
            art_obj_joint_dict_global = {k_: v_ + self._num_joints for k_, v_ in art_obj_joint_dict.items()}
            joint_info_["global_indices"] = art_obj_joint_dict_global
            self._num_joints += num_joints
            self._joint_info[art_obj_name] = joint_info_

        # robot
        num_joints = len(self._robot_joint_dict)
        joint_names_ = []
        for joint_i in range(num_joints):
            for joint_name, joint_idx in self._robot_joint_dict.items():
                if joint_idx == joint_i:
                    joint_names_.append(joint_name)

        assert len(joint_names_) == num_joints
        joint_info_ = {}
        joint_info_["names"] = joint_names_
        joint_info_["local_indices"] = self._robot_joint_dict
        joint_info_["global_indices"] = {k_: v_ + self._num_joints for k_, v_ in self._robot_joint_dict.items()}
        self._joint_info[self.robot.name] = joint_info_
        self._num_joints += num_joints

        ###################
        #### Body Info ####
        for obj_name, asset_dict in self._asset_dict_dict.items():
            num_bodies = len(asset_dict)
            rigid_body_names = []
            for i in range(num_bodies):
                for rigid_body_name, rigid_body_idx in asset_dict.items():
                    if rigid_body_idx == i:
                        rigid_body_names.append(rigid_body_name)
            assert len(rigid_body_names) == num_bodies
            body_info_ = {}
            body_info_["name"] = rigid_body_names
            body_info_["local_indices"] = asset_dict
            body_info_["global_indices"] = {k_: v_ + self._num_bodies for k_, v_ in asset_dict.items()}
            self._body_info[obj_name] = body_info_
            self._num_bodies += num_bodies

        num_bodies = len(self._robot_link_dict)
        rigid_body_names = []
        for i in range(num_bodies):
            for rigid_body_name, rigid_body_idx in self._robot_link_dict.items():
                if rigid_body_idx == i:
                    rigid_body_names.append(rigid_body_name)

        assert len(rigid_body_names) == num_bodies
        rigid_body_info_ = {}
        rigid_body_info_["name"] = rigid_body_names
        rigid_body_info_["local_indices"] = self._robot_link_dict
        rigid_body_info_["global_indices"] = {k_: v_ + self._num_bodies for k_, v_ in self._robot_link_dict.items()}
        self._body_info[self.robot.name] = rigid_body_info_
        self._num_bodies += num_bodies

        #################

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            ##  state update  ##
            self._envs.append(env)
            self._obj_handles.append([])
            self._env_rigid_body_global_indices.append({})
            self._articulated_obj_handles.append([])
            ####################

            # carefully set each object
            for obj_i, obj_asset in enumerate(obj_assets_list):
                # add object
                obj_pose = gymapi.Transform()
                obj_pose.p.x = obj_i * 0.2  # place to any position, will update immediately at reset stage
                obj_pose.p.y = 0.0
                obj_pose.p.z = 0.0
                obj_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
                obj_handle = self.gym.create_actor(env, obj_asset, obj_pose, "object", i, 0)

                self.gym.set_actor_scale(env, obj_handle, self.objects[obj_i].scale[0])
                if isinstance(self.objects[obj_i], PrimitiveCubeCfg):
                    color = gymapi.Vec3(
                        self.objects[obj_i].color[0],
                        self.objects[obj_i].color[1],
                        self.objects[obj_i].color[2],
                    )
                    self.gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                elif isinstance(self.objects[obj_i], PrimitiveSphereCfg):
                    color = gymapi.Vec3(
                        self.objects[obj_i].color[0],
                        self.objects[obj_i].color[1],
                        self.objects[obj_i].color[2],
                    )
                    self.gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                elif isinstance(self.objects[obj_i], RigidObjCfg):
                    pass
                elif isinstance(self.objects[obj_i], ArticulationObjCfg):
                    self._articulated_obj_handles[-1].append(obj_handle)
                else:
                    log.error("Unknown object type")
                    raise NotImplementedError
                self._obj_handles[-1].append(obj_handle)

                object_rigid_body_indices = {}
                for rigid_body_name, rigid_body_idx in self._asset_dict_dict[self.objects[obj_i].name].items():
                    rigid_body_idx = self.gym.find_actor_rigid_body_index(
                        env, obj_handle, rigid_body_name, gymapi.DOMAIN_SIM
                    )
                    object_rigid_body_indices[rigid_body_name] = rigid_body_idx

                self._env_rigid_body_global_indices[-1][self.objects[obj_i].name] = object_rigid_body_indices

            # # carefully add robot
            robot_handle = self.gym.create_actor(env, robot_asset, robot_pose, "robot", i, 2)
            self.gym.set_actor_scale(env, robot_handle, self.robot.scale[0])
            self._robot_handles.append(robot_handle)
            # set dof properties
            self.gym.set_actor_dof_properties(env, robot_handle, robot_dof_props)

            robot_rigid_body_indices = {}
            for rigid_body_name, rigid_body_idx in self._robot_link_dict.items():
                rigid_body_idx = self.gym.find_actor_rigid_body_index(
                    env, robot_handle, rigid_body_name, gymapi.DOMAIN_SIM
                )
                robot_rigid_body_indices[rigid_body_name] = rigid_body_idx

            self._env_rigid_body_global_indices[-1]["robot"] = robot_rigid_body_indices

        # GET initial state, copy for reset later
        self._initial_state = np.copy(self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL))

        ###### set VEWIER camera ######
        # point camera at middle env
        if not self.headless:  # TODO: update a default viewer
            cam_pos = gymapi.Vec3(1, 1, 1)
            cam_target = gymapi.Vec3(-1, -1, -0.5)
            middle_env = self._envs[self.num_envs // 2 + num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)
        ################################

    def get_states(self, env_ids: list[int] | None = None) -> list[EnvState]:
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        states = []
        self.gym.start_access_image_tensors(self.sim)
        for env_id in env_ids:
            env_state = {"objects": {}, "robots": {}, "cameras": {}, "metasim": {}}
            for obj_id, obj in enumerate(self.objects + [self._robot]):
                obj_state = {}
                object_type = "robots" if obj.name in self._robot_names else "objects"

                ## Basic states
                obj_state["pos"] = self._root_states.view(self.num_envs, -1, 13)[env_id, obj_id, :3].cpu()
                obj_state["rot"] = self._root_states.view(self.num_envs, -1, 13)[env_id, obj_id, 3:7].cpu()
                obj_state["vel"] = self._root_states.view(self.num_envs, -1, 13)[env_id, obj_id, 7:10].cpu()
                obj_state["ang_vel"] = self._root_states.view(self.num_envs, -1, 13)[env_id, obj_id, 10:].cpu()

                ## Joint states
                if isinstance(obj, ArticulationObjCfg):
                    obj_state["dof_pos"] = {
                        joint_name: self._dof_states.view(self.num_envs, -1, 2)[env_id, global_idx, 0].item()
                        for joint_name, global_idx in (self._joint_info[obj.name]["global_indices"]).items()
                    }
                    obj_state["dof_vel"] = {
                        joint_name: self._dof_states.view(self.num_envs, -1, 2)[env_id, global_idx, 1].item()
                        for joint_name, global_idx in (self._joint_info[obj.name]["global_indices"]).items()
                    }

                ## Actuator states
                ## XXX: Could non-robot objects have actuators?
                if isinstance(obj, BaseRobotCfg):
                    obj_state["dof_pos_target"] = {
                        joint_name: None  # TODO
                        for i, joint_name in enumerate(self._joint_info[obj.name]["names"])
                    }
                    obj_state["dof_vel_target"] = {
                        joint_name: None  # TODO
                        for i, joint_name in enumerate(self._joint_info[obj.name]["names"])
                    }
                    obj_state["dof_torque"] = {
                        joint_name: None  # TODO
                        for i, joint_name in enumerate(self._joint_info[obj.name]["names"])
                    }
                env_state[object_type][obj.name] = obj_state

                ## Body states
                ### XXX: will have bug when there are multiple objects with same structure, e.g. dual gripper

                for i, body_name in enumerate(self._body_info[obj.name]["name"]):
                    body_state = {}
                    body_state["pos"] = self._rigid_body_states[
                        self._body_info[obj.name]["global_indices"][body_name], :3
                    ].cpu()
                    body_state["rot"] = self._rigid_body_states[
                        self._body_info[obj.name]["global_indices"][body_name], 3:7
                    ].cpu()
                    body_state["vel"] = self._rigid_body_states[
                        self._body_info[obj.name]["global_indices"][body_name], 7:10
                    ].cpu()
                    body_state["ang_vel"] = self._rigid_body_states[
                        self._body_info[obj.name]["global_indices"][body_name], 10:
                    ].cpu()
                    body_state["com"] = self._rigid_body_states[
                        self._body_info[obj.name]["global_indices"][body_name], :3
                    ].cpu()
                    env_state["metasim"][f"metasim_body_{body_name}"] = body_state

            for i_cam, cam in enumerate(self._cameras):
                cam_state = {}
                cam_state[cam.name] = {
                    "rgb": self._rgb_tensors[env_id][i_cam][..., :3],
                    "depth": self._depth_tensors[env_id][i_cam],
                    "pos": torch.tensor(cam.pos, device=self.device),
                    "look_at": torch.tensor(cam.look_at, device=self.device),
                    "intrinsic": torch.zeros((3, 3), device=self.device),  # TODO: get intrinsic matrix
                    "extrinsic": torch.zeros((4, 4), device=self.device),  # TODO: get extrinsic matrix
                }
            env_state["cameras"] = cam_state
            states.append(env_state)
        self.gym.end_access_image_tensors(self.sim)
        return states

    def get_observation(self, action=None) -> dict:
        states = self.get_states()
        rgbs = [state["cameras"][self.cameras[0].name]["rgb"] for state in states]
        rgb_tensor = torch.stack(rgbs, dim=0)
        depths = [state["cameras"][self.cameras[0].name]["depth"] for state in states]
        depth_tensor = torch.stack(depths, dim=0)

        ## TODO: get the following items from states
        joint_qpos = (
            self._dof_states.reshape(self.num_envs, -1)[:, -len(self._joint_info[self.robot.name]["names"]) :]
            .clone()
            .cpu()
        )
        if self.robot.ee_prim_path is not None:
            robot_ee_state = (
                self._rigid_body_states.reshape(self.num_envs, -1, self._rigid_body_states.shape[1])[
                    :, self._env_rigid_body_global_indices[0]["robot"][self.robot.ee_prim_path], :
                ]
                .clone()
                .cpu()
            )  # object first, robot last
        else:
            robot_ee_state = None

        robot_root_state = (
            self._root_states.reshape(self.num_envs, -1, self._root_states.shape[1])[:, -1, :].clone().cpu()
        )  # object first, robot last

        robot_body_state = (
            self._rigid_body_states.reshape(self.num_envs, -1, self._rigid_body_states.shape[1])[
                :, -len(self._env_rigid_body_global_indices[0]["robot"]) :, :
            ]
            .clone()
            .cpu()
        )
        # action
        if action is not None:
            joint_qpos_target = torch.tensor(action, dtype=torch.float32, device=self.device).reshape(self.num_envs, -1)

        else:
            joint_qpos_target = torch.zeros(self.num_envs, len(self._joint_info[self.robot.name]["names"]))

        joint_qpos_target = joint_qpos_target.reshape(self.num_envs, -1)

        data_dict = {
            "rgb": rgb_tensor.clone() if rgb_tensor is not None else None,
            "depth": depth_tensor.clone() if depth_tensor is not None else None,
            ## Camera
            "cam_pos": [torch.tensor(self.cameras[i].pos) for i in range(len(self.cameras))] * self.num_envs,
            "cam_look_at": [torch.tensor(self.cameras[i].look_at) for i in range(len(self.cameras))] * self.num_envs,
            "cam_intr": [torch.zeros(3, 3) for i in range(len(self.cameras))]
            * self.num_envs,  # TODO: get intrinsic matrix
            "cam_extr": [torch.zeros(4, 4) for i in range(len(self.cameras))]
            * self.num_envs,  # TODO: get extrinsic matrix
            ## State
            "joint_qpos_target": joint_qpos_target,
            "joint_qpos": joint_qpos,  # align with old version
            "robot_ee_state": robot_ee_state,
            "robot_root_state": robot_root_state,
            "robot_body_state": robot_body_state,
        }

        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.cpu()
        torch.cuda.empty_cache()

        self.gym.end_access_image_tensors(self.sim)
        return data_dict

    @property
    def episode_length_buf(self) -> list[int]:
        return self._episode_length_buf

    ############################################################
    ## Gymnasium main methods
    ############################################################
    def _get_action_array_all(self, actions: list[Action]):
        action_array_list = []

        for action_data in actions:
            flat_vals = []
            for joint_i, joint_name in enumerate(self._joint_info[self.robot.name]["names"]):
                flat_vals.append(action_data["dof_pos_target"][joint_name])  # TODO: support other actions

            action_array = torch.tensor(flat_vals, dtype=torch.float32, device=self.device).unsqueeze(0)

            action_array_list.append(action_array)
        action_array_all = torch.cat(action_array_list, dim=0)
        return action_array_all

    def set_dof_targets(self, obj_name: str, actions: list[Action]):
        action_input = torch.zeros_like(self._dof_states[:, 0].clone())
        action_array_all = self._get_action_array_all(actions)
        robot_dim = action_array_all.shape[1]

        assert (
            action_input.shape[0] % self._num_envs == 0
        )  # WARNING: obj dim(env0), robot dim(env0), obj dim(env1), robot dim(env1) ...
        chunk_size = action_input.shape[0] // self._num_envs
        robot_dim_index = [
            i * chunk_size + offset
            for i in range(self.num_envs)
            for offset in range(chunk_size - robot_dim, chunk_size)
        ]
        action_input[robot_dim_index] = action_array_all.float().to(self.device).reshape(-1)

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(action_input))

    def refresh_render(self) -> None:
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Refresh cameras and viewer
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        if not self.headless:
            self.gym.draw_viewer(self.viewer, self.sim, False)

    def simulate(self) -> None:
        # Step the physics
        for _ in range(self.scenario.decimation):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

        # Refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh cameras and viewer
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        if not self.headless:
            self.gym.draw_viewer(self.viewer, self.sim, False)

        self.gym.sync_frame_time(self.sim)

    def set_states(self, states: list[EnvState], env_ids: list[int] | None = None):
        ## TODO: support the case when env_ids != list(range(self.num_envs))
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        assert len(states) == self.num_envs
        pos_list = []
        rot_list = []
        q_list = []
        states_flat = [{**state["objects"], **state["robots"]} for state in states]
        for state in states_flat:
            pos_list_i = []
            rot_list_i = []
            q_list_i = []
            for obj in self.objects:
                obj_name = obj.name
                pos = np.array(state[obj_name].get("pos", [0.0, 0.0, 0.0]))
                rot = np.array(state[obj_name].get("rot", [1.0, 0.0, 0.0, 0.0]))
                obj_quat = [rot[1], rot[2], rot[3], rot[0]]  # IsaacGym convention

                pos_list_i.append(pos)
                rot_list_i.append(obj_quat)
                if isinstance(obj, ArticulationObjCfg):
                    obj_joint_q = np.zeros(len(self._articulated_joint_dict_dict[obj_name]))
                    articulated_joint_dict = self._articulated_joint_dict_dict[obj_name]
                    for joint_name, joint_idx in articulated_joint_dict.items():
                        if "dof_pos" in state[obj_name]:
                            obj_joint_q[joint_idx] = state[obj_name]["dof_pos"][joint_name]
                        else:
                            log.warning(f"No dof_pos for {joint_name} in {obj_name}")
                            obj_joint_q[joint_idx] = 0.0
                    q_list_i.append(obj_joint_q)

            pos_list_i.append(np.array(state[self.robot.name].get("pos", [0.0, 0.0, 0.0])))
            rot = np.array(state[self.robot.name].get("rot", [1.0, 0.0, 0.0, 0.0]))
            robot_quat = [rot[1], rot[2], rot[3], rot[0]]
            rot_list_i.append(robot_quat)

            robot_dof_state_i = np.zeros(len(self._robot_joint_dict))
            if "dof_pos" in state[self.robot.name]:
                for joint_name, joint_idx in self._robot_joint_dict.items():
                    robot_dof_state_i[joint_idx] = state[self.robot.name]["dof_pos"][joint_name]
            else:
                for joint_name, joint_idx in self._robot_joint_dict.items():
                    robot_dof_state_i[joint_idx] = (
                        self.robot.joint_limits[joint_name][0] + self.robot.joint_limits[joint_name][1]
                    ) / 2

            q_list_i.append(robot_dof_state_i)
            pos_list.append(pos_list_i)
            rot_list.append(rot_list_i)
            q_list.append(q_list_i)

        self._set_actor_root_state(pos_list, rot_list)
        self._set_actor_joint_state(q_list)

        # Refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

    def _set_actor_root_state(self, position_list, rotation_list):
        new_root_states = self._root_states.clone()
        new_root_states_pos = torch.tensor(np.array(position_list), dtype=torch.float32, device=self.device).reshape(
            -1, 3
        )
        new_root_states_rot = torch.tensor(np.array(rotation_list), dtype=torch.float32, device=self.device).reshape(
            -1, 4
        )
        new_root_states[:, :3] = new_root_states_pos
        new_root_states[:, 3:7] = new_root_states_rot
        # self.rb_states[:, self.actor_id, :7] = target_pose
        root_reset_actors_indices = torch.unique(
            torch.tensor(
                np.arange(new_root_states.shape[0]),
                dtype=torch.float32,
                device=self.device,
            )
        ).to(dtype=torch.int32)
        res = self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(new_root_states),
            gymtorch.unwrap_tensor(root_reset_actors_indices),
            len(root_reset_actors_indices),
        )
        assert res

        return

    def _set_actor_joint_state(self, joint_pos_list):
        new_dof_states = self._dof_states.clone()

        new_dof_pos = []
        # Process each element in M x N
        for env_i in range(self.num_envs):
            flat_vals = []
            for i in range(len(joint_pos_list[env_i])):
                flat_vals.extend(joint_pos_list[env_i][i])
            flat_array = np.array(flat_vals)
            new_dof_pos.append(flat_array)
        new_dof_pos = torch.tensor(np.array(new_dof_pos), dtype=torch.float32, device=self.device)

        zero_vel = torch.zeros_like(new_dof_pos)
        new_dof_states[:, 0] = new_dof_pos.reshape(-1)
        new_dof_states[:, 1] = zero_vel.reshape(-1)
        res = self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(new_dof_states))
        assert res
        return

    def close(self) -> None:
        pass

    ############################################################
    ## Set object states per step
    ############################################################
    def _set_object_joint_pos(
        self,
        object: BaseObjCfg,
        joint_pos: list[float],
    ) -> None:
        assert len(joint_pos) == self.env.scene.articulations[object.name].num_joints

        pos = torch.tensor(joint_pos, device=self.env.device)
        vel = torch.zeros_like(pos)
        self.env.scene.articulations[object.name].write_joint_state_to_sim(
            pos, vel, env_ids=torch.tensor([0], device=self.env.device)
        )

    ############################################################
    ## Utils
    ############################################################
    def get_object_joint_names(self, object: BaseObjCfg) -> list[str]:
        if isinstance(object, ArticulationObjCfg):
            return self.env.scene.articulations[object.name].joint_names
        else:
            return []

    @property
    def num_envs(self) -> int:
        return self._num_envs


# TODO: try to align handler API and use GymWrapper instead
IsaacgymEnv: type[EnvWrapper[IsaacgymHandler]] = GymEnvWrapper(IsaacgymHandler)
