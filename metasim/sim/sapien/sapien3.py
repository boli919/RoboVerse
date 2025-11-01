"""Implemention of Sapien Handler.

This file contains the implementation of Sapien3Handler, which is a subclass of BaseSimHandler.
Sapien3Handler is used to handle the simulation environment using Sapien.
Currently using Sapien 3.x
"""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import sapien
import sapien.core as sapien_core
import torch
from loguru import logger as log
from packaging.version import parse as parse_version
from sapien.utils import Viewer

from metasim.cfg.objects import (
    ArticulationObjCfg,
    NonConvexRigidObjCfg,
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.cfg.robots import BaseRobotCfg
from metasim.sim import BaseSimHandler, EnvWrapper, GymEnvWrapper
from metasim.types import Action, EnvState
from metasim.utils.math import quat_from_euler_np
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState


class Sapien3Handler(BaseSimHandler):
    """Sapien3 Handler class."""

    def __init__(self, scenario):
        assert parse_version(sapien.__version__) >= parse_version("3.0.0a0"), "Sapien3 is required"
        assert parse_version(sapien.__version__) < parse_version("4.0.0"), "Sapien3 is required"
        log.warning("Sapien3 is still under development, some metasim apis yet don't have sapien3 support")
        super().__init__(scenario)
        self.headless = scenario.headless
        self._actions_cache: list[Action] = []
        self._robot_contacts = []
        # Store contacts between any two objects for the last refresh_render call
        self._all_contacts = []
        
        # Multi-environment support
        self._num_envs = scenario.num_envs
        self.scenes = []  # List of scene instances
        self.loaders = []  # List of URDF loaders per scene
        self.viewers = []  # List of viewers per scene (if not headless)
        
        # Per-env data structures: list of dicts
        self.object_ids_list = []  # list[dict[str, Entity]]
        self.link_ids_list = []  # list[dict[str, list[Component]]]
        self.object_joint_order_list = []  # list[dict[str, list[str]]]
        self.camera_ids_list = []  # list[dict[str, Camera]]
        self._previous_dof_pos_target_list = []  # list[dict[str, ndarray]]
        self._previous_dof_vel_target_list = []  # list[dict[str, ndarray]]
        self._previous_dof_torque_target_list = []  # list[dict[str, ndarray]]
        
        # Calculate scene assignment for each env
        self.env_scene_cfgs = []
        if scenario.scenes is not None and len(scenario.scenes) > 0:
            for env_id in range(self._num_envs):
                scene_idx = env_id % len(scenario.scenes)
                self.env_scene_cfgs.append(scenario.scenes[scene_idx])
                log.info(f"Env {env_id}: assigned scene '{scenario.scenes[scene_idx].name if scenario.scenes[scene_idx].name else f'scene_{scene_idx}'}'")
        else:
            # Use single scene for all envs
            for env_id in range(self._num_envs):
                self.env_scene_cfgs.append(scenario.scene)
                log.info(f"Env {env_id}: using default scene")

    def load_scene_for_env(self, scene, scene_cfg, env_id, env_offset=None):
        """Loads the scene into the specified simulation scene instance with position offset."""
        if scene_cfg is None or not hasattr(scene_cfg, "usd_path") or scene_cfg.usd_path is None:
            log.warning(f"Env {env_id}: Scene config has no usd_path, skipping scene loading.")
            return

        if env_offset is None:
            env_offset = np.array([0.0, 0.0, 0.0])

        builder = scene.create_actor_builder()
        scale = scene_cfg.scale if scene_cfg.scale is not None else (1.0, 1.0, 1.0)
        builder.add_visual_from_file(scene_cfg.usd_path, scale=scale)
        builder.add_nonconvex_collision_from_file(scene_cfg.usd_path, scale=scale)
        
        scene_name = f"{scene_cfg.name}_env{env_id}" if scene_cfg.name else f"scene_env{env_id}"
        static_object = builder.build_static(name=scene_name)
        
        # Apply position offset for grid layout
        scene_pos = np.array(scene_cfg.default_position) + env_offset
        static_object.set_pose(
            sapien_core.Pose(p=scene_pos, q=scene_cfg.quat)
        )
        log.info(f"Env {env_id}: Loaded scene '{scene_name}' at position {scene_pos}")
    
    def _set_camera_look_at_for_scene(self, scene, camera, pos, look_at):
        """Set camera look at for a specific scene."""
        pos = np.array(pos)
        look_at = np.array(look_at)
        forward = look_at - pos
        forward = forward / np.linalg.norm(forward)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        rotation_matrix = np.stack([forward, -right, up], axis=1)
        from scipy.spatial.transform import Rotation as R
        quat = R.from_matrix(rotation_matrix).as_quat()
        quat_sapien = np.array([quat[3], quat[0], quat[1], quat[2]])
        camera.set_pose(sapien_core.Pose(p=pos, q=quat_sapien))

    def _build_sapien(self):
        self.engine = sapien_core.Engine()  # Create a physical simulation engine
        self.renderer = sapien_core.SapienRenderer()  # Create a renderer
        self.engine.set_renderer(self.renderer)

        scene_config = sapien_core.SceneConfig()
        scene_config.gravity = [0, 0, -9.81]
        
        # Calculate env spacing for positioning
        num_per_row = int(math.sqrt(self._num_envs))
        spacing = self.scenario.env_spacing

        # Create ONE scene for all environments (like IsaacGym)
        log.info(f"Creating {self._num_envs} environments in a single scene with spacing {spacing}")
        
        scene = self.engine.create_scene(scene_config)
        scene.set_timestep(self.scenario.sim_params.dt if self.scenario.sim_params.dt is not None else 1 / 100)
        
        # Add ground (shared by all envs)
        ground_material = self.renderer.create_material()
        ground_material.base_color = np.array([202, 164, 114, 256]) / 256
        ground_material.specular = 0.5
        scene.add_ground(altitude=0, render_material=ground_material)
        
        # Add lights (shared by all envs)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
        
        # Create URDF loader (shared)
        loader = scene.create_urdf_loader()
        
        # Store the single scene
        self.scene = scene
        self.scenes = [scene]  # Keep list for compatibility
        self.loader = loader
        self.loaders = [loader]
        
        # Initialize per-env data structures
        for env_id in range(self._num_envs):
            # Calculate position offset for this env (grid layout)
            row = env_id // num_per_row
            col = env_id % num_per_row
            env_offset = np.array([col * spacing * 2, row * spacing * 2, 0.0])
            
            # Load scene model for this env with offset
            self.load_scene_for_env(scene, self.env_scene_cfgs[env_id], env_id, env_offset)
            
            # Initialize per-env data structures
            object_ids: dict[str, sapien_core.Entity] = {}
            link_ids: dict[str, list[sapien.physx.PhysxArticulationLinkComponent]] = {}
            previous_dof_pos_target: dict[str, np.ndarray] = {}
            previous_dof_vel_target: dict[str, np.ndarray] = {}
            previous_dof_torque_target: dict[str, np.ndarray] = {}
            object_joint_order = {}
            camera_ids = {}
            
            # Add cameras for this env
            for camera in self.cameras:
                camera_pos_offset = np.array(camera.pos) + env_offset
                camera_look_at_offset = np.array(camera.look_at) + env_offset
                
                camera_id = scene.add_camera(
                    name=f"{camera.name}_env{env_id}",
                    width=camera.width,
                    height=camera.height,
                    fovy=np.deg2rad(camera.vertical_fov),
                    near=camera.clipping_range[0],
                    far=camera.clipping_range[1],
                )
                camera_ids[camera.name] = camera_id
                self._set_camera_look_at_for_scene(scene, camera_id, camera_pos_offset, camera_look_at_offset)
            
            # Store per-env data
            self.object_ids_list.append(object_ids)
            self.link_ids_list.append(link_ids)
            self.object_joint_order_list.append(object_joint_order)
            self.camera_ids_list.append(camera_ids)
            self._previous_dof_pos_target_list.append(previous_dof_pos_target)
            self._previous_dof_vel_target_list.append(previous_dof_vel_target)
            self._previous_dof_torque_target_list.append(previous_dof_torque_target)
        
        # Load objects and robots for each environment
        for env_id in range(self._num_envs):
            self._load_objects_for_env(env_id)
        
        # For backward compatibility, keep references to first env
        self.object_ids = self.object_ids_list[0]
        self.link_ids = self.link_ids_list[0]
        self._previous_dof_pos_target = self._previous_dof_pos_target_list[0]
        self._previous_dof_vel_target = self._previous_dof_vel_target_list[0]
        self._previous_dof_torque_target = self._previous_dof_torque_target_list[0]
        self.object_joint_order = self.object_joint_order_list[0]
        self.camera_ids = self.camera_ids_list[0]
        
        # Setup viewer - ONE viewer for all envs (like IsaacGym)
        if not self.headless:
            viewer = Viewer(self.renderer)
            viewer.set_scene(self.scene)
            
            # Calculate camera position to see all envs
            # Position camera to view the center of the grid
            grid_center_x = (num_per_row - 1) * spacing
            grid_center_y = (num_per_row - 1) * spacing
            camera_distance = max(3.0, num_per_row * spacing * 1.5)
            
            camera_pos = np.array([grid_center_x + camera_distance, grid_center_y - camera_distance, camera_distance])
            camera_target = np.array([grid_center_x, grid_center_y, 0.0])
            direction_vector = camera_target - camera_pos
            yaw = math.atan2(direction_vector[1], direction_vector[0])
            pitch = math.atan2(direction_vector[2], math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2))
            roll = 0
            viewer.set_camera_xyz(x=camera_pos[0], y=camera_pos[1], z=camera_pos[2])
            viewer.set_camera_rpy(r=roll, p=pitch, y=-yaw)
            self.viewer = viewer
            self.viewers = [viewer]
            log.info(f"Viewer positioned at {camera_pos}, looking at {camera_target}")
        
        self.debug_points = []
        self.debug_lines = []
        
        # Update render for the single scene
        self.scene.update_render()
        
        # Take pictures from all cameras
        for env_id in range(self._num_envs):
            for camera_name, camera_id in self.camera_ids_list[env_id].items():
                camera_id.take_picture()
    
    def _load_objects_for_env(self, env_id):
        """Load objects and robots for a specific environment in the shared scene."""
        scene = self.scene
        loader = self.loader
        object_ids = self.object_ids_list[env_id]
        link_ids = self.link_ids_list[env_id]
        object_joint_order = self.object_joint_order_list[env_id]
        previous_dof_pos_target = self._previous_dof_pos_target_list[env_id]
        previous_dof_vel_target = self._previous_dof_vel_target_list[env_id]
        previous_dof_torque_target = self._previous_dof_torque_target_list[env_id]
        
        # Calculate position offset for this env (grid layout)
        num_per_row = int(math.sqrt(self._num_envs))
        spacing = self.scenario.env_spacing
        row = env_id // num_per_row
        col = env_id % num_per_row
        env_offset = np.array([col * spacing * 2, row * spacing * 2, 0.0])
        
        for object in [*self.objects, self.robot]:
            if isinstance(object, (ArticulationObjCfg, BaseRobotCfg)):
                is_mesh_load = hasattr(object, 'mesh_path') and object.mesh_path and object.mesh_path.endswith('.obj')
                is_urdf_load = hasattr(object, 'urdf_path') and object.urdf_path

                if is_urdf_load:
                    loader.fix_root_link = object.fix_base_link
                    loader.scale = object.scale[0]
                    file_path = object.urdf_path
                    curr_id = loader.load(file_path)
                    obj_pos = np.array(object.default_position) + env_offset
                    pose = sapien_core.Pose(p=obj_pos, q=object.default_orientation)
                    curr_id.set_root_pose(pose)
                    log.debug(f"[DEBUG] Env {env_id}: Loaded URDF '{object.name}' at pose: {pose}")

                    object_ids[object.name] = curr_id

                    active_joints = curr_id.get_active_joints()
                    cur_joint_names = [j.get_name() for j in active_joints]
                    object_joint_order[object.name] = cur_joint_names

                    if isinstance(object, BaseRobotCfg):
                        for joint in active_joints:
                            stiffness = object.actuators[joint.get_name()].stiffness
                            damping = object.actuators[joint.get_name()].damping
                            joint.set_drive_property(stiffness, damping)
                    else:
                        for joint in active_joints:
                            joint.set_drive_property(0, 0)
                
                elif is_mesh_load:
                    log.info(f"Env {env_id}: Attempting to load .obj as rigid body: {object.mesh_path}")
                    builder = scene.create_actor_builder()
                    density = getattr(object, 'density', 1000.0)
                    
                    builder.add_convex_collision_from_file(filename=object.mesh_path, scale=object.scale, density=density)
                    builder.add_visual_from_file(filename=object.mesh_path, scale=object.scale)
                    
                    actor = builder.build(name=f"{object.name}_env{env_id}") if not object.fix_base_link else builder.build_static(name=f"{object.name}_env{env_id}")
                    obj_pos = np.array(object.default_position) + env_offset
                    pose = sapien_core.Pose(p=obj_pos, q=object.default_orientation)
                    actor.set_pose(pose)
                    log.debug(f"[DEBUG] Env {env_id}: Loaded OBJ '{object.name}' at pose: {pose}")

                    object_ids[object.name] = actor
                    object_joint_order[object.name] = []

                else:
                    if not isinstance(object, BaseRobotCfg) or (isinstance(object, BaseRobotCfg) and object.urdf_path is None):
                        log.warning(f"Env {env_id}: Object '{object.name}' has no valid urdf_path or mesh_path. Skipped.")

            elif isinstance(object, PrimitiveCubeCfg):
                actor_builder = scene.create_actor_builder()
                actor_builder.add_box_collision(half_size=object.half_size, density=object.density)
                actor_builder.add_box_visual(
                    half_size=object.half_size,
                    material=sapien_core.render.RenderMaterial(
                        base_color=object.color[:3] + [1] if object.color else [1.0, 1.0, 0.0, 1.0]
                    ),
                )
                box = actor_builder.build(name=f"{object.name}_env{env_id}")
                obj_pos = np.array(object.default_position) + env_offset
                pose = sapien_core.Pose(p=obj_pos, q=object.default_orientation)
                box.set_pose(pose)
                log.debug(f"[DEBUG] Env {env_id}: Loaded PrimitiveCube '{object.name}' at pose: {pose}")
                object_ids[object.name] = box
                object_joint_order[object.name] = []

            elif isinstance(object, PrimitiveSphereCfg):
                actor_builder = scene.create_actor_builder()
                actor_builder.add_sphere_collision(radius=object.radius, density=object.density)
                actor_builder.add_sphere_visual(
                    radius=object.radius,
                    material=sapien_core.render.RenderMaterial(
                        base_color=object.color[:3] + [1] if object.color else [1.0, 1.0, 0.0, 1.0]
                    ),
                )
                sphere = actor_builder.build(name=f"{object.name}_env{env_id}")
                obj_pos = np.array(object.default_position) + env_offset
                pose = sapien_core.Pose(p=obj_pos, q=object.default_orientation)
                sphere.set_pose(pose)
                log.debug(f"[DEBUG] Env {env_id}: Loaded PrimitiveSphere '{object.name}' at pose: {pose}")
                object_ids[object.name] = sphere
                object_joint_order[object.name] = []

            elif isinstance(object, NonConvexRigidObjCfg):
                builder = scene.create_actor_builder()
                obj_pos = np.array(object.mesh_pose[:3]) + env_offset
                scene_pose = sapien_core.Pose(p=obj_pos, q=np.array(object.mesh_pose[3:]))
                builder.add_nonconvex_collision_from_file(object.usd_path, scene_pose)
                builder.add_visual_from_file(object.usd_path, scene_pose)
                curr_id = builder.build_static(name=f"{object.name}_env{env_id}")
                log.debug(f"[DEBUG] Env {env_id}: Loaded NonConvexRigidObj '{object.name}' at pose: {scene_pose}")
                object_ids[object.name] = curr_id
                object_joint_order[object.name] = []

            elif isinstance(object, RigidObjCfg):
                is_mesh_load = hasattr(object, 'mesh_path') and object.mesh_path and object.mesh_path.endswith('.obj')
                is_urdf_load = hasattr(object, 'urdf_path') and object.urdf_path

                if is_urdf_load:
                    loader.fix_root_link = object.fix_base_link
                    loader.scale = object.scale[0]
                    file_path = object.urdf_path
                    curr_id: sapien_core.Entity
                    try:
                        curr_id = loader.load(file_path)
                    except Exception as e:
                        log.warning(f"Error loading {file_path}: {e}")
                        curr_id_list = loader.load_multiple(file_path)
                        for id_item in curr_id_list:
                            if len(id_item):
                                curr_id = id_item
                                break
                    if isinstance(curr_id, list):
                        curr_id = curr_id[0]
                    obj_pos = np.array(object.default_position) + env_offset
                    pose = sapien_core.Pose(p=obj_pos, q=object.default_orientation)
                    curr_id.set_pose(pose)
                    log.debug(f"[DEBUG] Env {env_id}: Loaded RigidObj '{object.name}' at pose: {pose}")
                    object_ids[object.name] = curr_id
                    object_joint_order[object.name] = []

                elif is_mesh_load:
                    log.info(f"Env {env_id}: Attempting to load .obj as rigid body: {object.mesh_path}")
                    builder = scene.create_actor_builder()
                    density = getattr(object, 'density', 1000.0)
                    
                    builder.add_convex_collision_from_file(filename=object.mesh_path, scale=object.scale, density=density)
                    builder.add_visual_from_file(filename=object.mesh_path, scale=object.scale)
                    
                    actor = builder.build(name=f"{object.name}_env{env_id}") if not object.fix_base_link else builder.build_static(name=f"{object.name}_env{env_id}")
                    obj_pos = np.array(object.default_position) + env_offset
                    pose = sapien_core.Pose(p=obj_pos, q=object.default_orientation)
                    actor.set_pose(pose)
                    log.debug(f"[DEBUG] Env {env_id}: Loaded OBJ '{object.name}' at pose: {pose}")

                    object_ids[object.name] = actor
                    object_joint_order[object.name] = []

                else:
                    log.warning(f"Env {env_id}: Object '{object.name}' has no valid urdf_path or mesh_path. Skipped.")

            if object.name in object_ids:
                loaded_entity = object_ids[object.name]
                if hasattr(loaded_entity, 'get_links'):
                    link_ids[object.name] = loaded_entity.get_links()
                else:
                    link_ids[object.name] = [comp for comp in loaded_entity.get_components() if isinstance(comp, sapien_core.physx.PhysxRigidBaseComponent)]

                if isinstance(object, (ArticulationObjCfg, BaseRobotCfg)) and len(object_joint_order.get(object.name, [])) > 0:
                    previous_dof_pos_target[object.name] = np.zeros((len(object_joint_order[object.name]),), dtype=np.float32)
                    previous_dof_vel_target[object.name] = np.zeros((len(object_joint_order[object.name]),), dtype=np.float32)
                    previous_dof_torque_target[object.name] = np.zeros((len(object_joint_order[object.name]),), dtype=np.float32)
                else:
                    if object.name not in link_ids:
                         link_ids[object.name] = []

    def _apply_action(self, instance: sapien_core.physx.PhysxArticulation, pos_action=None, vel_action=None):
        qf = instance.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        instance.set_qf(qf)
        if pos_action is not None:
            for joint in instance.get_active_joints():
                joint.set_drive_target(pos_action[joint.get_name()])
        if vel_action is not None:
            for joint in instance.get_active_joints():
                joint.set_drive_velocity_target(vel_action[joint.get_name()])

    def set_dof_targets(self, obj_name, target: list[Action]):
        # For multi-env, target is a list of actions, one per env
        for env_id in range(self._num_envs):
            object_ids = self.object_ids_list[env_id]
            object_joint_order = self.object_joint_order_list[env_id]
            previous_dof_pos_target = self._previous_dof_pos_target_list[env_id]
            previous_dof_vel_target = self._previous_dof_vel_target_list[env_id]
            
            if obj_name not in object_ids:
                continue
                
            instance = object_ids[obj_name]
            if isinstance(instance, sapien_core.physx.PhysxArticulation):
                action = target[env_id] if env_id < len(target) else target[0]
                pos_target = action.get("dof_pos_target", None)
                vel_target = action.get("dof_vel_target", None)
                pos_target_arr = (
                    np.array([pos_target[name] for name in object_joint_order[obj_name]]) if pos_target else None
                )
                vel_target_arr = (
                    np.array([vel_target[name] for name in object_joint_order[obj_name]]) if vel_target else None
                )
                previous_dof_pos_target[obj_name] = pos_target_arr
                previous_dof_vel_target[obj_name] = vel_target_arr
                self._apply_action(instance, pos_target, vel_target)

    def _simulate(self):
        # Simulate the single scene containing all environments
        for i in range(self.scenario.decimation):
            self.scene.step()
        
        self.scene.update_render()
        
        if not self.headless and self.viewer:
            self.viewer.render()
        
        # Take pictures from all cameras in all envs
        for env_id in range(self._num_envs):
            camera_ids = self.camera_ids_list[env_id]
            for camera_name, camera_id in camera_ids.items():
                camera_id.take_picture()

    def launch(self) -> None:
        self._build_sapien()

    def close(self):
        if not self.headless and self.viewer:
            self.viewer.close()
        self.scene = None

    def _get_link_states_for_env(self, obj_name: str, link_ids_dict) -> tuple[list, torch.Tensor]:
        """Get link states for a specific object in a specific environment."""
        link_name_list = []
        link_state_list = []

        if len(link_ids_dict.get(obj_name, [])) == 0:
            return [], torch.zeros((0, 13), dtype=torch.float32)

        for link in link_ids_dict[obj_name]:
            pose = link.get_pose()
            pos = torch.tensor(pose.p, dtype=torch.float32)
            rot = torch.tensor(pose.q, dtype=torch.float32)
            if isinstance(link, sapien.physx.PhysxRigidStaticComponent):
                vel = torch.zeros(3, dtype=torch.float32)
                ang_vel = torch.zeros(3, dtype=torch.float32)
            else:
                vel = torch.tensor(link.get_linear_velocity(), dtype=torch.float32)
                ang_vel = torch.tensor(link.get_angular_velocity(), dtype=torch.float32)
            link_state = torch.cat([pos, rot, vel, ang_vel], dim=-1).unsqueeze(0)
            link_name_list.append(link.get_name())
            link_state_list.append(link_state)
        link_state_tensor = torch.cat(link_state_list, dim=0)
        return link_name_list, link_state_tensor
    
    def _get_link_states(self, obj_name: str) -> tuple[list, torch.Tensor]:
        """Backward compatibility wrapper."""
        return self._get_link_states_for_env(obj_name, self.link_ids)

    def _get_states(self, env_ids=None) -> TensorState:
        """Get states from all environments and batch them together."""
        if env_ids is None:
            env_ids = list(range(self._num_envs))
        
        # Collect states from each environment
        all_env_states = []
        for env_id in env_ids:
            env_state = self._get_single_env_state(env_id)
            all_env_states.append(env_state)
        
        # Batch states together
        object_states = {}
        robot_states = {}
        camera_states = {}
        
        # Batch object states
        if len(self.objects) > 0 and all_env_states:
            for obj in self.objects:
                obj_name = obj.name
                root_states = torch.cat([s['objects'][obj_name].root_state for s in all_env_states], dim=0)
                body_names = all_env_states[0]['objects'][obj_name].body_names
                body_states = torch.cat([s['objects'][obj_name].body_state for s in all_env_states], dim=0)
                
                if all_env_states[0]['objects'][obj_name].joint_pos is not None:
                    joint_pos = torch.cat([s['objects'][obj_name].joint_pos for s in all_env_states], dim=0)
                    joint_vel = torch.cat([s['objects'][obj_name].joint_vel for s in all_env_states], dim=0)
                    object_states[obj_name] = ObjectState(
                        root_state=root_states,
                        body_names=body_names,
                        body_state=body_states,
                        joint_pos=joint_pos,
                        joint_vel=joint_vel
                    )
                else:
                    object_states[obj_name] = ObjectState(
                        root_state=root_states,
                        body_names=body_names,
                        body_state=body_states
                    )
        
        # Batch robot states
        if self.robot and all_env_states:
            robot_name = self.robot.name
            root_states = torch.cat([s['robots'][robot_name].root_state for s in all_env_states], dim=0)
            body_names = all_env_states[0]['robots'][robot_name].body_names
            body_states = torch.cat([s['robots'][robot_name].body_state for s in all_env_states], dim=0)
            joint_pos = torch.cat([s['robots'][robot_name].joint_pos for s in all_env_states], dim=0)
            joint_vel = torch.cat([s['robots'][robot_name].joint_vel for s in all_env_states], dim=0)
            
            joint_pos_target = None
            if all_env_states[0]['robots'][robot_name].joint_pos_target is not None:
                joint_pos_target = torch.cat([s['robots'][robot_name].joint_pos_target for s in all_env_states], dim=0)
            
            joint_vel_target = None
            if all_env_states[0]['robots'][robot_name].joint_vel_target is not None:
                joint_vel_target = torch.cat([s['robots'][robot_name].joint_vel_target for s in all_env_states], dim=0)
            
            joint_effort_target = None
            if all_env_states[0]['robots'][robot_name].joint_effort_target is not None:
                joint_effort_target = torch.cat([s['robots'][robot_name].joint_effort_target for s in all_env_states], dim=0)
            
            robot_states[robot_name] = RobotState(
                root_state=root_states,
                body_names=body_names,
                body_state=body_states,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                joint_pos_target=joint_pos_target,
                joint_vel_target=joint_vel_target,
                joint_effort_target=joint_effort_target
            )
        
        # Batch camera states
        if len(self.cameras) > 0 and all_env_states:
            for camera in self.cameras:
                camera_name = camera.name
                rgbs = torch.stack([s['cameras'][camera_name].rgb.squeeze(0) for s in all_env_states], dim=0)
                depths = torch.stack([s['cameras'][camera_name].depth.squeeze(0) for s in all_env_states], dim=0)
                camera_states[camera_name] = CameraState(rgb=rgbs, depth=depths)
        
        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, sensors={})
    
    def _get_single_env_state(self, env_id: int) -> dict:
        """Get state for a single environment."""
        object_ids = self.object_ids_list[env_id]
        link_ids = self.link_ids_list[env_id]
        object_joint_order = self.object_joint_order_list[env_id]
        previous_dof_pos_target = self._previous_dof_pos_target_list[env_id]
        previous_dof_vel_target = self._previous_dof_vel_target_list[env_id]
        previous_dof_torque_target = self._previous_dof_torque_target_list[env_id]
        camera_ids = self.camera_ids_list[env_id]
        
        object_states = {}
        for obj in self.objects:
            if obj.name not in object_ids:
                continue
            obj_inst = object_ids[obj.name]
            pose = obj_inst.get_pose()
            link_names, link_state = self._get_link_states_for_env(obj.name, link_ids)
            
            if isinstance(obj_inst, sapien_core.physx.PhysxArticulation):
                pos = torch.tensor(pose.p, dtype=torch.float32)
                rot = torch.tensor(pose.q, dtype=torch.float32)
                vel = torch.tensor(obj_inst.get_root_linear_velocity(), dtype=torch.float32)
                ang_vel = torch.tensor(obj_inst.get_root_angular_velocity(), dtype=torch.float32)
                root_state = torch.cat([pos, rot, vel, ang_vel], dim=-1).unsqueeze(0)
                joint_reindex = self.get_joint_reindex(obj.name, object_joint_order)
                state = ObjectState(
                    root_state=root_state,
                    body_names=link_names,
                    body_state=link_state.unsqueeze(0),
                    joint_pos=torch.tensor(obj_inst.get_qpos()[joint_reindex], dtype=torch.float32).unsqueeze(0),
                    joint_vel=torch.tensor(obj_inst.get_qvel()[joint_reindex], dtype=torch.float32).unsqueeze(0),
                )
            else:
                rigid_component = None
                for comp in obj_inst.get_components():
                    if isinstance(comp, sapien_core.physx.PhysxRigidBaseComponent):
                        rigid_component = comp
                        break
                
                if rigid_component:
                    pos = torch.tensor(pose.p, dtype=torch.float32)
                    rot = torch.tensor(pose.q, dtype=torch.float32)
                    if isinstance(rigid_component, sapien.physx.PhysxRigidStaticComponent):
                        vel = torch.zeros(3, dtype=torch.float32)
                        ang_vel = torch.zeros(3, dtype=torch.float32)
                    else:
                        vel = torch.tensor(rigid_component.get_linear_velocity(), dtype=torch.float32)
                        ang_vel = torch.tensor(rigid_component.get_angular_velocity(), dtype=torch.float32)
                else:
                    pos = torch.tensor(pose.p, dtype=torch.float32)
                    rot = torch.tensor(pose.q, dtype=torch.float32)
                    vel = torch.zeros(3, dtype=torch.float32)
                    ang_vel = torch.zeros(3, dtype=torch.float32)

                root_state = torch.cat([pos, rot, vel, ang_vel], dim=-1).unsqueeze(0)
                state = ObjectState(root_state=root_state, body_names=link_names, body_state=link_state.unsqueeze(0))
            object_states[obj.name] = state

        robot_states = {}
        if self.robot:
            robot = self.robot
            if robot.name in object_ids:
                robot_inst = object_ids[robot.name]
                assert isinstance(robot_inst, sapien_core.physx.PhysxArticulation)
                pose = robot_inst.get_pose()
                pos = torch.tensor(pose.p, dtype=torch.float32)
                rot = torch.tensor(pose.q, dtype=torch.float32)
                vel = torch.tensor(robot_inst.get_root_linear_velocity(), dtype=torch.float32)
                ang_vel = torch.tensor(robot_inst.get_root_angular_velocity(), dtype=torch.float32)
                root_state = torch.cat([pos, rot, vel, ang_vel], dim=-1).unsqueeze(0)
                joint_reindex = self.get_joint_reindex(robot.name, object_joint_order)
                link_names, link_state = self._get_link_states_for_env(robot.name, link_ids)
                pos_target = (
                    torch.tensor(previous_dof_pos_target[robot.name], dtype=torch.float32).unsqueeze(0)
                    if previous_dof_pos_target.get(robot.name) is not None
                    else None
                )
                vel_target = (
                    torch.tensor(previous_dof_vel_target[robot.name], dtype=torch.float32).unsqueeze(0)
                    if previous_dof_vel_target.get(robot.name) is not None
                    else None
                )
                effort_target = (
                    torch.tensor(previous_dof_torque_target[robot.name], dtype=torch.float32).unsqueeze(0)
                    if previous_dof_torque_target.get(robot.name) is not None
                    else None
                )
                state = RobotState(
                    root_state=root_state,
                    body_names=link_names,
                    body_state=link_state.unsqueeze(0),
                    joint_pos=torch.tensor(robot_inst.get_qpos()[joint_reindex], dtype=torch.float32).unsqueeze(0),
                    joint_vel=torch.tensor(robot_inst.get_qvel()[joint_reindex], dtype=torch.float32).unsqueeze(0),
                    joint_pos_target=pos_target,
                    joint_vel_target=vel_target,
                    joint_effort_target=effort_target,
                )
                robot_states[robot.name] = state

        camera_states = {}
        for camera in self.cameras:
            if camera.name in camera_ids:
                cam_inst = camera_ids[camera.name]
                rgb = cam_inst.get_picture("Color")[..., :3]
                rgb = (rgb * 255).clip(0, 255).astype("uint8")
                rgb = torch.from_numpy(rgb.copy())
                depth = -cam_inst.get_picture("Position")[..., 2]
                depth = torch.from_numpy(depth.copy()).to(dtype=torch.float32)
                state = CameraState(rgb=rgb.unsqueeze(0), depth=depth.unsqueeze(0))
                camera_states[camera.name] = state

        return {'objects': object_states, 'robots': robot_states, 'cameras': camera_states}

    def get_joint_reindex(self, obj_name: str, object_joint_order_dict=None) -> list[int]:
        """Get joint reindexing for a given object."""
        if object_joint_order_dict is None:
            object_joint_order_dict = self.object_joint_order
        if obj_name not in object_joint_order_dict:
            return []
        joint_names = object_joint_order_dict[obj_name]
        return list(range(len(joint_names)))  # Simple identity reindex for now
    
    def refresh_render(self):
        log.debug("refresh_render called!")  # Confirm call
        
        # Update render for the single scene
        self.scene.update_render()
        
        if not self.headless and self.viewer:
            self.viewer.render()
        
        # Take pictures from all cameras
        for env_id in range(self._num_envs):
            camera_ids = self.camera_ids_list[env_id]
            for camera_name, camera_id in camera_ids.items():
                camera_id.take_picture()

        # Add collision detection (log **all** contacts in the scene)
        self._robot_contacts = []
        self._all_contacts = []
        contacts = self.scene.get_contacts()
        log.debug(f"Found {len(contacts)} contacts in the scene")
        for contact in contacts:
            actor0 = contact.actor0.name if contact.actor0 else 'Unknown'
            actor1 = contact.actor1.name if contact.actor1 else 'Unknown'
            log.info(f"[Collision][Sapien3] {actor0} ↔ {actor1}")
            self._all_contacts.append((actor0, actor1, contact))

            if self.robot and self.robot.name in actor0:
                other = actor1 if self.robot.name in actor0 else actor0
                self._robot_contacts.append((self.robot.name, other, contact))

    def _set_states(self, states, env_ids=None):
        """Set states for all environments."""
        if env_ids is None:
            env_ids = list(range(self._num_envs))
        
        if isinstance(states, list):
            # states is a list of states, one per env
            for i, env_id in enumerate(env_ids):
                if i < len(states):
                    self._set_single_env_state(env_id, states[i])
        else:
            # Single state dict, apply to all envs
            for env_id in env_ids:
                self._set_single_env_state(env_id, states)
    
    def _set_single_env_state(self, env_id: int, state_dict):
        """Set state for a single environment."""
        object_ids = self.object_ids_list[env_id]
        object_joint_order = self.object_joint_order_list[env_id]
        
        states_flat = state_dict["objects"] | state_dict["robots"]
        for name, val in states_flat.items():
            if name not in object_ids:
                continue
            obj_id = object_ids[name]

            if isinstance(obj_id, sapien_core.physx.PhysxArticulation):
                joint_names = object_joint_order[name]
                qpos_list = [val["dof_pos"][joint_name] for i, joint_name in enumerate(joint_names)]
                obj_id.set_qpos(np.array(qpos_list))

            obj_id.set_pose(sapien_core.Pose(p=val["pos"], q=val["rot"]))

    @property
    def actions_cache(self) -> list[Action]:
        return self._actions_cache

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if obj_name in self.object_joint_order:
            joint_names = deepcopy(self.object_joint_order[obj_name])
            if sort:
                joint_names.sort()
            return joint_names
        else:
            return []

    def get_body_names(self, obj_name, sort=True):
        if obj_name in self.link_ids:
            body_names = deepcopy([link.get_name() for link in self.link_ids[obj_name]])
            if sort:
                return sorted(body_names)
            else:
                return deepcopy(body_names)
        return []

    def set_camera_look_at(self, camera_name: str, pos: tuple[float, float, float], look_at: tuple[float, float, float]):
        camera = self.camera_ids[camera_name]
        pos = np.array(pos)
        look_at = np.array(look_at)
        forward = look_at - pos
        forward = forward / np.linalg.norm(forward)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        rotation_matrix = np.stack([forward, -right, up], axis=1)
        from scipy.spatial.transform import Rotation as R
        quat = R.from_matrix(rotation_matrix).as_quat()
        quat_sapien = np.array([quat[3], quat[0], quat[1], quat[2]])
        camera.set_pose(sapien_core.Pose(p=pos, q=quat_sapien))

    @property
    def robot_contacts(self):
        return self._robot_contacts

    @property
    def all_contacts(self):
        """Return all contacts detected in the last ``refresh_render`` call."""
        return self._all_contacts


Sapien3Env: type[EnvWrapper[Sapien3Handler]] = GymEnvWrapper(Sapien3Handler)