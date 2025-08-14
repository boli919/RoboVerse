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

    def load_scene(self):
        """Loads the scene into the simulation."""
        if not hasattr(self.scenario.scene, "usd_path") or self.scenario.scene.usd_path is None:
            log.warning("Scene config has no usd_path, skipping scene loading.")
            return

        builder = self.scene.create_actor_builder()
        scale = self.scenario.scene.scale if self.scenario.scene.scale is not None else (1.0, 1.0, 1.0)
        builder.add_visual_from_file(self.scenario.scene.usd_path, scale=scale)
        builder.add_nonconvex_collision_from_file(self.scenario.scene.usd_path, scale=scale)
        static_object = builder.build_static(name=self.scenario.scene.name)
        static_object.set_pose(
            sapien_core.Pose(p=self.scenario.scene.default_position, q=self.scenario.scene.quat)
        )

    def _build_sapien(self):
        self.engine = sapien_core.Engine()  # Create a physical simulation engine
        self.renderer = sapien_core.SapienRenderer()  # Create a renderer

        scene_config = sapien_core.SceneConfig()
        scene_config.gravity = [0, 0, -9.81]

        self.engine.set_renderer(self.renderer)
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(self.scenario.sim_params.dt if self.scenario.sim_params.dt is not None else 1 / 100)
        ground_material = self.renderer.create_material()
        ground_material.base_color = np.array([202, 164, 114, 256]) / 256
        ground_material.specular = 0.5
        self.scene.add_ground(altitude=0, render_material=ground_material)

        self.load_scene()

        self.loader = self.scene.create_urdf_loader()

        # Add lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

        # Add agents
        self.object_ids: dict[str, sapien_core.Entity] = {}
        self.link_ids: dict[str, list[sapien.physx.PhysxArticulationLinkComponent]] = {}
        self._previous_dof_pos_target: dict[str, np.ndarray] = {}
        self._previous_dof_vel_target: dict[str, np.ndarray] = {}
        self._previous_dof_torque_target: dict[str, np.ndarray] = {}
        self.object_joint_order = {}
        self.camera_ids = {}

        for camera in self.cameras:
            camera_id = self.scene.add_camera(
                name=camera.name,
                width=camera.width,
                height=camera.height,
                fovy=np.deg2rad(camera.vertical_fov),
                near=camera.clipping_range[0],
                far=camera.clipping_range[1],
            )
            self.camera_ids[camera.name] = camera_id
            self.set_camera_look_at(camera.name, camera.pos, camera.look_at)

        for object in [*self.objects, self.robot]:
            if isinstance(object, (ArticulationObjCfg, BaseRobotCfg)):
                is_mesh_load = hasattr(object, 'mesh_path') and object.mesh_path and object.mesh_path.endswith('.obj')
                is_urdf_load = hasattr(object, 'urdf_path') and object.urdf_path

                if is_urdf_load:
                    self.loader.fix_root_link = object.fix_base_link
                    self.loader.scale = object.scale[0]
                    file_path = object.urdf_path
                    curr_id = self.loader.load(file_path)
                    pose = sapien_core.Pose(p=object.default_position, q=object.default_orientation)
                    curr_id.set_root_pose(pose)
                    log.debug(f"[DEBUG] Loaded URDF '{object.name}' at pose: {pose}")

                    self.object_ids[object.name] = curr_id

                    active_joints = curr_id.get_active_joints()
                    cur_joint_names = [j.get_name() for j in active_joints]
                    self.object_joint_order[object.name] = cur_joint_names

                    if isinstance(object, BaseRobotCfg):
                        for joint in active_joints:
                            stiffness = object.actuators[joint.get_name()].stiffness
                            damping = object.actuators[joint.get_name()].damping
                            joint.set_drive_property(stiffness, damping)
                    else:
                        for joint in active_joints:
                            joint.set_drive_property(0, 0)
                
                elif is_mesh_load:
                    log.info(f"Attempting to load .obj as rigid body: {object.mesh_path}")
                    builder = self.scene.create_actor_builder()
                    density = getattr(object, 'density', 1000.0)
                    
                    builder.add_convex_collision_from_file(filename=object.mesh_path, scale=object.scale, density=density)
                    builder.add_visual_from_file(filename=object.mesh_path, scale=object.scale)
                    
                    actor = builder.build(name=object.name) if not object.fix_base_link else builder.build_static(name=object.name)
                    pose = sapien_core.Pose(p=object.default_position, q=object.default_orientation)
                    actor.set_pose(pose)
                    log.debug(f"[DEBUG] Loaded OBJ '{object.name}' at pose: {pose}")

                    self.object_ids[object.name] = actor
                    self.object_joint_order[object.name] = []

                else:
                    if not isinstance(object, BaseRobotCfg) or (isinstance(object, BaseRobotCfg) and object.urdf_path is None):
                        log.warning(f"Object '{object.name}' has no valid urdf_path or mesh_path. Skipped.")

            elif isinstance(object, PrimitiveCubeCfg):
                actor_builder = self.scene.create_actor_builder()
                actor_builder.add_box_collision(half_size=object.half_size, density=object.density)
                actor_builder.add_box_visual(
                    half_size=object.half_size,
                    material=sapien_core.render.RenderMaterial(
                        base_color=object.color[:3] + [1] if object.color else [1.0, 1.0, 0.0, 1.0]
                    ),
                )
                box = actor_builder.build(name=object.name)
                pose = sapien_core.Pose(p=object.default_position, q=object.default_orientation)
                box.set_pose(pose)
                log.debug(f"[DEBUG] Loaded PrimitiveCube '{object.name}' at pose: {pose}")
                self.object_ids[object.name] = box
                self.object_joint_order[object.name] = []

            elif isinstance(object, PrimitiveSphereCfg):
                actor_builder = self.scene.create_actor_builder()
                actor_builder.add_sphere_collision(radius=object.radius, density=object.density)
                actor_builder.add_sphere_visual(
                    radius=object.radius,
                    material=sapien_core.render.RenderMaterial(
                        base_color=object.color[:3] + [1] if object.color else [1.0, 1.0, 0.0, 1.0]
                    ),
                )
                sphere = actor_builder.build(name=object.name)
                pose = sapien_core.Pose(p=object.default_position, q=object.default_orientation)
                sphere.set_pose(pose)
                log.debug(f"[DEBUG] Loaded PrimitiveSphere '{object.name}' at pose: {pose}")
                self.object_ids[object.name] = sphere
                self.object_joint_order[object.name] = []

            elif isinstance(object, NonConvexRigidObjCfg):
                builder = self.scene.create_actor_builder()
                scene_pose = sapien_core.Pose(p=np.array(object.mesh_pose[:3]), q=np.array(object.mesh_pose[3:]))
                builder.add_nonconvex_collision_from_file(object.usd_path, scene_pose)
                builder.add_visual_from_file(object.usd_path, scene_pose)
                curr_id = builder.build_static(name=object.name)
                log.debug(f"[DEBUG] Loaded NonConvexRigidObj '{object.name}' at pose: {scene_pose}")
                self.object_ids[object.name] = curr_id
                self.object_joint_order[object.name] = []

            elif isinstance(object, RigidObjCfg):
                is_mesh_load = hasattr(object, 'mesh_path') and object.mesh_path and object.mesh_path.endswith('.obj')
                is_urdf_load = hasattr(object, 'urdf_path') and object.urdf_path

                if is_urdf_load:
                    self.loader.fix_root_link = object.fix_base_link
                    self.loader.scale = object.scale[0]
                    file_path = object.urdf_path
                    curr_id: sapien_core.Entity
                    try:
                        curr_id = self.loader.load(file_path)
                    except Exception as e:
                        log.warning(f"Error loading {file_path}: {e}")
                        curr_id_list = self.loader.load_multiple(file_path)
                        for id_item in curr_id_list:
                            if len(id_item):
                                curr_id = id_item
                                break
                    if isinstance(curr_id, list):
                        curr_id = curr_id[0]
                    pose = sapien_core.Pose(p=object.default_position, q=object.default_orientation)
                    curr_id.set_pose(pose)
                    log.debug(f"[DEBUG] Loaded RigidObj '{object.name}' at pose: {pose}")
                    self.object_ids[object.name] = curr_id
                    self.object_joint_order[object.name] = []

                elif is_mesh_load:
                    log.info(f"Attempting to load .obj as rigid body: {object.mesh_path}")
                    builder = self.scene.create_actor_builder()
                    density = getattr(object, 'density', 1000.0)
                    
                    builder.add_convex_collision_from_file(filename=object.mesh_path, scale=object.scale, density=density)
                    builder.add_visual_from_file(filename=object.mesh_path, scale=object.scale)
                    
                    actor = builder.build(name=object.name) if not object.fix_base_link else builder.build_static(name=object.name)
                    pose = sapien_core.Pose(p=object.default_position, q=object.default_orientation)
                    actor.set_pose(pose)
                    log.debug(f"[DEBUG] Loaded OBJ '{object.name}' at pose: {pose}")

                    self.object_ids[object.name] = actor
                    self.object_joint_order[object.name] = []

                else:
                    log.warning(f"Object '{object.name}' has no valid urdf_path or mesh_path. Skipped.")

            if object.name in self.object_ids:
                loaded_entity = self.object_ids[object.name]
                if hasattr(loaded_entity, 'get_links'):
                    self.link_ids[object.name] = loaded_entity.get_links()
                else:
                    self.link_ids[object.name] = [comp for comp in loaded_entity.get_components() if isinstance(comp, sapien_core.physx.PhysxRigidBaseComponent)]

                if isinstance(object, (ArticulationObjCfg, BaseRobotCfg)) and len(self.object_joint_order.get(object.name, [])) > 0:
                    self._previous_dof_pos_target[object.name] = np.zeros((len(self.object_joint_order[object.name]),), dtype=np.float32)
                    self._previous_dof_vel_target[object.name] = np.zeros((len(self.object_joint_order[object.name]),), dtype=np.float32)
                    self._previous_dof_torque_target[object.name] = np.zeros((len(self.object_joint_order[object.name]),), dtype=np.float32)
                else:
                    if object.name not in self.link_ids:
                         self.link_ids[object.name] = []

        if not self.headless:
            self.viewer = Viewer(self.renderer)
            self.viewer.set_scene(self.scene)

        if not self.headless:
            camera_pos = np.array([1.5, -1.5, 1.5])
            camera_target = np.array([0.0, 0.0, 0.0])
            direction_vector = camera_target - camera_pos
            yaw = math.atan2(direction_vector[1], direction_vector[0])
            pitch = math.atan2(direction_vector[2], math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2))
            roll = 0
            self.viewer.set_camera_xyz(x=camera_pos[0], y=camera_pos[1], z=camera_pos[2])
            self.viewer.set_camera_rpy(r=roll, p=pitch, y=-yaw)

        self.debug_points = []
        self.debug_lines = []

        self.scene.update_render()
        for camera_name, camera_id in self.camera_ids.items():
            camera_id.take_picture()

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
        instance = self.object_ids[obj_name]
        if isinstance(instance, sapien_core.physx.PhysxArticulation):
            action = target[0]
            pos_target = action.get("dof_pos_target", None)
            vel_target = action.get("dof_vel_target", None)
            pos_target_arr = (
                np.array([pos_target[name] for name in self.object_joint_order[obj_name]]) if pos_target else None
            )
            vel_target_arr = (
                np.array([vel_target[name] for name in self.object_joint_order[obj_name]]) if vel_target else None
            )
            self._previous_dof_pos_target[obj_name] = pos_target_arr
            self._previous_dof_vel_target[obj_name] = vel_target_arr
            self._apply_action(instance, pos_target, vel_target)

    def _simulate(self):
        for i in range(self.scenario.decimation):
            self.scene.step()

        self.scene.update_render()
        if not self.headless:
            self.viewer.render()
        for camera_name, camera_id in self.camera_ids.items():
            camera_id.take_picture()

    def launch(self) -> None:
        self._build_sapien()

    def close(self):
        if not self.headless and self.viewer:
            self.viewer.close()
        self.scene = None

    def _get_link_states(self, obj_name: str) -> tuple[list, torch.Tensor]:
        link_name_list = []
        link_state_list = []

        if len(self.link_ids.get(obj_name, [])) == 0:
            return [], torch.zeros((0, 13), dtype=torch.float32)

        for link in self.link_ids[obj_name]:
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

    def _get_states(self, env_ids=None) -> EnvState:
        object_states = {}
        for obj in self.objects:
            obj_inst = self.object_ids[obj.name]
            pose = obj_inst.get_pose()
            link_names, link_state = self._get_link_states(obj.name)
            if isinstance(obj_inst, sapien_core.physx.PhysxArticulation):
                pos = torch.tensor(pose.p, dtype=torch.float32)
                rot = torch.tensor(pose.q, dtype=torch.float32)
                vel = torch.tensor(obj_inst.get_root_linear_velocity(), dtype=torch.float32)
                ang_vel = torch.tensor(obj_inst.get_root_angular_velocity(), dtype=torch.float32)
                root_state = torch.cat([pos, rot, vel, ang_vel], dim=-1).unsqueeze(0)
                joint_reindex = self.get_joint_reindex(obj.name)
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
                    log.warning(f"Could not find a physics component for actor '{obj.name}'. Reporting zero velocity.")
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
            robot_inst = self.object_ids[robot.name]
            assert isinstance(robot_inst, sapien_core.physx.PhysxArticulation)
            pose = robot_inst.get_pose()
            pos = torch.tensor(pose.p, dtype=torch.float32)
            rot = torch.tensor(pose.q, dtype=torch.float32)
            vel = torch.tensor(robot_inst.get_root_linear_velocity(), dtype=torch.float32)
            ang_vel = torch.tensor(robot_inst.get_root_angular_velocity(), dtype=torch.float32)
            root_state = torch.cat([pos, rot, vel, ang_vel], dim=-1).unsqueeze(0)
            joint_reindex = self.get_joint_reindex(robot.name)
            link_names, link_state = self._get_link_states(robot.name)
            pos_target = (
                torch.tensor(self._previous_dof_pos_target[robot.name], dtype=torch.float32).unsqueeze(0)
                if self._previous_dof_pos_target.get(robot.name) is not None
                else None
            )
            vel_target = (
                torch.tensor(self._previous_dof_vel_target[robot.name], dtype=torch.float32).unsqueeze(0)
                if self._previous_dof_vel_target.get(robot.name) is not None
                else None
            )
            effort_target = (
                torch.tensor(self._previous_dof_torque_target[robot.name], dtype=torch.float32).unsqueeze(0)
                if self._previous_dof_torque_target.get(robot.name) is not None
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
            cam_inst = self.camera_ids[camera.name]
            rgb = cam_inst.get_picture("Color")[..., :3]
            rgb = (rgb * 255).clip(0, 255).astype("uint8")
            rgb = torch.from_numpy(rgb.copy())
            depth = -cam_inst.get_picture("Position")[..., 2]
            depth = torch.from_numpy(depth.copy()).to(dtype=torch.float32)
            state = CameraState(rgb=rgb.unsqueeze(0), depth=depth.unsqueeze(0))
            camera_states[camera.name] = state

        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, sensors={})

    def refresh_render(self):
        log.debug("refresh_render called!")  # Confirm call
        self.scene.update_render()
        if not self.headless:
            self.viewer.render()
        for camera_name, camera_id in self.camera_ids.items():
            camera_id.take_picture()

        # Add collision detection (log **all** contacts in the scene)
        self._robot_contacts = []
        self._all_contacts = []
        contacts = self.scene.get_contacts()
        log.debug(f"Found {len(contacts)} contacts")  # Log number of contacts
        for contact in contacts:
            actor0 = contact.actor0.name if contact.actor0 else 'Unknown'
            actor1 = contact.actor1.name if contact.actor1 else 'Unknown'
            # Log every contact pair so user can see collisions between any two objects
            log.info(f"[Collision][Sapien3] {actor0} ↔ {actor1}")
            self._all_contacts.append((actor0, actor1, contact))

            # Additionally keep the per-robot contacts list for backward compatibility
            if self.robot and self.robot.name in (actor0, actor1):
                other = actor1 if actor0 == self.robot.name else actor0
                self._robot_contacts.append((self.robot.name, other, contact))

    def _set_states(self, states, env_ids=None):
        if isinstance(states, list):
            states = states[0]

        states_flat = states["objects"] | states["robots"]
        for name, val in states_flat.items():
            if name not in self.object_ids:
                continue
            obj_id = self.object_ids[name]

            if isinstance(obj_id, sapien_core.physx.PhysxArticulation):
                joint_names = self.object_joint_order[name]
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