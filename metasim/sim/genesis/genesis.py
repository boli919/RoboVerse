from __future__ import annotations

import genesis as gs
import os
import numpy as np
import torch
from genesis.engine.entities.rigid_entity import RigidEntity, RigidJoint
from genesis.vis.camera import Camera
from loguru import logger as log
from scipy.spatial.transform import Rotation

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg, _FileBasedMixin
from metasim.cfg.scenario import SceneCfg
from metasim.sim import BaseSimHandler, GymEnvWrapper
from metasim.types import Action, EnvState
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState


class GenesisHandler(BaseSimHandler):
    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self._actions_cache: list[Action] = []
        self.object_inst_dict: dict[str, RigidEntity] = {}
        self.camera_inst_dict: dict[str, Camera] = {}
        # Create a rotation for correcting coordinate system differences (180 degrees around Z-axis)
        self.z_180_rotation = Rotation.from_euler('z', 180, degrees=True)

    def load_scene(self) -> None:
        """Load a static scene mesh if specified in the scenario configuration."""
        scene_cfg = self.scenario.scene
        if scene_cfg is None:
            return

        asset_path: str | None = None
        if getattr(scene_cfg, "usd_path", None):
            asset_path = scene_cfg.usd_path
        elif getattr(scene_cfg, "urdf_path", None):
            asset_path = scene_cfg.urdf_path
        elif getattr(scene_cfg, "mjcf_path", None):
            asset_path = scene_cfg.mjcf_path

        if asset_path is None:
            log.warning("Scene config has no asset path (usd/urdf/mjcf), skipping scene loading.")
            return

        if not os.path.exists(asset_path):
            log.warning(f"Scene asset not found: {asset_path}. Skipping scene loading.")
            return

        scale_val = 1.0
        if getattr(scene_cfg, "scale", None) is not None:
            scale_val = scene_cfg.scale if isinstance(scene_cfg.scale, (int, float)) else scene_cfg.scale[0]

        try:
            ext = os.path.splitext(asset_path)[1].lower()
            if ext in {".urdf", ".xml"}:
                ent = self.scene_inst.add_entity(
                    gs.morphs.URDF(file=asset_path, fixed=True, scale=scale_val),
                )
            else:
                ent = self.scene_inst.add_entity(
                    gs.morphs.Mesh(file=asset_path, scale=scale_val),
                )

            self.object_inst_dict[scene_cfg.name or "scene"] = ent
            log.info(f"Loaded scene asset '{asset_path}' into Genesis.")
        except Exception as e:
            log.warning(f"Failed to load scene asset '{asset_path}' in Genesis: {e}")
            return

    def launch(self) -> None:
        try:
            gs.init(backend=gs.gpu)
        except RuntimeError as exc:
            if "already initialized" in str(exc):
                log.warning("Genesis has already been initialized – skipping second initialization.")
            else:
                raise

        self.scene_inst = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.scenario.sim_params.dt if self.scenario.sim_params.dt is not None else 1 / 100,
                substeps=1,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=self.scenario.num_envs),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=not self.headless,
        )

        self.scene_inst.add_entity(gs.morphs.Plane())
        self.load_scene()

        self.robot_inst: RigidEntity = self.scene_inst.add_entity(
            gs.morphs.URDF(
                file=self.robot.urdf_path,
                fixed=self.robot.fix_base_link,
                merge_fixed_links=self.robot.collapse_fixed_joints,
            ),
            material=gs.materials.Rigid(gravity_compensation=1 if not self.robot.enabled_gravity else 0),
        )
        self.object_inst_dict[self.robot.name] = self.robot_inst

        for obj in self.scenario.objects:
            if isinstance(obj, _FileBasedMixin):
                if isinstance(obj.scale, (tuple, list)):
                    obj.scale = obj.scale[0]
                    log.warning(
                        f"Genesis does not support different scaling for each axis for {obj.name}, using scale={obj.scale}"
                    )
            if isinstance(obj, PrimitiveCubeCfg):
                obj_inst = self.scene_inst.add_entity(
                    gs.morphs.Box(size=obj.size), surface=gs.surfaces.Default(color=obj.color)
                )
            elif isinstance(obj, PrimitiveSphereCfg):
                obj_inst = self.scene_inst.add_entity(
                    gs.morphs.Sphere(radius=obj.radius), surface=gs.surfaces.Default(color=obj.color)
                )
            elif isinstance(obj, (RigidObjCfg, ArticulationObjCfg)):
                # -------------------------------------------------------------
                # Determine which asset path to use and load the object.
                # Priority: URDF > mesh_path > usd_path > mjcf_path. This allows
                # users to specify a variety of formats such as OBJ/STL/PLY via
                # ``mesh_path``.
                # -------------------------------------------------------------
                asset_path: str | None = None
                if getattr(obj, "urdf_path", None):
                    asset_path = obj.urdf_path
                elif getattr(obj, "mesh_path", None):
                    asset_path = obj.mesh_path
                elif getattr(obj, "usd_path", None):
                    asset_path = obj.usd_path
                elif getattr(obj, "mjcf_path", None):
                    asset_path = obj.mjcf_path

                if asset_path is None:
                    log.warning(
                        f"[Genesis] No valid asset path provided for object '{obj.name}'. Skipping this object.")
                    continue

                if not os.path.exists(asset_path):
                    log.warning(
                        f"[Genesis] Asset file for object '{obj.name}' not found at '{asset_path}'. Skipping this object.")
                    continue

                ext = os.path.splitext(asset_path)[1].lower()

                try:
                    if ext in {".urdf", ".xml", ".mjcf"}:
                        # Load as articulated/rigid URDF or MJCF object.
                        obj_inst = self.scene_inst.add_entity(
                            gs.morphs.URDF(file=asset_path, fixed=obj.fix_base_link, scale=obj.scale),
                        )
                    else:
                        # Treat all other formats (e.g., .obj, .stl, .ply, .usd) as meshes.
                        obj_inst = self.scene_inst.add_entity(
                            gs.morphs.Mesh(file=asset_path, scale=obj.scale),
                        )

                    log.info(
                        f"[Genesis] Successfully loaded object '{obj.name}' from '{asset_path}' (ext='{ext}').")
                except Exception as e:
                    log.warning(
                        f"[Genesis] Failed to load object '{obj.name}' from '{asset_path}' (ext='{ext}'): {e}")
                    continue
            else:
                raise NotImplementedError(f"Object type {type(obj)} not supported")
            self.object_inst_dict[obj.name] = obj_inst

        for camera in self.cameras:
            camera_inst = self.scene_inst.add_camera(
                res=(camera.width, camera.height),
                pos=camera.pos,
                lookat=camera.look_at,
                fov=camera.vertical_fov,
            )
            self.camera_inst_dict[camera.name] = camera_inst

        self.scene_inst.build(
            n_envs=self.scenario.num_envs, env_spacing=(self.scenario.env_spacing, self.scenario.env_spacing)
        )


        rb_pos_cfg = getattr(self.robot, "default_position", (0.0, 0.0, 0.0))
        rb_quat_cfg = getattr(self.robot, "default_orientation", (1.0, 0.0, 0.0, 0.0))  # w,x,y,z or x,y,z,w?

        rb_pos_corrected = (-rb_pos_cfg[0], -rb_pos_cfg[1], rb_pos_cfg[2])
        initial_pos = np.array([rb_pos_corrected] * self.scenario.num_envs, dtype=np.float32)

        if len(rb_quat_cfg) == 4 and abs(rb_quat_cfg[3]) > 0.99:
            rb_quat_cfg = (rb_quat_cfg[3], rb_quat_cfg[0], rb_quat_cfg[1], rb_quat_cfg[2])
        initial_quat = np.array([rb_quat_cfg] * self.scenario.num_envs, dtype=np.float32)

        self.robot_inst.set_pos(initial_pos)
        self.robot_inst.set_quat(initial_quat)

        # 2. Set scene position and rotation, applying coordinate system correction
        scene_cfg = self.scenario.scene
        if scene_cfg is not None and (scene_cfg.name or "scene") in self.object_inst_dict:
            ent = self.object_inst_dict[scene_cfg.name or "scene"]

            # Correct and set position
            pos_to_set = getattr(scene_cfg, "pos", None) or getattr(scene_cfg, "default_position", None)
            if pos_to_set is not None:
                log.info(f"Correcting scene position for Genesis coordinate system: {pos_to_set}")
                transformed_pos = np.array(pos_to_set, dtype=np.float32)
                transformed_pos[0] *= -1  # x' = -x
                transformed_pos[1] *= -1  # y' = -y
                pos_array = np.array([transformed_pos] * self.num_envs)
                ent.set_pos(pos_array)

            # Set rotation (usually identity for static scenes, so no correction needed unless specified)
            quat_to_set = getattr(scene_cfg, "rot", None) or getattr(scene_cfg, "quat", None)
            if quat_to_set is not None:
                if len(quat_to_set) == 4 and abs(quat_to_set[3]) > 0.99:
                    log.warning(f"Converting quat from (x, y, z, w) to (w, x, y, z) for scene {scene_cfg.name}")
                    quat_to_set = (quat_to_set[3], quat_to_set[0], quat_to_set[1], quat_to_set[2])
                ent.set_quat(np.array([quat_to_set] * self.num_envs))

        # 3. Set initial poses for all other objects from the scenario config
        log.info("Setting initial poses for scenario objects...")
        for obj_cfg in self.scenario.objects:
            if obj_cfg.name not in self.object_inst_dict:
                # log.warning(f"Object '{obj_cfg.name}' was configured but not found in instantiated objects. Skipping pose setting.")
                continue

            ent = self.object_inst_dict[obj_cfg.name]

            # Correct and set POSITION
            pos_to_set = getattr(obj_cfg, "default_position", None)
            if pos_to_set is not None:
                log.info(f"Setting initial position for '{obj_cfg.name}': {pos_to_set}")
                # Apply coordinate system correction for Genesis
                transformed_pos = np.array(pos_to_set, dtype=np.float32)
                transformed_pos[0] *= -1  # x' = -x
                transformed_pos[1] *= -1  # y' = -y
                pos_array = np.array([transformed_pos] * self.num_envs)
                ent.set_pos(pos_array)

            # Correct and set ROTATION
            quat_to_set = getattr(obj_cfg, "default_orientation", None)
            if quat_to_set is not None:
                log.info(f"Setting initial orientation for '{obj_cfg.name}': {quat_to_set}")
                
                # Ensure the input quaternion is a numpy array for all environments
                rot_data = np.array([quat_to_set] * self.num_envs, dtype=np.float32)
                
                # Config format is (x, y, z, w), which Scipy's from_quat expects.
                initial_rotation = Rotation.from_quat(rot_data)
                
                # Apply the 180-degree Z-axis rotation to correct the coordinate system.
                corrected_rotation = self.z_180_rotation * initial_rotation
                
                # Get the corrected quaternion in Scipy's (x, y, z, w) format.
                corrected_quat_scipy = corrected_rotation.as_quat()
                
                # Convert from Scipy's (x, y, z, w) to Genesis's (w, x, y, z) format.
                corrected_quat_genesis = corrected_quat_scipy[:, [3, 0, 1, 2]]
                
                ent.set_quat(rot_data)

    def _get_states(self, env_ids: list[int] | None = None) -> TensorState:
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        object_states = {}
        for obj in self.objects:
            obj_inst = self.object_inst_dict[obj.name]
            if isinstance(obj, ArticulationObjCfg):
                joint_reindex = self.get_joint_reindex(obj.name)
                state = ObjectState(
                    root_state=torch.cat(
                        [
                            obj_inst.get_pos(envs_idx=env_ids),
                            obj_inst.get_quat(envs_idx=env_ids),
                            obj_inst.get_vel(envs_idx=env_ids),
                            obj_inst.get_ang(envs_idx=env_ids),
                        ],
                        dim=-1,
                    ),
                    body_names=None,
                    body_state=None,
                    joint_pos=obj_inst.get_dofs_position(envs_idx=env_ids)[:, joint_reindex],
                    joint_vel=obj_inst.get_dofs_velocity(envs_idx=env_ids)[:, joint_reindex],
                )
            else:
                state = ObjectState(
                    root_state=torch.cat(
                        [
                            obj_inst.get_pos(envs_idx=env_ids),
                            obj_inst.get_quat(envs_idx=env_ids),
                            obj_inst.get_vel(envs_idx=env_ids),
                            obj_inst.get_ang(envs_idx=env_ids),
                        ],
                        dim=-1,
                    ),
                )
            object_states[obj.name] = state

        robot_states = {}
        for obj in [self.robot]:
            obj_inst = self.object_inst_dict[obj.name]
            joint_reindex = self.get_joint_reindex(obj.name)
            state = RobotState(
                root_state=torch.cat(
                    [
                        obj_inst.get_pos(envs_idx=env_ids),
                        obj_inst.get_quat(envs_idx=env_ids),
                        obj_inst.get_vel(envs_idx=env_ids),
                        obj_inst.get_ang(envs_idx=env_ids),
                    ],
                    dim=-1,
                ),
                body_names=None,
                body_state=None,
                joint_pos=obj_inst.get_dofs_position(envs_idx=env_ids)[:, joint_reindex],
                joint_vel=obj_inst.get_dofs_velocity(envs_idx=env_ids)[:, joint_reindex],
                joint_pos_target=None,
                joint_vel_target=None,
                joint_effort_target=None,
            )
            robot_states[obj.name] = state

        camera_states = {}
        for camera in self.cameras:
            camera_inst = self.camera_inst_dict[camera.name]
            rgb, depth, _, _ = camera_inst.render(depth=True)
            state = CameraState(
                rgb=torch.from_numpy(rgb.copy()).unsqueeze(0).repeat_interleave(self.num_envs, dim=0),
                depth=torch.from_numpy(depth.copy()).unsqueeze(0).repeat_interleave(self.num_envs, dim=0),
            )
            camera_states[camera.name] = state

        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, sensors={})

    def _set_states(self, states: list[EnvState], env_ids: list[int] | None = None) -> None:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        states_flat = [state["objects"] | state["robots"] for state in states]

        for obj in self.objects + [self.robot]:
            obj_inst = self.object_inst_dict[obj.name]

            # -------------------------------------------------------------
            # Trajectory / init_states may not include every object defined
            # in the current scenario. When the state dict for this object is
            # missing, we skip updating its pose instead of raising an error.
            # -------------------------------------------------------------
            if any(obj.name not in states_flat[env_id] for env_id in env_ids):
                log.warning(
                    f"[Genesis] No state data found for object '{obj.name}' in the provided trajectory; "
                    "keeping its current pose defined in the scenario config.")
                continue

            # Correct POSITION
            pos_data = np.array([states_flat[env_id][obj.name]["pos"] for env_id in env_ids])
            pos_data[:, 0] *= -1
            pos_data[:, 1] *= -1
            obj_inst.set_pos(pos_data)

            # Correct ROTATION
            rot_data = np.array([states_flat[env_id][obj.name]["rot"] for env_id in env_ids])
            # The incoming data is (w, x, y, z), convert to Scipy's (x, y, z, w)
            rot_data_scipy = rot_data[:, [1, 2, 3, 0]]
            traj_rotation = Rotation.from_quat(rot_data_scipy)
            corrected_rotation = self.z_180_rotation * traj_rotation
            corrected_quat_scipy = corrected_rotation.as_quat()
            # Convert back to Genesis's (w, x, y, z)
            corrected_quat_genesis = corrected_quat_scipy[:, [3, 0, 1, 2]]
            obj_inst.set_quat(corrected_quat_genesis)

            if isinstance(obj, ArticulationObjCfg):
                if obj.fix_base_link:
                    obj_inst.set_qpos(
                        np.array([
                            [
                                states_flat[env_id][obj.name]["dof_pos"][jn]
                                for jn in self.get_joint_names(obj.name, sort=False)
                            ]
                            for env_id in env_ids
                        ]),
                        envs_idx=env_ids,
                    )
                else:
                    traj_joint_names = list(states_flat[env_ids[0]][obj.name]["dof_pos"].keys())
                    name2joint = {j.name: j for j in self.object_inst_dict[obj.name].joints}
                    joint_names = [jn for jn in traj_joint_names if jn in name2joint and name2joint[jn].dof_idx_local is not None]
                    qs_idx_local = [name2joint[jn].dof_idx_local for jn in joint_names]
                    obj_inst.set_qpos(
                        np.array([
                            [states_flat[env_id][obj.name]["dof_pos"][jn] for jn in joint_names] for env_id in env_ids
                        ]),
                        qs_idx_local=qs_idx_local,
                        envs_idx=env_ids,
                    )

    def set_dof_targets(self, obj_name: str, actions: list[Action]) -> None:
        self._actions_cache = actions
        position = np.array([
            [
                actions[env_id][self.robot.name]["dof_pos_target"][jn]
                for jn in self.get_joint_names(obj_name, sort=False)
            ]
            for env_id in range(self.num_envs)
        ])

        dofs_to_control = [
            j.dof_idx_local
            for j in self.robot_inst.joints
            if j.dof_idx_local is not None
        ]
        if not self.object_dict[obj_name].fix_base_link:
            dofs_to_control = [idx for idx in dofs_to_control if self.robot_inst.joints[idx].name != self.robot_inst.base_joint.name]

        self.robot_inst.control_dofs_position(
            position=position,
            dofs_idx_local=dofs_to_control,
        )

    def _simulate(self):
        for _ in range(self.scenario.decimation):
            self.scene_inst.step()

        self._robot_contacts = []
        try:
            contacts = self.scene_inst.get_contacts()
            for c in contacts:
                if c.actor0 == self.robot_inst or c.actor1 == self.robot_inst:
                    other = c.actor1 if c.actor0 == self.robot_inst else c.actor0
                    self._robot_contacts.append((self.robot.name, other.name, c))
                    log.info(f"[Collision][Genesis] robot ↔ {other.name}, impulse={getattr(c, 'impulse', 'N/A')}")
        except AttributeError:
            
            pass

    @property
    def robot_contacts(self):
        return getattr(self, "_robot_contacts", [])

    def refresh_render(self):
        if not self.headless:
            self.scene_inst.viewer.update()
        self.scene_inst.visualizer.update()

    def close(self):
        pass

    def get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            joints: list[RigidJoint] = self.object_inst_dict[obj_name].joints
            valid_joints = [
                j for j in joints if j.dof_idx_local is not None and j.name != self.object_inst_dict[obj_name].base_joint.name
            ]

            if sort:
                return sorted([j.name for j in valid_joints])
            else:
                valid_joints.sort(key=lambda j: j.dof_idx_local)
                return [j.name for j in valid_joints]
        else:
            return []

    @property
    def num_envs(self) -> int:
        return self.scene_inst.n_envs

    @property
    def actions_cache(self) -> list[Action]:
        return self._actions_cache

    @property
    def device(self) -> torch.device:
        return gs.device


GenesisEnv = GymEnvWrapper(GenesisHandler)
