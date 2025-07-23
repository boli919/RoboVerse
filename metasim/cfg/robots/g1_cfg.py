from __future__ import annotations

"""Configuration for the **G1** humanoid robot.

The parameters come from *humanoid-general-motion-tracking/configs/robot_configs.py*.
Only the fields that are actually required by MetaSim are filled.  Feel free to
adjust joint limits or other fine-tuning parameters later.
"""

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class G1Cfg(BaseRobotCfg):
    """Humanoid G1 configuration (23 actuated DoFs)."""

    # ---------------------------------------------------------------------
    # Basic identifiers & asset paths
    # ---------------------------------------------------------------------

    name: str = "g1"
    num_joints: int = 23

    # NOTE: Make sure these asset files exist locally or can be fetched from
    # HuggingFace via `metasim.utils.hf_util`.
    usd_path: str | None = "roboverse_data/robots/g1/usd/g1.usd"  # placeholder
    mjcf_path: str | None = "roboverse_data/robots/g1/mjcf/g1.xml"  # placeholder
    urdf_path: str | None = "resources/robots/g1/g1.urdf"

    enabled_gravity: bool = True
    # Set to True for fixing the base link during debugging to prevent root body dynamics.
    fix_base_link: bool = True
    enabled_self_collisions: bool = False
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = True

    # ------------------------------------------------------------------
    # Actuator settings (stiffness / damping taken from original config)
    # ------------------------------------------------------------------

    _joint_names = [
        #  Left leg (6)
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        #  Right leg (6)
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        #  Waist (3)
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        #  Left arm (4)
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        #  Right arm (4)
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ]

    _stiffness_vals = [
        100, 100, 100, 150, 40, 40,  # left leg
        100, 100, 100, 150, 40, 40,  # right leg
        150, 150, 150,              # waist
        40, 40, 40, 40,             # left arm
        40, 40, 40, 40,             # right arm
    ]

    _damping_vals = [
        2, 2, 2, 4, 2, 2,           # left leg
        2, 2, 2, 4, 2, 2,           # right leg
        4, 4, 4,                    # waist
        5, 5, 5, 5,                 # left arm
        5, 5, 5, 5,                 # right arm
    ]

    actuators: dict[str, BaseActuatorCfg] = {
        jn: BaseActuatorCfg(stiffness=stf, damping=dmp)
        for jn, stf, dmp in zip(_joint_names, _stiffness_vals, _damping_vals)
    }

    # ------------------------------------------------------------------
    # Joint limits (rough – customise if you have precise specs)
    # ------------------------------------------------------------------

    joint_limits: dict[str, tuple[float, float]] = {jn: (-3.14, 3.14) for jn in _joint_names}

    # ------------------------------------------------------------------
    # Default pose (taken from original default_dof_pos array)
    # ------------------------------------------------------------------

    _default_pos_vals = [
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,      # left leg
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,      # right leg
        0.0, 0.0, 0.0,                       # waist
        0.0, 0.4, 0.0, 1.2,                  # left arm
        0.0, -0.4, 0.0, 1.2,                 # right arm
    ]

    default_joint_positions: dict[str, float] = {
        jn: pos for jn, pos in zip(_joint_names, _default_pos_vals)
    }

    # ------------------------------------------------------------------
    # Control type – effort for all joints (same as H1)
    # ------------------------------------------------------------------

    control_type: dict[str, Literal["position", "effort"]] = {jn: "effort" for jn in _joint_names}

    # ------------------------------------------------------------------
    # Useful body link groups (optional, mostly for reward or termination)
    # ------------------------------------------------------------------

    feet_links: list[str] = ["left_ankle_pitch", "right_ankle_pitch"]
    knee_links: list[str] = ["left_knee", "right_knee"]
    elbow_links: list[str] = ["left_elbow", "right_elbow"]

    terminate_after_contacts_on_links: list[str] = [
        "left_elbow",
        "right_elbow",
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
    ]
    terminate_contacts_links: list[str] = ["pelvis", "torso", "shoulder", "elbow"]
    penalized_contacts_links: list[str] = ["hip", "knee"]
