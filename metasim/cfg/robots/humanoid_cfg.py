from __future__ import annotations

from metasim.utils import configclass
from metasim.cfg.robots.base_robot_cfg import BaseActuatorCfg, BaseRobotCfg

@configclass
class HumanoidCfg(BaseRobotCfg):
    name: str = "humanoid"
    num_joints: int = 23
    fix_base_link: bool = False

    # Define 22 arm joints and 1 end-effector joint
    actuators: dict[str, BaseActuatorCfg] = {
        **{f"joint_{i+1}": BaseActuatorCfg() for i in range(22)},
        "joint_23": BaseActuatorCfg(is_ee=True),
    }

    # Dummy joint limits, may need adjustment
    joint_limits: dict[str, tuple[float, float]] = {
        **{f"joint_{i+1}": (-3.14, 3.14) for i in range(22)},
        "joint_23": (0.0, 1.0),
    }

    ee_body_name: str = "joint_23" # Assuming the last joint is the end-effector

    # Dummy values for gripper, retargeting might rely on these
    gripper_open_q = [0.0]
    gripper_close_q = [1.0]

    # These values are important for curobo IK solver
    curobo_tcp_rel_pos: tuple[float, float, float] = [0.0, 0.0, 0.0]
    curobo_tcp_rel_rot: tuple[float, float, float] = [0.0, 0.0, 0.0]
