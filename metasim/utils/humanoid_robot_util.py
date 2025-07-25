"""Helper functions for humanoid robots, including h1 and h1_simple_hand."""

from __future__ import annotations

import torch

from metasim.utils.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
)


def torso_upright(envstate, robot_name: str):
    """Returns the projection of the torso's z-axis onto the world's z-axis.

    Args:
        envstate (dict): Environment state dictionary.
        robot_name (str): Name of the robot.

    Returns:
        float: The projection value, shape=(1,)
    """
    quat = envstate["robots"][robot_name]["body"]["pelvis"]["rot"]  # (4,)
    xmat = matrix_from_quat(torch.tensor(quat).unsqueeze(0))[0]  # (3, 3)
    return xmat[2, 2].item()


def torso_upright_tensor(envstate, robot_name: str):
    """Returns the projection of the torso's z-axis onto the world's z-axis for a batch of environments.

    Args:
        envstate: Environment state object with batched robot states.
        robot_name (str): Name of the robot.

    Returns:
        torch.Tensor: Projection values, shape=(batch_size,)
    """
    robot_body_name = envstate.robots[robot_name].body_names
    body_id = robot_body_name.index("pelvis")
    quat = envstate.robots[robot_name].body_state[:, body_id, 3:7]  # (batch_size, 4)
    xmat = matrix_from_quat(quat)  # (batch_size, 3, 3)
    return xmat[:, 2, 2]


def head_height(envstate, robot_name: str):
    """Returns the height of the head, actually the neck."""
    # raise NotImplementedError("head_height is not implemented for isaacgym and isaaclab")
    return envstate["robots"][robot_name]["head"]["pos"][
        2
    ]  # Good for mujoco, but isaacgym and isaaclab don't have head site


def neck_height(envstate, robot_name: str):
    """Returns the height of the neck."""
    return (
        envstate["robots"][robot_name]["body"]["left_shoulder_roll_link"]["pos"][2]
        + envstate["robots"][robot_name]["body"]["right_shoulder_roll_link"]["pos"][2]
    ) / 2


def head_height_tensor(envstate, robot_name: str):
    """Returns the height of the neck."""
    robot_body_name = envstate.robots[robot_name].body_names
    body_id = robot_body_name.index("head")
    body_pos = envstate.robots[robot_name].body_state[:, body_id, 2]
    return body_pos


def neck_height_tensor(envstate, robot_name: str):
    """Returns the height of the neck."""
    robot_body_name = envstate.robots[robot_name].body_names
    body_id_l = robot_body_name.index("left_shoulder_roll_link")
    body_id_r = robot_body_name.index("right_shoulder_roll_link")
    body_pos_l = envstate.robots[robot_name].body_state[:, body_id_l, 2]
    body_pos_r = envstate.robots[robot_name].body_state[:, body_id_r, 2]
    return (body_pos_l + body_pos_r) / 2


def left_foot_height(envstate, robot_name: str):
    """Returns the height of the left foot."""
    # return envstate[f"{_METASIM_SITE_PREFIX}left_foot"]["pos"][2] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["left_ankle_link"]["pos"][2]


def right_foot_height(envstate, robot_name: str):
    """Returns the height of the right foot."""
    # return envstate[f"{_METASIM_SITE_PREFIX}right_foot"]["pos"][2] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["right_ankle_link"]["pos"][2]


def robot_position(envstate, robot_name: str):
    """Returns position of the robot."""
    return envstate["robots"][robot_name]["pos"]


def robot_position_tensor(envstate, robot_name: str):
    """Returns position of the robot."""
    return envstate.robots[robot_name].root_state[:, 0:3]


def robot_velocity(envstate, robot_name: str):
    """Returns the velocity of the robot."""
    return envstate["robots"][robot_name]["vel"]


def robot_root_state_tensor(envstate, robot_name: str):
    """Returns the velocity of the robot."""
    return envstate.robots[robot_name].root_state


def robot_velocity_tensor(envstate, robot_name: str):
    """Returns the velocity of the robot."""
    return envstate.robots[robot_name].root_state[:, 7:10]


def robot_ang_velocity_tensor(envstate, robot_name: str):
    """Returns the angluar velocity of the robot."""
    return envstate.robots[robot_name].root_state[:, 10:13]


def robot_local_lin_vel_tensor(envstate, robot_name: str):
    """Returns the local frame velocity of the robot."""
    return envstate.robots[robot_name].extra["base_lin_vel"]


def robot_local_ang_vel_tensor(envstate, robot_name: str):
    """Returns the local frame velocity of the robot."""
    return envstate.robots[robot_name].extra["base_ang_vel"]


def last_robot_velocity_tensor(envstate, robot_name: str):
    """Returns the velocity of the robot."""
    return envstate.robots[robot_name].extra["last_robot_velocity"]


def robot_local_velocity_tensor(envstate, robot_name: str) -> torch.Tensor:
    """Batched (B,) → (B,2) conversion from world-frame XY velocity to robot-frame XY velocity.

    Args:
    ----
    envstate :  your batched simulation state object
    robot_name : str
        Name of the queried robot.

    Returns:
    -------
    torch.Tensor [B, 2]
        v_x_fwd  (column 0) and v_y_lat (column 1) in *robot* coordinates.
    """
    # (B,3)  – only XY are used here, but keep Z so downstream code can stay unchanged.
    v_world = robot_velocity_tensor(envstate, robot_name)

    # (B,4)  quaternion.  ***Assumed layout = (w, x, y, z).***
    # If your codebase stores (x,y,z,w) just swap the last two lines of the unbind.
    q = robot_rotation_tensor(envstate, robot_name)
    w, x, y, z = q.unbind(-1)

    # Closed-form yaw   (cf. Shoemake 1985)
    # yaw = atan2( 2(w z + x y), 1 – 2(y² + z²) )
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))  # (B,)

    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)

    # Rotate world-frame XY into robot frame:
    v_x_local = v_world[:, 0] * cos_y + v_world[:, 1] * sin_y
    v_y_local = -v_world[:, 0] * sin_y + v_world[:, 1] * cos_y

    return torch.stack((v_x_local, v_y_local), dim=-1)


def default_dof_pos_tensor(envstates, robot_name: str):
    """Return the default pos of the robot."""
    return envstates.robots[robot_name].extra["default_pos"]


def ref_dof_pos_tensor(envstates, robot_name: str):
    """Return the default ref dof pos."""
    return envstates.robots[robot_name].extra["ref_dof_pos"]


def get_euler_xyz_tensor(quat):
    """Convert quaternion to Euler angles (roll, pitch, yaw) in radians for a batch of quaternions.

    Args:
        quat (torch.Tensor): Quaternion tensor of shape (N, 4) where N is the batch size.

    Returns:
        torch.Tensor: Euler angles tensor of shape (N, 3) where each row contains (roll, pitch, yaw).
    """
    r, p, w = euler_xyz_from_quat(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > torch.pi] -= 2 * torch.pi
    return euler_xyz


def robot_rotation(envstate, robot_name: str):
    """Returns the rotation of the robot."""
    return envstate["robots"][robot_name]["rot"]


def robot_rotation_tensor(envstate, robot_name: str):
    """Returns the rotation of the robot."""
    return envstate.robots[robot_name].root_state[:, 3:7]


def torso_vertical_orientation(envstate, robot_name: str):
    """Returns the z-projection of the torso orientation matrix.

    Args:
        envstate: Environment state object with batched robot states.
        robot_name (str): Name of the robot.

    Returns:
        torch.Tensor: z-axis projection, shape=(batch_size,)
    """
    quat = envstate["robots"][robot_name]["body"]["pelvis"]["rot"]  # (4,)
    xmat = matrix_from_quat(torch.tensor(quat).unsqueeze(0))[0]  # (3, 3)
    return xmat[2, :]


def dof_pos_tensor(envstate, robot_name: str):
    """Returns the pos."""
    return (
        envstate.robots[robot_name].joint_pos
        if envstate.robots[robot_name].joint_pos is not None
        else torch.zeros_like(envstate.robots[robot_name].joint_pos)
    )


def dof_vel_tensor(envstate, robot_name: str):
    """Returns  the pos."""
    return (
        envstate.robots[robot_name].joint_vel
        if envstate.robots[robot_name].joint_vel is not None
        else torch.zeros_like(envstate.robots[robot_name].joint_vel)
    )


def last_dof_vel_tensor(envstate, robot_name: str):
    """Returns last dof velocity."""
    return envstate.robots[robot_name].extra["last_dof_vel"]


def ref_dof_pos_tenosr(envstate, robot_name: str):
    """Returns the dof pos."""
    return envstate.robots[robot_name].extra["ref_dof_pos"]


def last_foot_pos_tensor(envstate, robot_name: str):
    """Returns last foot pos."""
    return envstate.robots[robot_name].extra["last_foot_pos"]


def foot_vel_tensor(envstate, robot_name: str):
    """Returns the foot vel."""
    return envstate.robots[robot_name].extra["foot_vel"]


def knee_pos_tensor(envstate, robot_name: str):
    """Returns the knee pos."""
    return envstate.robots[robot_name].extra["knee_pos"]


def elbow_pos_tensor(envstate, robot_name: str):
    """Returns  the elbow pos."""
    return envstate.robots[robot_name].extra["elbow_pos"]


def contact_forces_tensor(envstate, robot_name: str):
    """Returns the contact forces."""
    return envstate.robots[robot_name].extra["contact_forces"]


def gait_phase_tensor(envstate, robot_name: str):
    """Returns gait phase."""
    return envstate.robots[robot_name].extra["gait_phase"]


def foot_air_time_tensor(envstate, robot_name: str):
    """Returns the foot air time."""
    return envstate.robots[robot_name].extra["foot_air_time"]


def command_tensor(envstate, robot_name: str):
    """Returns the command."""
    return envstate.robots[robot_name].extra["commnad"]


def actuator_knee_pos_tensor(envstate, robot_name: str):
    """Returns  the knee pos."""
    knee_pos = envstate.robots[robot_name].extra["knee_states"][:, :, :2]
    if knee_pos is None:
        raise ValueError(f"feet_pos is None for robot {robot_name}")
    return knee_pos


def contact_force_tensor(envstate, robot_name: str):
    """Returns  the knee pos."""
    contact_force = envstate.robots[robot_name].extra["contact_forces"]
    if contact_force is None:
        raise ValueError(f"feet_pos is None for robot {robot_name}")
    return contact_force


def actuator_forces(envstate, robot_name: str):
    """Returns  the forces applied by the actuators."""
    return (
        torch.tensor([x for x in envstate["robots"][robot_name]["dof_torque"].values()])
        if envstate["robots"][robot_name].get("dof_torque", None) is not None
        else torch.zeros(len(envstate["robots"][robot_name]["dof_pos"]))
    )


def actuator_forces_tensor(envstate, robot_name: str):
    """Returns the forces applied by the actuators."""
    return (
        envstate.robots[robot_name].joint_effort_target
        if envstate.robots[robot_name].joint_effort_target is not None
        else torch.zeros_like(envstate.robots[robot_name].joint_pos)
    )


def left_hand_position(envstate, robot_name: str):
    """Returns the position of the left hand."""
    # return envstate[f"{_METASIM_SITE_PREFIX}left_hand"]["pos"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["left_elbow_link"]["pos"]


def left_hand_velocity(envstate, robot_name: str):
    """Returns the velocity of the left hand."""
    # return envstate[f"{_METASIM_BODY_PREFIX}left_hand"]["left_hand_subtreelinvel"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["left_elbow_link"]["vel"]


def left_hand_orientation(envstate, robot_name: str):
    """Returns the orientation of the left hand."""
    # return envstate[f"{_METASIM_SITE_PREFIX}left_hand"]["rot"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["left_elbow_link"]["rot"]


def right_hand_position(envstate, robot_name: str):
    """Returns the position of the right hand."""
    # return envstate[f"{_METASIM_SITE_PREFIX}right_hand"]["pos"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["right_elbow_link"]["pos"]


def right_hand_velocity(envstate, robot_name: str):
    """Returns the velocity of the right hand."""
    # return envstate[f"{_METASIM_BODY_PREFIX}right_hand"]["right_hand_subtreelinvel"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["right_elbow_link"]["vel"]


def right_hand_orientation(envstate, robot_name: str):
    """Returns the orientation of the right hand."""
    # return envstate[f"{_METASIM_SITE_PREFIX}right_hand"]["rot"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["right_elbow_link"]["rot"]


def actions_tensor(envstate, robot_name: str):
    """Return actions tensor."""
    return envstate.robots[robot_name].extra["actions"]


def last_actions_tensor(envstate, robot_name: str):
    """Return last actions tensor."""
    return envstate.robots[robot_name].extra["last_actions"]
