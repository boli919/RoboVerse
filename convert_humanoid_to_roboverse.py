import argparse
import gzip
import os
import pickle as pkl
from typing import Dict, List

import numpy as np
from loguru import logger as log


def tensor_to_list(tensor):
    if hasattr(tensor, "tolist"):
        return tensor.tolist()
    return tensor


def convert_humanml_to_v2(input_path: str, output_path: str):
    """
    Converts a HumanML3D replay pickle file to a RoboVerse v2 trajectory format.
    A virtual 'humanoid' robot config is implicitly defined here.
    """
    robot_name = "humanoid"
    # The source pkl has 23 DoFs. We'll create generic names for them.
    joint_names = [f"joint_{i+1}" for i in range(23)]
    num_joints = len(joint_names)
    log.info(f"Using a virtual robot '{robot_name}' with {num_joints} actuated joints")

    # Load source pickle (HumanML3D replay format)
    with open(input_path, "rb") as f:
        src = pkl.load(f)

    root_pos: np.ndarray = src["root_pos"]  # (T, 3)
    root_rot: np.ndarray = src["root_rot"]  # (T, 4)
    dof_pos: np.ndarray = src["dof_pos"]  # (T, N)

    assert dof_pos.shape[1] >= num_joints, (
        "Source DoF dimension is smaller than robot actuator number. "
        f"Expected ≥{num_joints}, got {dof_pos.shape[1]}"
    )

    T = dof_pos.shape[0]
    log.info(f"Loaded {T} frames from '{input_path}' (fps={src.get('fps', 'N/A')})")

    # Build actions & states sequence
    actions: List[Dict[str, Dict[str, float]]] = []
    states: List[Dict[str, Dict]] = []

    # Helper: (x, y, z, w) -> (w, x, y, z)
    def xyzw_to_wxyz(quat: np.ndarray | List[float]):  # type: ignore[return-any]
        if isinstance(quat, np.ndarray):
            return quat[[3, 0, 1, 2]]
        return [quat[3], quat[0], quat[1], quat[2]]

    for t in range(T):
        # Map DoF values to joint names
        q_dict = {jn: float(dof_pos[t, i]) for i, jn in enumerate(joint_names)}

        action_t = {"dof_pos_target": q_dict}
        rot_wxyz = xyzw_to_wxyz(root_rot[t])
        state_t = {
            robot_name: {
                "pos": tensor_to_list(root_pos[t]),
                "rot": tensor_to_list(rot_wxyz),
                "dof_pos": q_dict,
            }
        }

        actions.append(action_t)
        states.append(state_t)

    # Assemble v2 trajectory dict
    traj = {
        "init_state": states[0],
        "actions": actions,
        "states": states,
    }

    data_v2 = {robot_name: [traj]}

    # Save to output (pickle or pickle.gz)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith(".gz"):
        with gzip.open(output_path, "wb", compresslevel=1) as f:
            pkl.dump(data_v2, f)
    else:
        with open(output_path, "wb") as f:
            pkl.dump(data_v2, f)

    log.success(f"Saved v2 trajectory to '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Convert HumanML3D replay pickle to RoboVerse v2 format for retargeting.")
    parser.add_argument("--input", required=True, help="Path to the *.pkl source file from HumanML3D replay.")
    parser.add_argument("--output", required=True, help="Target path to save v2 trajectory (e.g., humanoid_v2.pkl).")
    args = parser.parse_args()

    convert_humanml_to_v2(args.input, args.output)


if __name__ == "__main__":
    main()
