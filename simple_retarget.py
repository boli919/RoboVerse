import argparse
import gzip
import os
import pickle as pkl
from typing import Dict, List

import numpy as np
from loguru import logger as log

# We will use the G1 Cfg to get the joint names
from metasim.cfg.robots.g1_cfg import G1Cfg


def tensor_to_list(tensor):
    if hasattr(tensor, "tolist"):
        return tensor.tolist()
    return tensor


def simple_retarget(input_path: str, output_path: str):
    """
    Directly maps humanoid trajectory to G1 robot trajectory.
    This is a simplified, direct-mapping approach without complex IK/FK retargeting.
    It assumes the first 21 joints of the humanoid correspond to the G1's joints.
    """
    g1_cfg = G1Cfg()
    g1_joint_names = list(g1_cfg.actuators.keys())
    num_g1_joints = len(g1_joint_names)
    log.info(f"Target robot 'g1' has {num_g1_joints} actuated joints.")

    # Load source v2 trajectory (humanoid)
    with open(input_path, "rb") as f:
        src_data = pkl.load(f)

    # Assuming the source robot is named 'g1' as we defined in the previous script
    src_traj = src_data["g1"][0]
    T = len(src_traj["actions"])
    log.info(f"Loaded {T} frames from source trajectory '{input_path}'.")

    # Build new actions & states sequence for G1
    actions: List[Dict[str, Dict[str, float]]] = []
    states: List[Dict[str, Dict]] = []

    # Get the original humanoid states to extract dof_pos
    original_states = src_traj["states"]

    for t in range(T):
        # Get all 23 DoF positions from the source humanoid trajectory
        humanoid_dof_pos = original_states[t]["g1"]["dof_pos"]

        # Create a dictionary for the first 21 joints, assuming they map to G1
        g1_q_dict = {g1_joint_names[i]: humanoid_dof_pos[f"joint_{i+1}"] for i in range(num_g1_joints)}

        action_t = {"dof_pos_target": g1_q_dict}

        # Use the root motion from the humanoid for the G1 robot's state
        state_t = {
            g1_cfg.name: {
                "pos": original_states[t]["g1"]["pos"],
                "rot": original_states[t]["g1"]["rot"],
                "dof_pos": g1_q_dict,
            }
        }

        actions.append(action_t)
        states.append(state_t)

    # Assemble v2 trajectory dict for G1
    traj = {
        "init_state": states[0],
        "actions": actions,
        "states": states,
    }

    data_v2 = {g1_cfg.name: [traj]}

    # Save to output (pickle or pickle.gz)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith(".gz"):
        with gzip.open(output_path, "wb", compresslevel=1) as f:
            pkl.dump(data_v2, f)
    else:
        with open(output_path, "wb") as f:
            pkl.dump(data_v2, f)

    log.success(f"Saved simplified retargeted v2 trajectory to '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Simplified retargeting from a humanoid v2 pkl to G1 robot.")
    parser.add_argument("--input", required=True, help="Path to the humanoid v2 source trajectory file.")
    parser.add_argument("--output", required=True, help="Target path to save G1 v2 trajectory.")
    args = parser.parse_args()

    simple_retarget(args.input, args.output)


if __name__ == "__main__":
    main()
