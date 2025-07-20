"""Convert a HumanML3D / humanoid-general-motion-tracking replay pickle file
into RoboVerse v2 trajectory format that `metasim` expects.

This script can process files in batches. It scans an input directory for
all ``*.pkl`` files, converts them, and saves them to an output directory,
preserving the subdirectory structure.

Usage::

    # Convert all files from a source directory to a target directory
    python scripts/convert_humanml_motion_to_v2.py \
        --input-dir /path/to/source_pkls/ \
        --output-dir /path/to/v2_trajectories/ \
        --robot g1
"""

from __future__ import annotations

import argparse
import gzip
import os
import pickle as pkl
from typing import Dict, List

import numpy as np
from loguru import logger as log

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------


def tensor_to_list(data):  # type: ignore[return-any]
    """Recursively convert ``np.ndarray`` or other tensor-ish objects to Python lists."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, dict):
        return {k: tensor_to_list(v) for k, v in data.items()}
    if isinstance(data, list):
        return [tensor_to_list(v) for v in data]
    return data


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Convert HumanML3D replay pickle to RoboVerse v2 format")
    parser.add_argument(
        "--input-dir",
        default="motion_data/mimic_filtered",
        help="Path to the source directory containing *.pkl files",
    )
    parser.add_argument(
        "--output-dir",
        default="motion_data/mimic_filtered_v2",
        help="Target directory to save v2 trajectories",
    )
    parser.add_argument(
        "--robot",
        default="g1",
        help="Robot name defined in metasim.cfg.robots (e.g. 'h1', 'h1_wrist', or custom).",
    )
    return parser


# -----------------------------------------------------------------------------
# Main conversion logic
# -----------------------------------------------------------------------------


def convert(input_path: str, output_path: str, robot_name: str) -> None:
    """Converts a single HumanML3D replay file to a RoboVerse v2 trajectory file."""
    # ------------------------------------------------------------------
    # 1. Dynamically import the robot cfg class
    # ------------------------------------------------------------------
    try:
        from importlib import import_module

        robot_module = import_module(f"metasim.cfg.robots.{robot_name}_cfg")
        robot_cls = None
        for attr in dir(robot_module):
            if attr.lower().startswith(robot_name) and attr.lower().endswith("cfg"):
                robot_cls = getattr(robot_module, attr)
                break
        if robot_cls is None:
            raise ImportError
        robot = robot_cls()
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            f"Unable to locate robot config for '{robot_name}'. Make sure a corresponding ``metasim.cfg.robots.{robot_name}_cfg`` module exists."  # noqa: E501
        ) from exc

    joint_names: List[str] = list(robot.actuators.keys())
    num_joints = len(joint_names)
    log.info(f"Detected robot '{robot.name}' with {num_joints} actuated joints")

    # ------------------------------------------------------------------
    # 2. Load source pickle (HumanML3D replay format)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Build actions & states sequence
    # ------------------------------------------------------------------
    actions: List[Dict[str, Dict[str, float]]] = []
    states: List[Dict[str, Dict]] = []

    # ------------------------------------------------------------------
    # Helper to convert quaternion from (x, y, z, w) to (w, x, y, z)
    # ------------------------------------------------------------------

    def xyzw_to_wxyz(quat: np.ndarray | list[float]):  # type: ignore[return-any]
        """Reorder quaternion from (x, y, z, w) to (w, x, y, z)."""
        if isinstance(quat, np.ndarray):
            return quat[[3, 0, 1, 2]]
        # assume list[float]
        return [quat[3], quat[0], quat[1], quat[2]]

    for t in range(T):
        # Map DoF values to joint names (truncate / ignore extra dims if any)
        q_dict = {jn: float(dof_pos[t, i]) for i, jn in enumerate(joint_names)}

        action_t = {"dof_pos_target": q_dict}

        # Convert quaternion order for RoboVerse/Sapien (w, x, y, z)
        rot_wxyz = xyzw_to_wxyz(root_rot[t])

        state_t = {
            robot.name: {
                "pos": tensor_to_list(root_pos[t]),
                "rot": tensor_to_list(rot_wxyz),
                "dof_pos": q_dict,
            }
        }

        actions.append(action_t)
        states.append(state_t)

    # ------------------------------------------------------------------
    # 4. Assemble v2 trajectory dict
    # ------------------------------------------------------------------
    traj = {
        "init_state": states[0],
        "actions": actions,
        "states": states,  # keep full state sequence for accurate replay; set to None to reduce size
    }

    data_v2 = {robot.name: [traj]}

    # ------------------------------------------------------------------
    # 5. Save to output (pickle or pickle.gz)
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith(".gz"):
        with gzip.open(output_path, "wb", compresslevel=1) as f:
            pkl.dump(data_v2, f)
    else:
        with open(output_path, "wb") as f:
            pkl.dump(data_v2, f)

    log.success(f"Saved v2 trajectory to '{output_path}'")


def main(args: argparse.Namespace) -> None:
    """Finds all motions in the input directory and converts them in batch."""
    if not os.path.isdir(args.input_dir):
        log.error(f"Input directory not found: {args.input_dir}")
        return

    log.info(f"Scanning for *.pkl files in '{args.input_dir}'...")
    converted_count = 0
    failed_count = 0

    for root, _, files in os.walk(args.input_dir):
        for filename in files:
            if not filename.endswith(".pkl"):
                continue

            input_path = os.path.join(root, filename)

            # Construct the output path, preserving the subdirectory structure
            relative_path = os.path.relpath(input_path, args.input_dir)
            base_name, _ = os.path.splitext(relative_path)
            output_filename = f"{base_name}_v2.pkl"
            output_path = os.path.join(args.output_dir, output_filename)

            log.info("-" * 80)
            log.info(f"Converting '{input_path}' -> '{output_path}'")
            try:
                convert(input_path, output_path, args.robot)
                converted_count += 1
            except Exception as e:
                log.error(f"Failed to convert '{input_path}': {e}")
                failed_count += 1

    log.info("-" * 80)
    log.success(f"Batch conversion complete. Converted {converted_count} files.")
    if failed_count > 0:
        log.warning(f"Failed to convert {failed_count} files.")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)