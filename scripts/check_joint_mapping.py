"""Check joint mapping correctness between a v2 trajectory file and the target robot configuration.

Usage::

    python scripts/check_joint_mapping.py \
        --traj  metasim/data/walk/v2/g1_v2.pkl.gz \
        --robot g1

It prints:
1. Robot actuator order (cfg definition)
2. Alphabetically-sorted joint names (Genesis/Sapien internal order)
3. The first state's ``dof_pos`` key order from the trajectory
4. Index mapping vectors so you can see mismatches at a glance.
"""

from __future__ import annotations

import argparse
import gzip
import pickle as pkl
from importlib import import_module


def load_robot_cfg(robot_name: str):  # type: ignore[return-any]
    mod = import_module(f"metasim.cfg.robots.{robot_name}_cfg")
    for attr in dir(mod):
        if attr.lower().startswith(robot_name) and attr.lower().endswith("cfg"):
            return getattr(mod, attr)()
    raise ImportError(f"Cannot find robot cfg class for '{robot_name}'.")


def load_v2(traj_path: str, robot_name: str):  # type: ignore[return-any]
    opener = gzip.open if traj_path.endswith(".gz") else open
    with opener(traj_path, "rb") as f:
        data = pkl.load(f)
    traj = data[robot_name][0]
    first_state = traj["states"][0][robot_name]
    return first_state["dof_pos"]


def main():
    parser = argparse.ArgumentParser(description="Verify joint mapping between v2 file and robot cfg")
    parser.add_argument("--traj", required=True, help="Path to *_v2.pkl[.gz] trajectory file")
    parser.add_argument("--robot", default="g1", help="Robot cfg name (e.g. g1, h1)")
    args = parser.parse_args()

    robot = load_robot_cfg(args.robot)
    dof_pos_dict = load_v2(args.traj, robot.name)

    cfg_order = list(robot.actuators.keys())
    alpha_order = sorted(cfg_order)
    traj_order = list(dof_pos_dict.keys())

    print("=== Robot actuator order (cfg) ===")
    print(cfg_order)
    print("\n=== Alphabetical order (internal) ===")
    print(alpha_order)
    print("\n=== dof_pos key order in trajectory ===")
    print(traj_order)

    # Build mapping vectors
    map_cfg2traj = [traj_order.index(jn) if jn in traj_order else -1 for jn in cfg_order]
    map_alpha2traj = [traj_order.index(jn) if jn in traj_order else -1 for jn in alpha_order]

    print("\nIndex of each cfg joint in trajectory (-1 means missing):")
    print(map_cfg2traj)
    print("\nIndex of each alphabetical joint in trajectory:")
    print(map_alpha2traj)

    missing = [jn for jn, idx in zip(cfg_order, map_cfg2traj) if idx == -1]
    if missing:
        print("\n[WARN] Missing joints in trajectory:", missing)
    else:
        print("\nAll robot joints present in trajectory.")

    # Simple correctness heuristic: both mapping lists should be strictly increasing
    def is_strict_increasing(seq):
        return all(earlier < later for earlier, later in zip(seq, seq[1:]))

    if is_strict_increasing(map_cfg2traj):
        print("\n✓ cfg_order matches trajectory order.")
    else:
        print("\n✗ cfg_order does NOT match trajectory order. You probably need remapping in the converter.")

    if is_strict_increasing(map_alpha2traj):
        print("✓ alphabetical order matches trajectory order (good for Genesis/Sapien).")
    else:
        print("✗ alphabetical order does NOT match trajectory order (converter should sort keys).")


if __name__ == "__main__":
    main()
