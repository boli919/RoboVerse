import pickle
import numpy as np

# Please replace this with the actual path to your motion file
motion_file_path = '/home/boruili/humanoid-general-motion-tracking/assets/motions/humanml3d_replayed_filtered/000000_replayed.pkl'

with open(motion_file_path, 'rb') as f:
    data = pickle.load(f)

print(f"Keys in motion file: {data.keys()}")

if 'joint_names' in data:
    joint_names = data['joint_names']
    print(f"Joint names from motion file ({len(joint_names)}): {joint_names}")
else:
    print("'joint_names' key not found in motion file.")

if 'dof_pos' in data:
    dof_pos = data['dof_pos']
    if isinstance(dof_pos, np.ndarray):
        print(f"Shape of 'dof_pos': {dof_pos.shape}")
    else:
        print(f"'dof_pos' is not a numpy array.")

if 'root_pos' in data:
    root_pos = data['root_pos']
    if isinstance(root_pos, np.ndarray):
        print(f"Shape of 'root_pos': {root_pos.shape}")
    else:
        print(f"'root_pos' is not a numpy array.")

if 'root_rot' in data:
    root_rot = data['root_rot']
    if isinstance(root_rot, np.ndarray):
        print(f"Shape of 'root_rot': {root_rot.shape}")
    else:
        print(f"'root_rot' is not a numpy array.")
