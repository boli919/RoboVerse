#!/usr/bin/env python3
"""
Example demonstrating how scenes are used in RoboVerse.

This is a simple example showing how to:
1. Create a scenario with a specific scene
2. Use scene configurations
3. Load and run a simulation with a scene
"""

from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.scenes import KitchenCfg, ManycoreScene827313Cfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils.setup_util import get_sim_env_class

# Example 1: Using a kitchen scene
print("=== Example 1: Kitchen Scene ===")
kitchen_scene = KitchenCfg()
print(f"Scene name: {kitchen_scene.name}")
print(f"USD path: {kitchen_scene.usd_path}")
print(f"Default position: {kitchen_scene.default_position}")
print(f"Scale: {kitchen_scene.scale}")

# Example 2: Using a manycore scene
print("\n=== Example 2: Manycore Scene ===")
manycore_scene = ManycoreScene827313Cfg()
print(f"Scene name: {manycore_scene.name}")
print(f"USD path: {manycore_scene.usd_path}")
print(f"Default position: {manycore_scene.default_position}")
print(f"Scale: {manycore_scene.scale}")

# Example 3: Create a scenario with a scene
print("\n=== Example 3: Scenario with Scene ===")
scenario = ScenarioCfg(
    robots=["franka"],
    scene=kitchen_scene,
    sim="isaaclab",
    headless=True,
    num_envs=1,
)

# Add a camera to view the scene
scenario.cameras = [
    PinholeCameraCfg(
        width=512, 
        height=512, 
        pos=(2.0, -2.0, 1.5), 
        look_at=(0.0, 0.0, 0.0)
    )
]

print(f"Scenario created with scene: {scenario.scene.name}")
print(f"Try add table: {scenario.try_add_table}")
print(f"Number of environments: {scenario.num_envs}")

# Example 4: Load scene by name
print("\n=== Example 4: Load Scene by Name ===")
from metasim.utils.setup_util import get_scene

# Load scene by name
loaded_scene = get_scene("kitchen")
print(f"Loaded scene: {loaded_scene.name}")
print(f"USD path: {loaded_scene.usd_path}")

# Example 5: Scene randomization
print("\n=== Example 5: Scene Randomization ===")
random_scenario = ScenarioCfg(
    robots=["franka"],
    sim="isaaclab",
    headless=True,
    num_envs=1,
)

# Enable scene randomization
random_scenario.random.scene = True
print(f"Scene randomization enabled: {random_scenario.random.scene}")

# Note: When scene randomization is enabled, the system will randomly
# select from available scenes during runtime