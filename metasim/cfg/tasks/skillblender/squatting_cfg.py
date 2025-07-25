"""Squattting config in SkillBench in Skillblender"""

from __future__ import annotations

from typing import Callable

import torch

from metasim.cfg.simulator_params import SimParamCfg
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg
from metasim.cfg.tasks.skillblender.base_humanoid_cfg import BaseHumanoidCfg
from metasim.cfg.tasks.skillblender.base_legged_cfg import (
    BaseConfig,
    CommandRanges,
    CommandsConfig,
    LeggedRobotCfgPPO,
    RewardCfg,
)
from metasim.cfg.tasks.skillblender.reward_func_cfg import (
    reward_default_joint_pos,
    reward_dof_acc,
    reward_dof_vel,
    reward_feet_distance,
    reward_orientation,
    reward_torques,
    reward_upper_body_pos,
)
from metasim.types import EnvState

# from metasim.cfg.tasks.skillblender.reward_func_cfg import *  # FIXME star import
from metasim.utils import configclass
from metasim.utils.humanoid_robot_util import *


# define new reward function
def reward_squatting(env_states: EnvState, robot_name: str, cfg: BaseRLTaskCfg):
    """
    Calculates the reward based on the difference between the current root height and the target root height.
    """
    root_height = env_states.robots[robot_name].root_state[:, 2].unsqueeze(1)
    ref_root_height = env_states.robots[robot_name].extra["ref_root_height"]
    root_height_diff = root_height - ref_root_height  # [num_envs, 1]
    root_height_error = torch.mean(torch.abs(root_height_diff), dim=1)
    return torch.exp(-4 * root_height_error), root_height_error


class SquattingCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = "OnPolicyRunner"  # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        wandb = True
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 60  # per iteration
        max_iterations = 15001  # 3001  # number of policy updates

        # logging
        save_interval = 1000  # check for potential saves every this many iterations
        experiment_name = "squatting"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and ckpt


# TODO task config override robot config
class robot_asset(BaseConfig):
    fix_base_link: bool = False
    penalize_contacts_on = ["hip", "knee", "pelvis", "torso", "shoulder", "elbow"]


@configclass
class SquattingRewardCfg(RewardCfg):
    base_height_target = 0.89
    min_dist = 0.2
    max_dist = 0.5
    # put some settings here for LLM parameter tuning
    target_joint_pos_scale = 0.17  # rad
    target_feet_height = 0.06  # m
    cycle_time = 0.64  # sec
    # if true negative total rewards are clipped at zero (avoids early termination problems)
    only_positive_rewards = True
    # tracking reward = exp(error*sigma)
    tracking_sigma = 5
    max_contact_force = 700  # forces above this value are penalized


@configclass
class SquattingCfg(BaseHumanoidCfg):
    """Cfg class for Skillbench:Stepping."""

    task_name = "walking"
    sim_params = SimParamCfg(
        dt=0.001,
        contact_offset=0.01,
        substeps=1,
        num_position_iterations=4,
        num_velocity_iterations=0,
        bounce_threshold_velocity=0.1,
        replace_cylinder_with_capsule=False,
        friction_offset_threshold=0.04,
        num_threads=10,
    )

    ppo_cfg = SquattingCfgPPO()
    reward_cfg = SquattingRewardCfg()
    command_ranges = CommandRanges(lin_vel_x=[-0, 0], lin_vel_y=[-0, 0], ang_vel_yaw=[-0, 0], heading=[-0, 0])
    command_ranges.root_height_std = 0.2
    command_ranges.min_root_height = 0.2
    command_ranges.max_root_height = 1.1

    num_actions = 19
    command_dim = 1
    c_frame_stack = 3
    frame_stack = 1
    num_single_obs = 3 * num_actions + 6 + command_dim  #
    num_observations = int(frame_stack * num_single_obs)
    single_num_privileged_obs = 3 * num_actions + 18 + 3
    num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

    commands = CommandsConfig(num_commands=4, resampling_time=10.0)

    reward_functions: list[Callable] = [
        reward_squatting,
        reward_upper_body_pos,
        reward_orientation,
        reward_torques,
        reward_dof_vel,
        reward_dof_acc,
        reward_feet_distance,
        reward_default_joint_pos,
    ]

    # TODO: check why this configuration not work as well as the original one, that is probably a bug in infra.

    reward_weights: dict[str, float] = {
        "squatting": 5,
        "feet_distance": 0.5,
        "upper_body_pos": 0.5,
        "default_joint_pos": 0.5,
        "orientation": 1.0,
        "torques": -1e-5,
        "dof_vel": -5e-4,
        "dof_acc": -1e-7,
    }
