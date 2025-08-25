import gym
import numpy as np

# Import ManiSkill3 and its utility wrappers
import mani_skill3.envs
from mani_skill3.utils.wrappers import ManiSkill_TimeLimitWrapper as TimeLimit

def make_env(cfg):
    """
    Creates a ManiSkill3 environment instance based on the provided configuration.

    This function is designed to be called by a main evaluation script that
    iterates through a list of tasks. It dynamically creates a single
    environment for the specified task and applies necessary wrappers.

    Args:
        cfg (dict): The configuration object loaded from the YAML file,
                    which contains the current task name.

    Returns:
        gym.Env: The created and wrapped ManiSkill3 environment.
    """
    control_mode = cfg.get("control_mode", "pd_ee_delta_pos")
    
    print(f"Making ManiSkill3 environment for task: {cfg.task} with control mode: {control_mode}")

    try:
        env = gym.make(
            cfg.task,
            obs_mode="state",
            control_mode=control_mode,
        )
        env = TimeLimit(env, max_episode_steps=cfg.episode_length)
        print("ManiSkill3 environment created successfully.")
        return env

    except Exception as e:
        print(f"Error creating environment: {e}")
        raise
