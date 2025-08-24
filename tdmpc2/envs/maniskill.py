import gym
import numpy as np
from mani_skill.utils.wrappers import ManiSkillWrapper, ContinuousTaskWrapper
import mani_skill.envs as ms_envs
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.envs import make as make_ms_env

# This wrapper simplifies the observation space and action space for TD-MPC
# It handles the multi-step control frequency to align with the agent's control loop.
class TDMPCWrapper(gym.Wrapper):
    def __init__(self, env, obs_mode, control_mode):
        super().__init__(env)
        self.env = env
        self.observation_space = self.env.observation_space[obs_mode]
        self.action_space = self.env.action_space[control_mode]
        
    def reset(self, **kwargs):
        # The ManiSkill reset returns a dictionary of observations. We extract the 'state' obs.
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        # ManiSkill step returns a dictionary of observations. We extract the 'state' obs.
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        return obs_dict, reward, terminated, truncated, info

def make_env(env_id, obs_mode="state", control_mode="pd_ee_delta_pose", record_dir=None, render_mode="rgb_array"):
    """
    This function creates a ManiSkill3 environment with the specified configuration.
    It is designed to be called by your main training or evaluation scripts.

    Args:
        env_id (str): The name of the ManiSkill3 task (e.g., "PickCube-v1").
        obs_mode (str): The observation mode. 'state' is used for latent space models.
        control_mode (str): The robot control mode.
        record_dir (str): Directory to save recorded episodes.
        render_mode (str): Rendering mode for the environment.
    """
    
    # 1. Create the base ManiSkill3 environment
    env = make_ms_env(
        env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
    )
    
    # 2. Wrap the environment with the TDMPCWrapper to simplify spaces
    env = TDMPCWrapper(env, obs_mode, control_mode)
    
    # 3. Add a wrapper to record episodes, if a directory is provided
    if record_dir is not None:
        print(f"Recording episodes to {record_dir}")
        env = RecordEpisode(env, record_dir)
        
    return env