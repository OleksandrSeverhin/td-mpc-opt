import gymnasium as gym
import numpy as np
from envs.wrappers.time_limit import TimeLimit
import mani_skill.envs

# ManiSkill3 task configurations
MANISKILL3_TASKS = {
    'pick-cube': dict(
        env='PickCube-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'stack-cube': dict(
        env='StackCube-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'peg-insertion-side': dict(
        env='PegInsertionSide-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'push-cube': dict(
        env='PushCube-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'pull-cube': dict(
        env='PullCube-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'place-sphere': dict(
        env='PlaceSphere-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'lift-peg-upright': dict(
        env='LiftPegUpright-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'plug-charger': dict(
        env='PlugCharger-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'pick-single-ycb': dict(
        env='PickSingleYCB-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'roll-ball': dict(
        env='RollBall-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'poke-cube': dict(
        env='PokeCube-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'pull-cube-tool': dict(
        env='PullCubeTool-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'assembling-kits': dict(
        env='AssemblingKits-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'push-t': dict(
        env='PushT-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'two-robot-pick-cube': dict(
        env='TwoRobotPickCube-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'two-robot-stack-cube': dict(
        env='TwoRobotStackCube-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'pick-cube-so100': dict(
        env='PickCubeSO100-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'pick-cube-widow-xai': dict(
        env='PickCubeWidowXAI-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'turn-faucet': dict(
        env='TurnFaucet-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'open-cabinet-door': dict(
        env='OpenCabinetDoor-v1',
        control_mode='pd_ee_delta_pose',
    ),
    'open-cabinet-drawer': dict(
        env='OpenCabinetDrawer-v1',
        control_mode='pd_ee_delta_pose',
    ),
}

class ManiSkill3Wrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.observation_space = self.env.observation_space
        
        # Handle action space normalization for ManiSkill3
        if hasattr(self.env.action_space, 'low') and hasattr(self.env.action_space, 'high'):
            self.action_space = gym.spaces.Box(
                low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
                high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
                dtype=self.env.action_space.dtype,
            )
        else:
            self.action_space = self.env.action_space
    
    def reset(self, **kwargs):
        # ManiSkill3 returns obs, info from reset
        obs, info = self.env.reset(**kwargs)
        return obs, info
   
    def step(self, action):
        # Execute action for multiple timesteps as in original
        total_reward = 0
        for _ in range(2):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        # Return in ManiSkill3 format (obs, reward, terminated, truncated, info)
        return obs, total_reward, terminated, truncated, info
    
    @property
    def unwrapped(self):
        return self.env.unwrapped
    
    def render(self, *args, **kwargs):
        # ManiSkill3 uses different render modes
        return self.env.render()

def make_env(task, cfg):
    """Make environment, filtering out unsupported parameters for ManiSkill3."""
    
    # Get ManiSkill3 config
    maniskill3_cfg = getattr(cfg, 'maniskill3', {})
    env_kwargs = maniskill3_cfg.get('env_kwargs', {})
    
    # Only include supported parameters for ManiSkill3
    supported_kwargs = {
        'obs_mode': env_kwargs.get('obs_mode', 'state'),
        'control_mode': env_kwargs.get('control_mode', 'pd_ee_delta_pose'),
    }
    
    # Add optional parameters if they exist
    if 'render_mode' in env_kwargs:
        supported_kwargs['render_mode'] = env_kwargs['render_mode']
    if 'control_freq' in env_kwargs:
        supported_kwargs['control_freq'] = env_kwargs['control_freq']
    
    # Create environment with only supported parameters
    env = gym.make(task, **supported_kwargs)
    
    # Apply any additional wrappers here if needed
    
    return env

def make_multitask_env(cfg, task_id):
    """
    Make environment for a specific task in multi-task setting.
    """
    if not hasattr(cfg, 'tasks') or not cfg.tasks:
        raise ValueError('No tasks specified for multi-task environment')
    
    if task_id >= len(cfg.tasks):
        raise ValueError(f'Task ID {task_id} out of range. Available tasks: {len(cfg.tasks)}')
    
    # Create a copy of config for this specific task
    task_cfg = type(cfg)()
    for key, value in vars(cfg).items():
        setattr(task_cfg, key, value)
    
    # Set the specific task
    task_cfg.task = cfg.tasks[task_id].lower().replace('-', '_').replace('_v1', '').replace('_v0', '')
    
    return make_env(task_cfg)

def maniskill(task_id, cfg):
    """
    Main function for creating ManiSkill3 environments.
    Compatible with evaluation scripts.
    """
    # Map the task_id (which is actually the env name) to our format
    env_id = task_id
    
    # Create minimal kwargs that ManiSkill3 supports
    env_kwargs = {
        'obs_mode': 'state',
        'control_mode': 'pd_ee_delta_pose',
    }
    
    # Add render_mode if specified in config
    maniskill3_cfg = getattr(cfg, 'maniskill3', {})
    config_env_kwargs = maniskill3_cfg.get('env_kwargs', {})
    if 'render_mode' in config_env_kwargs:
        env_kwargs['render_mode'] = config_env_kwargs['render_mode']
    if 'control_freq' in config_env_kwargs:
        env_kwargs['control_freq'] = config_env_kwargs['control_freq']
    
    # Create environment directly (this matches what the evaluation script expects)
    try:
        env = gym.make(env_id, **env_kwargs)
    except Exception as e:
        raise ValueError(f'Failed to create environment {env_id}. Error: {e}')
    
    # Apply wrapper
    env = ManiSkill3Wrapper(env, cfg)
    
    # Apply time limit
    max_episode_steps = config_env_kwargs.get('max_episode_steps', 50)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env.max_episode_steps = env._max_episode_steps
    
    return env