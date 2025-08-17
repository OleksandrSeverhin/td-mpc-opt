import os
import torch
import hydra
import gymnasium as gym
from envs.maniskill import maniskill  
from omegaconf import DictConfig, OmegaConf
from tdmpc2 import TDMPC2


def make_env(task_name, cfg):
    """Create a ManiSkill3 environment for a given task."""
    env = gym.make(
        task_name,
        obs_mode=cfg.maniskill3.env_kwargs.obs_mode,
        control_mode=cfg.maniskill3.env_kwargs.control_mode,
        render_mode=cfg.maniskill3.env_kwargs.render_mode,
        sim_freq=cfg.maniskill3.env_kwargs.sim_freq,
        control_freq=cfg.maniskill3.env_kwargs.control_freq,
        max_episode_steps=cfg.maniskill3.env_kwargs.max_episode_steps,
    )
    return env


def populate_spaces(cfg):
    """
    Populate obs_shape, action_dim, and episode_length in cfg
    using the first ManiSkill3 environment.
    """
    tasks = cfg.tasks if cfg.get("multitask", False) else [cfg.task]
    example_task = tasks[0]

    env = make_env(example_task, cfg)
    obs_space = env.observation_space
    act_space = env.action_space

    # Flatten obs space if Dict
    if hasattr(obs_space, "spaces"):
        obs_shape = sum(
            int(torch.tensor(space.shape).numel())
            for space in obs_space.spaces.values()
        )
    else:
        obs_shape = int(torch.tensor(obs_space.shape).numel())

    action_dim = act_space.shape[0]
    episode_length = env.spec.max_episode_steps

    # Save to config
    cfg.obs_shape = obs_shape
    cfg.action_dim = action_dim
    cfg.episode_length = episode_length

    if cfg.get("multitask", False):
        cfg.obs_shapes = {t: obs_shape for t in tasks}
        cfg.action_dims = {t: action_dim for t in tasks}
        cfg.episode_lengths = {t: episode_length for t in tasks}

    env.close()
    return cfg


def evaluate(cfg: DictConfig):
    print("Original config:")
    print(OmegaConf.to_yaml(cfg))

    # Auto-populate obs/action dims
    cfg = populate_spaces(cfg)

    print("Updated config with obs/action shapes:")
    print(OmegaConf.to_yaml(cfg))

    # Load agent
    agent = TDMPC2(cfg)

    tasks = cfg.tasks if cfg.get("multitask", False) else [cfg.task]

    for task in tasks:
        env = make_env(task, cfg)
        returns = []
        for ep in range(cfg.eval_episodes):
            obs, _ = env.reset()
            done, total_reward, steps = False, 0, 0
            while not done:
                action = agent.act(obs, eval_mode=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
            returns.append(total_reward)
            print(f"[{task}] Episode {ep+1}: return={total_reward:.2f}, steps={steps}")
        avg_return = sum(returns) / len(returns)
        print(f"[{task}] Average return over {cfg.eval_episodes} episodes: {avg_return:.2f}")
        env.close()


@hydra.main(config_path="./student_config", config_name="config_maniskill3")
def main(cfg: DictConfig):
    evaluate(cfg)


if __name__ == "__main__":
    main()
