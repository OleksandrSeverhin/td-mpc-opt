import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_envs
from tdmpc2 import TDMPC2


def configure_teacher(base_cfg):
    cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)
    cfg_dict.update({
        "model_size": 317,
        "num_enc_layers": 5,
        "enc_dim": 4096,
        "mlp_dim": 4096,
        "num_q": 8,
        "latent_dim": 1376,
        "true_latent_dim": 1376,
        "obs_shape": {"state": [24]},
        "action_dim": 6,
        "action_dims": [6] * 30,
        "multitask": True,
        "task_dim": 96,
        "tasks": [str(i) for i in range(30)]
    })
    return OmegaConf.create(cfg_dict)


def parse_step(step_out):
    if len(step_out) == 4:
        obs, reward, done, info = step_out
    else:
        obs, reward, term, trunc, info = step_out
        if isinstance(term, torch.Tensor):
            done = term.any() or trunc.any()
        else:
            done = term or trunc
    return obs, reward, done, info


def get_scalar(value):
    return value[0].item() if hasattr(value, "item") else value


def run_episode(env, model, action_shape):
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    done = False
    ep_reward = 0.0
    success = 0

    while not done:
        t_obs = obs[:, :24]
        t_task = torch.zeros(
            t_obs.shape[0], dtype=torch.long, device=model.device
        )

        with torch.no_grad():
            t_act = model.act(t_obs, eval_mode=True, task=t_task)

        action = torch.zeros((1, action_shape), dtype=torch.float32)
        action[:, :6] = t_act

        obs, reward, done, info = parse_step(env.step(action))
        ep_reward += get_scalar(reward)

        if isinstance(info, dict) and "success" in info:
            if get_scalar(info["success"]):
                success = 1

    return ep_reward, success


@hydra.main(config_name="config", config_path=".", version_base=None)
def evaluate(cfg: dict):
    assert torch.cuda.is_available()
    set_seed(cfg.seed)

    OmegaConf.set_struct(cfg, False)
    base_cfg = parse_cfg(cfg)

    print(colored("Initializing Environment...", "yellow"))
    env = make_envs(base_cfg, 1, is_eval=True)

    print(colored("Configuring Teacher Architecture...", "green"))
    teacher_cfg = configure_teacher(base_cfg)

    print(colored("Loading Teacher Weights...", "blue"))
    model = TDMPC2(teacher_cfg)
    model.load(base_cfg.checkpoint, strict=True)
    model.model.eval()

    print(colored(f"Running Baseline on {base_cfg.env_id}...", "cyan"))
    ep_rewards = []
    ep_successes = []
    num_episodes = 5000
    action_shape = env.action_space.shape[-1]

    for ep in range(num_episodes):
        reward, success = run_episode(env, model, action_shape)
        ep_rewards.append(reward)
        ep_successes.append(success)

        print(
            f"Episode {ep + 1} | Reward: {reward:.2f} | "
            f"Success: {success}"
        )

    print(colored("========================================", "magenta"))
    print(colored(f"TEACHER BASELINE ({base_cfg.env_id})", "magenta"))
    print(colored(f"Average Reward:  {np.mean(ep_rewards):.2f}", "magenta"))
    print(colored(f"Success Rate:    {np.mean(ep_successes):.2f}", "magenta"))
    print(colored("========================================", "magenta"))


if __name__ == "__main__":
    evaluate()