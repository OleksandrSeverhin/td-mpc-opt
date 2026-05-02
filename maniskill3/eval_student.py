import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_envs
from tdmpc2 import TDMPC2


def configure_student_model(cfg):
    cfg.model_size = 1
    cfg.num_enc_layers = 2
    cfg.enc_dim = 256
    cfg.mlp_dim = 512
    cfg.latent_dim = 512
    return cfg


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


def run_episode(env, model):
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    done = False
    ep_reward = 0.0
    success = 0

    while not done:
        with torch.no_grad():
            action = model.act(obs, eval_mode=True)

        obs, reward, done, info = parse_step(env.step(action))
        ep_reward += get_scalar(reward)

        if isinstance(info, dict) and "success" in info:
            if get_scalar(info["success"]):
                success = 1

    return ep_reward, success


@hydra.main(config_name="config", config_path=".", version_base=None)
def evaluate_student(cfg: dict):
    assert torch.cuda.is_available()
    set_seed(cfg.seed)

    OmegaConf.set_struct(cfg, False)
    base_cfg = configure_student_model(parse_cfg(cfg))

    env = make_envs(base_cfg, 1, is_eval=True)
    model = TDMPC2(base_cfg)
    model.load(base_cfg.checkpoint)
    model.model.eval()

    ep_rewards = []
    ep_successes = []
    num_episodes = 100 

    for ep in range(num_episodes):
        reward, success = run_episode(env, model)
        ep_rewards.append(reward)
        ep_successes.append(success)

        print(
            f"Episode {ep + 1}/{num_episodes} | "
            f"Reward: {reward:.2f} | Success: {success}"
        )

    print("========================================")
    print(f"STUDENT BASELINE ({base_cfg.env_id})")
    print(f"Average Reward:  {np.mean(ep_rewards):.2f}")
    print(f"Success Rate:    {np.mean(ep_successes):.2f}")
    print("========================================")


if __name__ == "__main__":
    evaluate_student()