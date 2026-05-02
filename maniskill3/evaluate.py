import os
import warnings

os.environ["MUJOCO_GL"] = "egl"
warnings.filterwarnings("ignore")

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_envs
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


@hydra.main(config_name="config", config_path=".")
def evaluate(cfg: dict):
    assert torch.cuda.is_available()
    assert cfg.eval_episodes_per_env > 0

    cfg = parse_cfg(cfg)
    if cfg.multitask:
        msg = "Multitask models not supported for maniskill."
        assert not cfg.multitask, colored(msg, "red", attrs=["bold"])

    set_seed(cfg.seed)
    print(colored(f"Task: {cfg.env_id}", "blue", attrs=["bold"]))

    model_sz = cfg.get("model_size", "default")
    print(colored(f"Model size: {model_sz}", "blue", attrs=["bold"]))
    
    chkpt_msg = f"Checkpoint: {cfg.checkpoint}"
    print(colored(chkpt_msg, "blue", attrs=["bold"]))

    env = make_envs(cfg, cfg.num_eval_envs, is_eval=True)

    agent = TDMPC2(cfg)
    if not os.path.exists(cfg.checkpoint):
        raise FileNotFoundError(f"Missing checkpoint: {cfg.checkpoint}")
    agent.load(cfg.checkpoint)

    if cfg.multitask:
        msg = f"Evaluating agent on {len(cfg.tasks)} tasks:"
    else:
        msg = f"Evaluating agent on {cfg.env_id}:"
    print(colored(msg, "yellow", attrs=["bold"]))

    if cfg.save_video_local:
        video_dir = os.path.join(cfg.work_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

    scores = []
    tasks = cfg.tasks if cfg.multitask else [cfg.env_id]
    device = "cuda" if cfg.env_type == "gpu" else "cpu"

    for task in tasks:
        has_succ = False
        has_fail = False
        ep_rewards = []
        ep_successes = []
        ep_fails = []
        video_idx = 0

        for _ in range(cfg.eval_episodes_per_env):
            obs, _ = env.reset()
            done = torch.zeros(
                cfg.num_eval_envs, dtype=torch.bool, device=device
            )
            ep_rew = torch.zeros(cfg.num_eval_envs, device=device)
            t = 0
            frames = []

            if cfg.save_video_local:
                frames.append(env.render().cpu())

            while not done[0]:
                action = agent.act(obs, t0=(t == 0), eval_mode=True)
                obs, reward, term, trunc, info = env.step(action)
                done = term | trunc
                ep_rew += reward
                t += 1

                if cfg.save_video_local:
                    frames.append(env.render().cpu())

            ep_rewards.append(ep_rew.mean().item())

            if "success" in info:
                has_succ = True
                succ_val = info["final_info"]["success"]
                ep_successes.append(succ_val.float().mean().item())

            if "fail" in info:
                has_fail = True
                fail_val = info["final_info"]["fail"]
                ep_fails.append(fail_val.float().mean().item())

            if cfg.save_video_local:
                videos = np.array(frames).transpose([1, 0, 2, 3, 4])
                for vid in videos:
                    vid_path = os.path.join(
                        video_dir, f"{task}-{video_idx}.mp4"
                    )
                    imageio.mimsave(vid_path, vid, fps=15)
                    video_idx += 1

        avg_rew = np.nanmean(ep_rewards)
        avg_succ = np.nanmean(ep_successes) if has_succ else 0.0
        avg_fail = np.nanmean(ep_fails) if has_fail else 0.0

        if cfg.multitask:
            if task.startswith("mw-"):
                scores.append(avg_succ * 100)
            else:
                scores.append(avg_rew / 10)

        log_str = f"  {task:<22}\tR: {avg_rew:.01f}  "
        if has_succ:
            log_str += f"\tS: {avg_succ:.02f}"
        if has_fail:
            log_str += f"\tF: {avg_fail:.02f}"
        print(colored(log_str, "yellow"))

    if cfg.multitask:
        mean_score = np.mean(scores)
        msg = f"Normalized score: {mean_score:.02f}"
        print(colored(msg, "yellow", attrs=["bold"]))


if __name__ == "__main__":
    evaluate()