import multiprocessing
import os
import warnings

os.environ["MUJOCO_GL"] = "egl"
os.environ["LAZY_LEGACY_OP"] = "0"
warnings.filterwarnings("ignore")

import hydra
import torch
from omegaconf import OmegaConf
from termcolor import colored

from common.buffer import Buffer
from common.logger import Logger, print_run
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_envs
from tdmpc2 import TDMPC2
from trainer.online_trainer import OnlineTrainer

torch.backends.cudnn.benchmark = True


@hydra.main(config_name="config", config_path=".", version_base=None)
def train(cfg: dict):
    assert torch.cuda.is_available()
    assert cfg.steps > 0

    cfg = parse_cfg(cfg)
    if cfg.multitask:
        msg = "Multitask models not supported for maniskill."
        assert not cfg.multitask, colored(msg, "red", attrs=["bold"])

    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)

    manager = multiprocessing.Manager()
    video_path = cfg.work_dir / "eval_video"

    if cfg.save_video_local:
        os.makedirs(video_path, exist_ok=True)

    logger = Logger(cfg, manager)
    env = make_envs(cfg, cfg.num_envs)
    eval_env = make_envs(
        cfg,
        cfg.num_eval_envs,
        video_path=video_path,
        is_eval=True,
        logger=logger
    )

    print_run(cfg)
    agent = TDMPC2(cfg)

    if logger._wandb is not None:
        logger._wandb.config.update(
            OmegaConf.to_container(cfg, resolve=True),
            allow_val_change=True
        )

    trainer = OnlineTrainer(
        cfg=cfg,
        env=env,
        eval_env=eval_env,
        agent=agent,
        buffer=Buffer(cfg),
        logger=logger,
    )
    trainer.train()
    print("\nTraining completed successfully")


if __name__ == "__main__":
    train()