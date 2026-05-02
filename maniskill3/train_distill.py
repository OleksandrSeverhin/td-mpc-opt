import multiprocessing

import hydra
import torch
from omegaconf import OmegaConf
from termcolor import colored

from common.buffer import Buffer
from common.logger import Logger
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_envs
from tdmpc2 import TDMPC2
from trainer.online_trainer import OnlineTrainer

torch.backends.cudnn.benchmark = True


def build_student_cfg(base_cfg, env):
    cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    cfg_dict.update({
        "obs_shape": {"state": [obs_dim]},
        "action_dim": act_dim,
        "action_dims": [act_dim]
    })
    return OmegaConf.create(cfg_dict)


def build_teacher_cfg(student_cfg):
    cfg_dict = OmegaConf.to_container(student_cfg, resolve=True)
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


@hydra.main(config_name="config", config_path=".", version_base=None)
def train(cfg: dict):
    assert torch.cuda.is_available()
    set_seed(cfg.seed)

    OmegaConf.set_struct(cfg, False)
    base_cfg = parse_cfg(cfg)

    print(colored("Initializing Environment...", "yellow"))
    env = make_envs(base_cfg, base_cfg.num_envs)

    print(colored("Configuring Student (1M)...", "green"))
    student_cfg = build_student_cfg(base_cfg, env)
    teacher_cfg = build_teacher_cfg(student_cfg)

    print(colored("Loading Teacher Weights...", "blue"))
    teacher_model = TDMPC2(teacher_cfg)
    teacher_model.load(base_cfg.checkpoint, strict=True)
    teacher_model.model.eval()

    for p in teacher_model.model.parameters():
        p.requires_grad = False

    obs_dim = env.observation_space.shape[-1]
    msg = f"Initializing Student with {obs_dim} inputs..."
    print(colored(msg, "green"))

    agent = TDMPC2(student_cfg, teacher=teacher_model)
    eval_env = make_envs(
        student_cfg, student_cfg.num_eval_envs, is_eval=True
    )
    logger = Logger(student_cfg, multiprocessing.Manager())

    trainer = OnlineTrainer(
        cfg=student_cfg,
        env=env,
        eval_env=eval_env,
        agent=agent,
        buffer=Buffer(student_cfg),
        logger=logger,
    )
    trainer.train()


if __name__ == "__main__":
    train()