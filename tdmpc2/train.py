import os
import warnings

os.environ["MUJOCO_GL"] = "egl"
os.environ["LAZY_LEGACY_OP"] = "0"
warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict.tensordict import TensorDict
from termcolor import colored

from common.buffer import Buffer
from common.logger import Logger
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.distill_trainer import DistillTrainer


TASKS = [
    "walker-stand", "walker-walk", "walker-run", "cheetah-run", "reacher-easy",
    "reacher-hard", "acrobot-swingup", "pendulum-swingup", "cartpole-balance",
    "cartpole-balance-sparse", "cartpole-swingup", "cartpole-swingup-sparse",
    "cup-catch", "finger-spin", "finger-turn-easy", "finger-turn-hard",
    "fish-swim", "hopper-stand", "hopper-hop", "walker-walk-backwards",
    "walker-run-backwards", "cheetah-run-backwards", "cheetah-run-front",
    "cheetah-run-back", "cheetah-jump", "hopper-hop-backwards",
    "reacher-three-easy", "reacher-three-hard", "cup-spin", "pendulum-spin"
]


def _format_tensor(tensor: torch.Tensor | None, default_shape: tuple, default_val: float) -> torch.Tensor:
    if tensor is not None:
        return tensor.unsqueeze(0).cpu() if tensor.dim() == 1 else tensor.cpu()
    return torch.full(default_shape, default_val)


def to_td(
    obs: torch.Tensor,
    action: torch.Tensor | None = None,
    reward: float | None = None,
    done: bool | None = None,
    task: str | int | None = None
) -> TensorDict:
    task_idx = TASKS.index(task) if isinstance(task, str) else (task if task is not None else 0)

    obs_tensor = _format_tensor(obs, (1,), float("nan"))
    action_tensor = _format_tensor(action, (1, 6), float("nan"))

    reward_tensor = torch.tensor([reward if reward is not None else float("nan")], dtype=torch.float32)
    done_tensor = torch.tensor([done if done is not None else False], dtype=torch.bool)
    task_tensor = torch.tensor([task_idx], dtype=torch.long)

    return TensorDict({
        "obs": obs_tensor,
        "action": action_tensor,
        "reward": reward_tensor,
        "done": done_tensor,
        "task": task_tensor
    }, batch_size=(1,))


def _load_teacher(cfg: DictConfig) -> TDMPC2:
    teacher_cfg = parse_cfg(cfg.teacher_config)
    set_seed(teacher_cfg.seed)
    
    teacher_model = TDMPC2(teacher_cfg)
    teacher_model.load(teacher_cfg.checkpoint)
    teacher_model.model.eval()
    
    return teacher_model


def _setup_student(cfg: DictConfig, teacher_model: TDMPC2) -> tuple:
    student_cfg = parse_cfg(cfg.student_config)
    student_cfg.is_moe_student = True
    set_seed(student_cfg.seed)

    env = make_env(student_cfg)
    buffer = Buffer(student_cfg)
    model = TDMPC2(student_cfg, teacher_model=teacher_model)
    logger = Logger(student_cfg)
    
    return student_cfg, env, buffer, model, logger


def _get_current_task_idx(env) -> int:
    raw_task = getattr(env, "task", 0)
    return TASKS.index(raw_task) if isinstance(raw_task, str) else raw_task


def _process_done_flag(done: bool | list | tuple | np.ndarray) -> bool:
    if isinstance(done, (list, tuple, np.ndarray)):
        return any(done)
    return done


def _run_training_loop(
    cfg: DictConfig,
    env,
    buffer: Buffer,
    student_model: TDMPC2,
    trainer: DistillTrainer,
    logger: Logger
) -> None:
    obs, done, t = env.reset(), False, 0
    curr_task_idx = _get_current_task_idx(env)
    tds = [to_td(obs, task=curr_task_idx)]
    seed_steps = cfg.batch_size * 2

    print(colored("Starting training loop...", "green"))
    
    for step in range(cfg.steps):
        if step < seed_steps:
            action = torch.tensor(env.action_space.sample(), dtype=torch.float32)
        else:
            with torch.no_grad():
                task_tensor = torch.tensor([curr_task_idx], device="cuda", dtype=torch.long)
                action = student_model.act(obs, t0=(t == 0), eval_mode=False, task=task_tensor)

        next_obs, reward, raw_done, _ = env.step(action)
        done = _process_done_flag(raw_done)

        tds.append(to_td(next_obs, action, reward, done, curr_task_idx))
        
        obs = next_obs
        t += 1

        if done:
            buffer.add(torch.cat(tds))
            obs, done, t = env.reset(), False, 0
            curr_task_idx = _get_current_task_idx(env)
            tds = [to_td(obs, task=curr_task_idx)]

        if step >= seed_steps and buffer._num_eps > 0:
            try:
                metrics = trainer.update(buffer)
                if step % 1000 == 0:
                    print(f"Step {step} | Total Loss: {metrics['total_loss']:.4f}")
                    logger.log(metrics)
            except Exception as e:
                print(colored(f"Training failed at step {step}: {e}", "red"))
                break


@hydra.main(config_name="generic", config_path=".")
def train(cfg: DictConfig) -> None:
    assert torch.cuda.is_available(), "CUDA is not available."
    OmegaConf.set_struct(cfg, False)

    teacher_model = _load_teacher(cfg)
    student_cfg, env, buffer, student_model, logger = _setup_student(cfg, teacher_model)

    print(colored("Work dir:", "yellow", attrs=["bold"]), student_cfg.work_dir)

    teachers_dict = {i: teacher_model for i in range(len(TASKS))}
    trainer = DistillTrainer(
        student=student_model,
        teachers=teachers_dict,
        alpha=student_cfg.latent_distill_alpha
    )

    _run_training_loop(student_cfg, env, buffer, student_model, trainer, logger)
    print(colored("Process complete.", "cyan"))


if __name__ == "__main__":
    train()
