import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import OmegaConf

from common.parser import parse_cfg
from common.seed import set_seed
from tdmpc2 import TDMPC2


class OfflineBuffer:
    def __init__(self, data_dir: str, cfg):
        self.cfg = cfg
        all_obs, all_act, all_rew = [], [], []

        for filename in os.listdir(data_dir):
            if not filename.endswith(".pt"):
                continue

            filepath = os.path.join(data_dir, filename)
            ep = torch.load(filepath, weights_only=False)

            if ep["obs"].shape[-1] != 24:
                continue

            obs = ep["obs"]
            act = ep["action"]
            rew = ep["reward"]
            length = obs.shape[0]

            if length < 6:
                pad = obs[-1:].repeat(6 - length, 1)
                obs = torch.cat([obs, pad])
            if act.shape[0] < 5:
                pad = act[-1:].repeat(5 - act.shape[0], 1)
                act = torch.cat([act, pad])
            if rew.shape[0] < 5:
                pad = rew[-1:].repeat(5 - rew.shape[0], 1)
                rew = torch.cat([rew, pad])

            pad_length = obs.shape[0]
            for i in range(pad_length - 5):
                all_obs.append(obs[i : i + 6])
                all_act.append(act[i : i + 5])
                all_rew.append(rew[i : i + 5])

        self.obs_data = torch.stack(all_obs).to("cuda")
        self.act_data = torch.stack(all_act).to("cuda")
        self.rew_data = torch.stack(all_rew).to("cuda")
        self.n_samples = self.obs_data.shape[0]

        self.task_out = torch.zeros(
            self.cfg.batch_size, dtype=torch.long, device="cuda"
        )

    def sample(self):
        idxs = torch.randint(
            0, self.n_samples, (self.cfg.batch_size,), device="cuda"
        )

        obs_out = self.obs_data[idxs].transpose(0, 1).contiguous()
        act_out = self.act_data[idxs].transpose(0, 1).contiguous()
        rew_out = self.rew_data[idxs].transpose(0, 1).contiguous()

        return obs_out, act_out, rew_out, self.task_out


def build_teacher_cfg(cfg):
    teacher_cfg = cfg.copy()
    teacher_cfg.model_size = 317
    teacher_cfg.multitask = True
    teacher_cfg.tasks = [str(i) for i in range(30)]
    teacher_cfg.action_dims = [6] * 30
    teacher_cfg.episode_lengths = [cfg.episode_length] * 30
    teacher_cfg.num_enc_layers = 5
    teacher_cfg.enc_dim = 4096
    teacher_cfg.mlp_dim = 4096
    teacher_cfg.latent_dim = 1376
    teacher_cfg.num_q = 8
    teacher_cfg.task_dim = 96
    return teacher_cfg


def build_student_cfg(cfg):
    student_cfg = cfg.copy()
    student_cfg.model_size = 1
    student_cfg.num_enc_layers = 2
    student_cfg.enc_dim = 256
    student_cfg.mlp_dim = 512
    student_cfg.latent_dim = 512
    return student_cfg


@hydra.main(config_name="config", config_path=".", version_base=None)
def train_offline(cfg: dict):
    assert torch.cuda.is_available()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    OmegaConf.set_struct(cfg, False)
    cfg.episode_length = 100
    cfg.horizon = 5

    cfg = parse_cfg(cfg)
    cfg.action_dim = 6
    cfg.obs_shape = {"state": [24]}

    set_seed(cfg.seed)

    wandb.init(
        project="tdmpc2_offline_distillation",
        name=cfg.exp_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    dataset_path = getattr(cfg, "dataset_path", None)
    if dataset_path is None:
        raise ValueError("A dataset_path must be provided.")

    buffer = OfflineBuffer(dataset_path, cfg)

    teacher_cfg = build_teacher_cfg(cfg)
    teacher_agent = TDMPC2(teacher_cfg)
    ckpt_path = "/root/td-mpc-opt/tdmpc2/models/multitask/mt30-317M.pt"

    teacher_agent.load(ckpt_path)
    teacher_agent.model.eval()
    for param in teacher_agent.model.parameters():
        param.requires_grad = False

    student_cfg = build_student_cfg(cfg)
    student_agent = TDMPC2(student_cfg, teacher_model=teacher_agent)
    student_agent._get_horizon = lambda *args, **kwargs: 5

    save_dir = Path(f"logs/{cfg.exp_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    for step in range(cfg.steps):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            stats = student_agent.update(buffer, step=step)

        if step > 0 and step % 10000 == 0:
            wandb.log(stats, step=step)

        if step > 0 and step % 50000 == 0:
            intermediate_path = save_dir / f"model_{step}.pt"
            student_agent.save(str(intermediate_path))

    save_path = save_dir / "model.pt"
    student_agent.save(str(save_path))
    wandb.finish()


if __name__ == "__main__":
    train_offline()