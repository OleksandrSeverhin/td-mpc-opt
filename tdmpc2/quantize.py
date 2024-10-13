import os
import time
from datetime import timedelta
os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')

import wandb
import hydra
import imageio
import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from omegaconf import OmegaConf
from tdmpc2 import TDMPC2


# config, config_multi_distill
# @hydra.main(config_name='config_mt30', config_path='./model1')
def quantize():

    # cfg = parse_cfg(cfg)
    # set_seed(cfg.seed)
    # env = make_env(cfg)

    # agent = TDMPC2(cfg)

    checkpoint = torch.load('/home/dmytrok/rl_exp/tdmpc2-opt/tdmpc2/logs/mt30/1/mt30_317M_800k_200Kpredistill_buffer50k_dist0_45_20240929/models/final.pt')
    state_dict = checkpoint['model']
    
    # FP16 Quantization
    for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor):
                if state_dict[key].dtype == torch.float32:
                    state_dict[key] = state_dict[key].half()

    quantized_state_dict = {"model": state_dict}
    torch.save(quantized_state_dict, 'mt30_1M_steps_045dist_mixed.pt')
    
    # INT8 Quantization
    def quantize_tensor(tensor):
        if tensor.dtype == torch.float32:
            return torch.quantize_per_tensor(tensor, scale=1.0, zero_point=0, dtype=torch.qint8)
        return tensor

    int8_state_dict = {k: quantize_tensor(v) if isinstance(v, torch.Tensor) else v 
                       for k, v in state_dict.items()}
    torch.save({"model": int8_state_dict}, 'mt30_1M_steps_045dist_int8.pt')

    # Mixed Precision Quantization
    def mixed_quantize_tensor(tensor, key):
        if tensor.dtype == torch.float32:
            if 'weight' in key or 'bias' in key:  # Keep weights and biases in FP16
                return tensor.half()
            else:  # Quantize other tensors to INT8
                return torch.quantize_per_tensor(tensor, scale=1.0, zero_point=0, dtype=torch.qint8)
        return tensor

    mixed_state_dict = {k: mixed_quantize_tensor(v, k) if isinstance(v, torch.Tensor) else v 
                        for k, v in state_dict.items()}
    torch.save({"model": mixed_state_dict}, 'mt30_1M_steps_045dist_mixed.pt')

    # Example usage
    # torch.save({"model": model_quantized.state_dict()}, 'mt30_1M_quantized.pt')
if __name__ == '__main__':
    quantize()