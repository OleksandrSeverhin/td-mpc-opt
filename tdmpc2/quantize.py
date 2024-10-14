import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

def quantize():

    checkpoint = torch.load('mt30_317M_1M_dist0_45_20240929/models/final.pt')
    state_dict = checkpoint['model']
    
    # FP16 Quantization
    for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor):
                if state_dict[key].dtype == torch.float32:
                    state_dict[key] = state_dict[key].half()

    quantized_state_dict = {"model": state_dict}
    torch.save(quantized_state_dict, 'mt30_1M_steps_045dist_fp16.pt')
    
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
            if 'weight' in key or 'bias' in key: 
                return tensor.half()
            else: 
                return torch.quantize_per_tensor(tensor, scale=1.0, zero_point=0, dtype=torch.qint8)
        return tensor

    mixed_state_dict = {k: mixed_quantize_tensor(v, k) if isinstance(v, torch.Tensor) else v 
                        for k, v in state_dict.items()}
    torch.save({"model": mixed_state_dict}, 'mt30_1M_steps_045dist_mixed.pt')

    # Example usage
    # torch.save({"model": model_quantized.state_dict()}, 'mt30_1M_quantized.pt')
if __name__ == '__main__':
    quantize()