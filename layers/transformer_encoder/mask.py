import torch
import numpy as np

def generate_padding_mask(tensor: torch.Tensor):
    return torch.Tensor(tensor == 0).type(torch.float32)[:, np.newaxis, np.newaxis, :]