import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ClassificationLayer(nn.Module):
    def __init__(self, embedding_dim: int):
        super(ClassificationLayer, self).__init__()
        self.linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, tensor: Tensor):
        tensor = self.linear(tensor)
        tensor = F.gelu(tensor)
        return F.layer_norm(tensor, normalized_shape=tensor.shape)