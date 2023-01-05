import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Callable

class PositionWiseFeedForwardNetworks(nn.Module):
    def __init__(self, hidden_units: int, embedding_dim: int, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu):
        super(PositionWiseFeedForwardNetworks, self).__init__()
        
        self.hidden_layer = nn.Linear(in_features=embedding_dim, out_features=hidden_units)
        self.activation = activation
        self.output_layer = nn.Linear(in_features=hidden_units, out_features=embedding_dim)

    def forward(self, x: Tensor):
        x = self.hidden_layer(x)
        x = self.activation(x)
        return self.output_layer(x)