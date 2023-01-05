import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Callable
from encoder_layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, num_layers: int = 6, heads: int = 8, embedding_dim: int = 512, hidden_units: int = 2048, dropout_rate: float = 0.1, epsilon: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]]= F.relu):
        super(Encoder, self).__init__()
        self.encoder_layers = [EncoderLayer(embedding_dim, heads, hidden_units, dropout_rate, epsilon, activation) for _ in range(num_layers)]

    def forward(self, tensor: Tensor, is_train: bool = True, mask: Tensor = None):
        output = Tensor(0).type(torch.float32)
        for encoder_layer in self.encoder_layers:
            output += encoder_layer(tensor, is_train, mask)

        return output