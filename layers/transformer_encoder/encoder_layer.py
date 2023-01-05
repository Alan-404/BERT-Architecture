import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Callable
from position_networks import PositionWiseFeedForwardNetworks
from attention import MultiHeadAttention
from residual import ResidualConnection

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int = 512, heads: int=8, hidden_units: int=2048, dropout_rate: float=0.1, epsilon: float=0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu):
        super(self, EncoderLayer).__init__()
        self.multi_head_attention = MultiHeadAttention(heads, embedding_dim)
        self.ffn = PositionWiseFeedForwardNetworks(hidden_units, embedding_dim)

        self.residual_connection_1 = ResidualConnection(dropout_rate, epsilon)
        self.residual_connection_2 = ResidualConnection(dropout_rate, epsilon)


    def forward(self, tensor: Tensor, is_train: bool = True, mask: Tensor= None):
        q, k, v = tensor

        multi_head_out, _ = self.multi_head_attention(q, k, v, mask)

        sub_1_output = self.residual_connection_1(tensor, multi_head_out, is_train)

        ffn_output = self.ffn(sub_1_output)

        sub_2_output = self.residual_connection_2(sub_1_output, ffn_output, is_train)

        return sub_2_output

