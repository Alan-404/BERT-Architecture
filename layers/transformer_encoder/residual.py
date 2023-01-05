import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ResidualConnection(nn.Module):
    def __init__(self, dropout_rate: float, epsilon: float):
        super(ResidualConnection, self).__init__()
        self.epsilon = epsilon
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, input: Tensor, output: Tensor, is_train: bool = True):
        if is_train:
            output = self.dropout_layer(output)
        
        
        residual = input + output

        return F.layer_norm(residual, normalized_shape=residual.shape)