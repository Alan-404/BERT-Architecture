import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Callable
from layers.transformer_encoder.encoder_block import Encoder
from layers.classification import ClassificationLayer

class BERTModel(nn.Module):
    def __init__(self, token_size: int, num_layers_encoder: int = 12, embedding_dim: int = 768, attention_heads: int = 12, hidden_units: int = 2048, dropout_rate: float = 0.1, epsilon: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu):
        super(BERTModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers_encoder, heads=attention_heads, embedding_dim=embedding_dim, hidden_units=hidden_units, dropout_rate=dropout_rate, epsilon=epsilon, activation=activation)
        self.classfication_layer = ClassificationLayer(embedding_dim=embedding_dim)

        self.output_layer = nn.Linear(in_features=embedding_dim, out_features=token_size)
        
    def forward(self, tensor:Tensor, is_train: bool, mask: Tensor = None):
        tensor = self.encoder(tensor, is_train, mask)
        tensor = self.classfication_layer(tensor)
        tensor = self.output_layer(tensor)
        return tensor
