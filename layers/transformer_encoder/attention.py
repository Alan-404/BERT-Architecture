import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None):
    dk = torch.tensor((k.shape[-1])).type(torch.float32)

    attention_scores = torch.matmul(q, k.transpose(-2, -1))
    attention_scores = attention_scores/torch.sqrt(dk)

    if mask is not None:
        attention_scores += mask*(-1e20)

    attention_weights = F.softmax(attention_scores, dim=-1)

    output = torch.matmul(attention_weights, v)

    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, embedding_dim:int):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.embedding_dim = embedding_dim

        self.linear_k = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.linear_q = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.linear_v = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.linear_out = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def splitting_head(self, tensor: Tensor):
        """ 
            tensor: dim = (batch_size, length, embedding_dim)
        """

        heading_samples = self.embedding_dim//self.heads

        batch_size = tensor.shape[0]
        length = tensor.shape[1]

        tensor = torch.reshape(tensor, (batch_size, length, self.heads, heading_samples))
        print(tensor.shape)
        tensor_heads = torch.transpose(tensor, 1, 2) # dim = (batch_size, heads, length, heading_samples)

        return tensor_heads

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None):
        batch_size = q.shape[0]
        length = q.shape[1]

        qw = self.linear_q(q)
        kw = self.linear_k(k)
        vw = self.linear_v(v)

        heading_q = self.splitting_head(qw)
        heading_k = self.splitting_head(kw)
        heading_v = self.splitting_head(vw)

        attention_result, attention_weights = scaled_dot_product_attention(heading_q, heading_k, heading_v, mask)
        attention_result = torch.transpose(attention_result, 1, 2)
        attention_result = torch.reshape(attention_result, (batch_size, length, self.embedding_dim))

        output = self.linear_out(attention_result)

        return output, attention_weights


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim: int):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim

        self.linear_k = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.linear_q = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.linear_v = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.linear_out = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None):
        qw = self.linear_q(q)
        kw = self.linear_k(k)
        vw = self.linear_v(v)

        attention_result, attention_weights = scaled_dot_product_attention(qw, kw, vw, mask)

        output = self.linear_out(attention_result)

        return output, attention_weights