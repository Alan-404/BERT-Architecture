import torch 
from torch import Tensor
import numpy as np

class PositionEmbedding:
    def encode_position(self, length: int, embedding_dim: int):
        pos = np.arange(length) # dim = (length, )
        pos = np.expand_dims(pos, axis=1) # dim = (length, 1)

        angles = np.arange(embedding_dim) # dim = (embedding_dim, )
        angles = np.expand_dims(angles, axis=0) # dim = (1, embedding_dim)
        angles[0::2] = angles[1::2]
        angles = 1/(10000*(angles/embedding_dim))

        pos_angles = np.dot(pos, angles) # dim = (length, embedding_dim)
        pos_angles[0::2] = np.sin(pos_angles[0::2])
        pos_angles[1::2] = np.cos(pos_angles[1::2])
        
        pos_angles = np.expand_dims(pos_angles, axis=0) # dim = (1, length, embedding_dim)

        return Tensor(pos_angles).type(torch.float32)

