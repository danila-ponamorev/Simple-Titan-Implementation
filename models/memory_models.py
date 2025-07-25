import torch
from torch import nn
from torch.nn import functional as F

class PersistentMemory(nn.Module):
    """Хранилище обучаемых 'персистентных' токенов."""
    def __init__(self, num_tokens: int, d_model: int):
        super().__init__()
        self.persistent_tokens = nn.Parameter(torch.randn(num_tokens, d_model))
        nn.init.xavier_uniform_(self.persistent_tokens)

    def forward(self) -> torch.Tensor:
        return self.persistent_tokens


class MemoryMLP(nn.Module):
    """Простая MLP, используемая как ядро нейронной памяти."""
    def __init__(self, dim: int, depth: int = 2):
        super().__init__()
        self.depth = depth
        self.dim = dim

        self.weights = nn.ParameterList([nn.Parameter(torch.ones(dim, dim) / 10.0) for _ in range(depth)])
        # for weight in self.weights:
        #     nn.init.xavier_uniform_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Входной тензор, форма (batch_size, seq_len, d_model)
        """
        for i, weight in enumerate(self.weights):
            x = x @ weight
            if i > 0:
                x = F.gelu(x)
        return x