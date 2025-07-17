import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    """
    Добавляет позиционные кодировки к входным эмбеддингам.
    Эта реализация соответствует формату (batch, seq_len, d_model).
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Тензор эмбеддингов, форма (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class AbsolutePositionalEncoding(nn.Module):
    """
    Генерирует и добавляет абсолютные позиционные кодировки.
    Принимает 'offset' для корректной работы с чанками.
    """
    def __init__(self, d_model: int, max_len: int = 20000, dropout: float = 0.1): # max_len для очень длинных контекстов
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Args:
            x: Тензор, форма (batch_size, seq_len, d_model)
            offset: Смещение для начала позиционного кодирования.
        """
        pos_encodings = self.pe[:, offset : offset + x.size(1)]
        return x + pos_encodings