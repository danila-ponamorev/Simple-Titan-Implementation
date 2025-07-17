from typing import Optional

import torch
from torch import nn

import math

class MaskedMultiHeadAttention(nn.Module):
    """Класс MaskedMultiHeadAttention"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(MaskedMultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout_q = nn.Dropout(dropout)
        self.dropout_k = nn.Dropout(dropout)
        self.dropout_v = nn.Dropout(dropout)
        self.dropout_attention = nn.Dropout(dropout)  # Attention dropout
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Входной тензор, форма (batch_size, seq_len, d_model)
            mask (torch.Tensor): Маска для Attention scores, форма (seq_len, seq_len)
        """
        batch_size = x.shape[0]

        # 1. Projections with dropout
        Q = self.dropout_q(self.q(x))  # Dropout на Q
        K = self.dropout_k(self.k(x))  # Dropout на K
        V = self.dropout_v(self.v(x))  # Dropout на V

        # 2. Split into heads
        Q = Q.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 3. Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Attention dropout (optional)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout_attention(attention_weights)  # Dropout на весах

        # 5. Context computation
        output = torch.matmul(attention_weights, V)

        # 6. Combine heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.dropout_out(self.fc_out(output))

        return output
