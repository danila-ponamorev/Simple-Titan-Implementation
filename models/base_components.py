from typing import Optional

from torch import nn
import torch

from .attention import MaskedMultiHeadAttention

class PositionwiseFeedForward(nn.Module):
    """
    Стандартный двухслойный Feed-Forward блок.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    """
    Стандартный блок декодера трансформера с pre-norm архитектурой.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MaskedMultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Входной тензор, форма (batch_size, seq_len, d_model)
            mask (torch.Tensor): Маска для attention layer, форма (seq_len, seq_len)
        """
        norm_x = self.norm1(x)
        attn_output = self.self_attn(norm_x, mask)
        attn_output = self.dropout_attn(attn_output)
        x_after_attn = x + attn_output

        norm_x_after_attn = self.norm2(x_after_attn)
        ff_output = self.feed_forward(norm_x_after_attn)
        ff_output = self.dropout_ff(ff_output)
        output = x_after_attn + ff_output

        return output
