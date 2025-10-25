from typing import Optional, List
from dataclasses import dataclass

from torch import nn
from torch.nn import functional as F
import torch

from .attention import MaskedMultiHeadAttention

@dataclass
class HRMInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor

#Лучше куда-нибудь в другое место
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps))

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

class SwiGLUMuchPeLu(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        activated = F.silu(self.w1(x)) * self.w2(x)
        return self.dropout(self.w3(activated))

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

class HRMBlock(nn.Module):
    """Тот же самый блок декодера трансформера, но с RMSNorm и немного другой свёрточной нейронной сетью"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MaskedMultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLUMuchPeLu(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, mask)
        x_after_attn = x + self.dropout(attn_out)
        x = x_after_attn + self.dropout(self.mlp(self.norm2(x_after_attn)))
        return x

class HRMModule(nn.Module):
    """Объединяет несколько HRMBlock в один модуль."""
    def __init__(self, layers: List[DecoderBlock], d_model):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, mask=None):
        hidden_states = hidden_states + input_injection

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)

        return hidden_states

#Пересмотреть метод forward (обработка рекурсивно, написать цикл для каждой из моделей),
class HRMInner(nn.Module):
    """Реализуют работу взаимодействия H_module и L_module"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, H_depth: int, L_depth: int, H_cycles: int, L_cycles: int, dropout: float = 0.1):
        super().__init__()
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        self.dropout = nn.Dropout(dropout)
        self.H_module = HRMModule([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(H_depth)], d_model)
        self.L_module = HRMModule([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(L_depth)], d_model)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, carry: HRMInnerCarry, mask=None):
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L
            for _H_step in range(self.H_cycles):
                for _L_step in range(self.L_cycles):
                    if not ((_H_step == self.H_cycles - 1) and (_L_step == self.L_cycles - 1)):
                        norm_injection = self.norm_1(z_H + x)
                        z_L = self.L_module(z_L, norm_injection, mask)
                if not (_H_step == self.H_cycles - 1):
                    # norm_injection = self.norm_2(z_L + x)
                    z_H = self.H_module(z_H, z_L, mask)

        assert not z_H.requires_grad and not z_L.requires_grad

        norm_injection = self.norm_1(z_H + x)
        z_L = self.L_module(z_L, norm_injection, mask)
        # norm_injection = self.norm_2(z_L + x)
        z_H = self.H_module(z_H, z_L, mask)

        new_carry = HRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        return new_carry