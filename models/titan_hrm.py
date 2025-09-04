#написать реализацию HRM Titan сюда
import math

from torch import nn
from torch.func import functional_call
import torch

from .base_components import HRMInner, HRMBlock, HRMInnerCarry
from .neural_memory import NeuralMemoryAsContextLayerWithResidual, NeuralMemoryAsContextLayer
from .positional_encoding import PositionalEncoding

class DecoderOnlyMACTitanHRM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            n_heads: int,
            d_ff: int,
            memory_depth: int,
            H_depth: int,
            L_depth: int,
            H_cycles: int,
            L_cycles: int,
            memory_lr: float = 1e-5,
            dropout: float = 0.1,
            max_len: int = 512
    ):
        super().__init__()
        self.d_model = d_model
        self.window_size = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.inner = HRMInner(d_model, n_heads, d_ff, H_depth, L_depth, H_cycles, L_cycles, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.neural_memory = NeuralMemoryAsContextLayerWithResidual(d_model, d_ff, n_heads, memory_depth, memory_lr)

        self.norm_memory = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def mask_to_attention_mask(self, mask):
        """
        Конвертирует маску паддинга в маску для внимания.
        """
        batch_size, seq_len = mask.shape
        device = mask.device
        causal_mask = torch.tril(
            torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)).to(device)

        expanded_mask = mask.unsqueeze(1) & mask.unsqueeze(2)

        causal_mask = causal_mask & expanded_mask
        return causal_mask

    def store(self, src, memory_state, past_surprise, chunk_size: int = 256):
        """
        Корректирует веса нейронной памяти в соответствии с текущими входами и предыдущими состояниями

        Args:
            src (torch.Tensor): Входной тензор, форма (batch_size, seq_len)
            memory_state (dict): Словарь вида {name: torch.Tensor}.
            past_surprise (list): Список, вида [torch.Tensor]
            chunk_size (int): Число, размер чанка (разбиения)
        """
        batch_size, seq_len = src.shape
        device = src.device

        new_state, surprise = memory_state, past_surprise
        with torch.no_grad():
            token_embedding_weights_dict = {name: p.clone().detach().to(device) for name, p in self.token_embedding.named_parameters()}
            positional_encoding_weights_dict = {name: p.clone().detach().to(device) for name, p in self.positional_encoding.named_parameters()}
        for i in range(0, seq_len, chunk_size):
            end_index = min(i + chunk_size, seq_len)
            chunk = src[:, i:end_index].to(device)
            _, chunk_len = chunk.shape
            embedded_src = functional_call(self.token_embedding, token_embedding_weights_dict, (chunk,))
            src_pos = functional_call(self.positional_encoding, positional_encoding_weights_dict, (embedded_src,))
            new_state, surprise = self.neural_memory.store(src_pos, new_state, surprise)

        return new_state, surprise

    def forward(self, src, memory_state, past_surprise):
        """
        Args:
            src (torch.Tensor): Входной тензор, форма (batch_size, seq_len)
            memory_state (dict): Словарь вида {name: torch.tensor}.
        """

        batch_size, seq_len = src.shape
        device = src.device
        mask = src != 0 #заменить на pad_token

        src_mask = self.mask_to_attention_mask(mask)
        # 1. Embeddings + positional encodings
        embedded_src = self.token_embedding(src) * math.sqrt(self.d_model)

        src_pos = self.positional_encoding(embedded_src)

        mem_res, new_state, surprise = self.neural_memory(src_pos, memory_state, past_surprise, src_mask)

        z_L = mem_res
        z_H = torch.ones_like(z_L)
        carry = HRMInnerCarry(z_H, z_L)
        new_carry = self.inner(mem_res, carry, src_mask)

        output = self.fc_out(new_carry.z_H)

        return output, new_state, surprise




