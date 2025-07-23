import torch
from torch import nn
from .attention import MaskedMultiHeadAttentionForMemory
from .neural_memory import NeuralMemoryFixedRhoAlphaQKV
from .base_components import PositionwiseFeedForward
from torch.func import functional_call


class NeuralMemoryAsContextLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, depth: int, alpha: float = 0.1, rho: float = 0.01,  learning_rate: float = 0.001, dropout: float = 0.1):
        super().__init__()

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.attention = MaskedMultiHeadAttentionForMemory(d_model, n_heads, dropout)
        self._neural_memory = NeuralMemoryFixedRhoAlphaQKV(d_model, depth, alpha, rho, learning_rate)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def reset_memory_batch(self, batch_size):
        memory_states, past_surprises = self._neural_memory.reset_memory_batch(batch_size, self.q.weight.device)
        return memory_states, past_surprises

    def store(self, x:torch.Tensor, memory_states, past_surprises):

        K = self.k(x)
        V = self.v(x)

        memory_states, past_surprises = self._neural_memory.store_batch(memory_states, past_surprises, K, V)

        return memory_states, past_surprises

    def forward(self, x:torch.Tensor, memory_states, past_surprises, mask):

        Q = self.q(x)

        retrieve = self._neural_memory.retrieve_batch(x, memory_states, Q)

        x_with_memory = torch.cat([retrieve, x], dim=1)

        x_with_memory_norm = self.norm_1(x_with_memory)

        Q = self.q(x_with_memory_norm)
        K = self.k(x_with_memory_norm)
        V = self.v(x_with_memory_norm)

        attention_output = self.attention(x_with_memory, Q, K, V, mask)

        x_after_attn = attention_output + x_with_memory
        x_after_attn_norm = self.norm_2(x_after_attn)

        ff_output = self.feed_forward(x_after_attn_norm)
        output = x_after_attn + ff_output

        memory_states, past_surprises = self._neural_memory.store_batch(memory_states, past_surprises, K, V)

        return output, memory_states, past_surprises

