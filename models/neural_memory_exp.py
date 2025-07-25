import time

import torch
from torch import nn
from .base_components import PositionwiseFeedForward
from .memory_models import MemoryMLP
import math
from torch.func import vmap, functional_call, grad
from torch.nn import functional as F
from utils.helpers import normalize_grad, normalize_grad_fast, debug_tensor

class TimeCheck:
    def __init__(self):
        self.cur_time = time.time()
    def time(self):
        print(time.time() - self.cur_time)
        self.cur_time = time.time()


class FastNeuralMemoryAsContextLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, depth: int, alpha: float = 0.1, rho: float = 0.01,  learning_rate: float = 0.001, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.memory_depth = depth

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

        self._memory_model = MemoryMLP(d_model, depth)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.alpha = alpha
        self.rho = rho
        self.memory_lr = learning_rate

        self.batch_retrieve = vmap(
            lambda state, q: functional_call(self._memory_model, state, (q,)),
        )

        self.grad_one = vmap(grad(self._forward_one, argnums=0),  # argnums=0 → grad w.r.t. state_dict
                        in_dims=({f'weights.{i}': 0 for i in range(self.memory_depth)}, 0, 0))


    def store(self, x: torch.Tensor, memory_state, past_surprise):
        with torch.no_grad():
            Q_for_retrieve = self.q(x)
            retrieved = self.batch_retrieve(memory_state, Q_for_retrieve)
            x_with_memory = torch.cat([retrieved, x], dim=1)
            # 1. Projections with dropout
            K = self.k(x_with_memory)
            V = self.v(x_with_memory)
        with torch.no_grad():
            grads = self.grad_one(memory_state, K, V)
            grads = normalize_grad_fast(grads, max_norm=1.0)

            new_state = {
                n: (1 - self.alpha) * p + self.rho * s - self.memory_lr * g
                for (n, p), s, g in zip(memory_state.items(), past_surprise.values(), grads.values())
            }
            surprise = {
                str(i): surprise
                for i, surprise in enumerate(grads.values())
            }

        return new_state, surprise



    def new_states_for_batch(self, batch_size, device):
        parameters_dict = {name: torch.ones((batch_size,) + p.shape).to(device)/10.0 for name, p in self._memory_model.named_parameters()}
        surprises_list = {str(i): torch.zeros((batch_size,) + p.shape).to(device) for i, p in enumerate(self._memory_model.parameters())}

        return parameters_dict, surprises_list

    def forward(self, x: torch.Tensor, memory_state, past_surprise, mask):
        batch_size = x.shape[0]
        
        # 0. Retrieve context from memory
        with torch.no_grad():
            Q_for_retrieve = self.q(x)
            retrieved = self.batch_retrieve(memory_state, Q_for_retrieve)
        
        x_with_memory = torch.cat([retrieved, x], dim=1)

        # 1. Projections with dropout
        Q = self.q(x_with_memory)
        K = self.k(x_with_memory)
        V = self.v(x_with_memory)

        with torch.no_grad():
            grads = self.grad_one(memory_state, K, V)
            grads = normalize_grad_fast(grads, max_norm=1.0)

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
        
        # 5. Context computation
        attn_res = torch.matmul(attention_weights, V)
        
        # 6. Combine heads
        attn_res = attn_res.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        attn_res = self.fc_out(attn_res)
        

        # 7. NeuralMemory update
        # new_state, surprise = batch_store(memory_state, past_surprise, K.contiguous().view(batch_size, -1, self.d_model), attn_res, self.alpha, self.rho, self.memory_lr)
        new_state = {
            n: (1 - self.alpha) * p + self.rho * s - self.memory_lr * g
            for (n, p), s, g in zip(memory_state.items(), past_surprise.values(), grads.values())
        }
        surprise = {
            str(i): surprise
            for i, surprise in enumerate(grads.values())
        }
        
        # 8. DecoderBlock Logic
        attn_res = self.norm_1(attn_res)
        x_after_attn = x_with_memory + attn_res
        
        norm_x_after_attn = self.norm_2(x_after_attn)
        ff_output = self.feed_forward(norm_x_after_attn)
        output = x_after_attn + ff_output

        return output, new_state, surprise

    def _forward_one(self, state_dict, k, v):
        pred = functional_call(self._memory_model, state_dict, (k,))
        loss = F.mse_loss(pred, v)
        return loss
