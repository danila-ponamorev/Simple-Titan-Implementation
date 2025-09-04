from typing import Dict, List

import torch
from torch import nn
import math
from torch.func import vmap, functional_call, grad
from torch.nn import functional as F

from .base_components import PositionwiseFeedForward
from .memory_models import MemoryMLP
from utils.helpers import normalize_grad, normalize_grad_fast, debug_tensor

class NeuralMemoryFixedRhoAlpha(nn.Module):
    """
    Слой нейронной памяти.
    Важно: Этот модуль не является дифференцируемым сквозь основной граф вычислений.
    Он обновляет веса (передаваемые вручную) с помощью собственного правила
    обновления на основе градиентов от локальной MSE-потери.
    """

    def __init__(self, d_model: int, depth: int, alpha: float = 0.1, rho: float = 0.01,  learning_rate: float = 0.001):
        super().__init__()
        self.d_model = d_model
        self.learning_rate = learning_rate

        self.to_k = nn.Linear(d_model, d_model)
        self.to_q = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)

        self.alpha = alpha
        self.rho = rho

        self.memory_model = MemoryMLP(d_model, depth)

    def store(self, x: torch.Tensor, memory_state: Dict[str, torch.Tensor], past_surprise: List[torch.Tensor]):
        # Включаем градиенты для входных данных и параметров памяти
        x.requires_grad_(True)
        for p in memory_state.values():
            p.requires_grad_(True)

        # Forward pass
        k = self.to_k(x)
        v = self.to_v(x)
        prediction_v = functional_call(self.memory_model, memory_state, (k,))
        loss = F.mse_loss(prediction_v, v)

        # Вычисляем градиенты
        with torch.enable_grad():
            # Градиенты для memory_state
            grad_tuple = torch.autograd.grad(
                loss,
                tuple(memory_state.values()),
                create_graph=False,
                retain_graph=True
            )
            grad_tuple = normalize_grad(grad_tuple, max_norm=1.0)

        # Обновляем веса памяти
        new_weights = {}
        for i, (name, param) in enumerate(memory_state.items()):
            new_weights[name] = (
                    (1 - self.alpha) * param
                    + self.rho * past_surprise[i]
                    - self.learning_rate * grad_tuple[i]
            ).clamp_(-100.0, 100.0)

        # Обновляем past_surprise
        for i, surprise_grad in enumerate(grad_tuple):
            past_surprise[i] = surprise_grad.clamp(-100.0, 100.0).clone()

        return new_weights, past_surprise

    def retrieve(self, x: torch.Tensor, memory_state) -> torch.Tensor:
        """
        Извлекает информацию из памяти для входного тензора x.
        Args:
            x (torch.Tensor): Входной тензор, форма (seq_len, d_model).
            memory_state (dict): Словарь вида {name: torch.Tensor}
        """
        x.requires_grad_(True)
        q = self.to_q(x)
        # print(q.requires_grad)
        with torch.no_grad():
            output = functional_call(self.memory_model, memory_state, (q,))
        return output

    def reset_memory(self):
        """
        Сбрасывает состояние памяти (веса-буферы и 'удивление')
        к их изначальным значениям.
        """
        clean_memory_model = MemoryMLP(self.d_model, self.memory_model.depth).to(self.to_k.weight.device)

        clean_memory_model_params_dict = {name: torch.ones_like(p) / 10.0 for name, p in clean_memory_model.named_parameters()}
        clean_memory_model_surprise = [torch.zeros_like(p) for p in clean_memory_model.parameters()]

        return clean_memory_model_params_dict, clean_memory_model_surprise

    def store_batch(self, x, memory_states, past_surprises):
        """
        Реализует метод store для батча.
        Args:
            x (torch.Tensor): Входной тензор, форма (batch_size, seq_len, d_model)
            memory_states (list): список состояний памяти вида [{name: torch.Tensor}]
            past_surprises (list): список предыдущих удивлений вида [[torch.Tensor]]
        """
        for i, memory_state in enumerate(memory_states):
            memory_states[i], past_surprises[i] = self.store(x[i, :, :], memory_states[i], past_surprises[i])
        return memory_states, past_surprises

    def retrieve_batch(self, x, memory_states):
        """
        Реализует метод retrieve для батча.
        Args:
            x (torch.Tensor): Входной тензор, форма (batch_size, seq_len, d_model)
            memory_states (list): Список состояний памяти вида [{name: torch.Tensor}]
        """
        output = x.clone().detach()

        for i, memory_state in enumerate(memory_states):
            output[i, :, :] = self.retrieve(x[i, :, :], memory_states[i])

        return output

    def reset_memory_batch(self, batch_size: int):
        """
        Реализует метод reset_memory для батча.
        Args:
            batch_size (int): размер батча
        """

        surprises_list = []
        memory_states_list = []

        for i in range(batch_size):
            new_memory_state, new_surprise = self.reset_memory()
            surprises_list.append(new_surprise)
            memory_states_list.append(new_memory_state)

        return memory_states_list, surprises_list


class NeuralMemory(nn.Module):
    """
    Слой нейронной памяти.
    Важно: Этот модуль не является дифференцируемым сквозь основной граф вычислений.
    Он обновляет свои веса (хранящиеся в буферах) с помощью собственного правила
    обновления на основе градиентов от локальной MSE-потери.
    """

    def __init__(self, d_model: int, depth: int, learning_rate: float = 0.001):
        super().__init__()
        self.d_model = d_model
        self.learning_rate = learning_rate

        self.to_k = nn.Linear(d_model, d_model)
        self.to_q = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)

        self.decay_control = nn.Linear(d_model, 1)
        self.past_surprise_control = nn.Linear(d_model, 1)

        self.memory_model = MemoryMLP(d_model, depth)

    def store(self, x: torch.Tensor, memory_state: Dict[str, torch.Tensor], past_surprise: List[torch.Tensor]):
        """
        Обновляет веса памяти.
        Args:
            x (torch.Tensor): Входной тензор, форма (seq_len, d_model), где N - кол-во токенов для обновления.
            memory_state (dict): Словарь вида {name: torch.Tensor}
        """
        new_weights = {name: p.detach().clone() for name, p in memory_state.items()}

        for p in memory_state.values():
            p.requires_grad_(True)

        k = self.to_k(x)
        v = self.to_v(x)

        alpha = torch.sigmoid(self.decay_control(x)).mean()
        rho = torch.sigmoid(self.past_surprise_control(x)).mean()

        prediction_v = functional_call(self.memory_model, memory_state, (k,))
        loss = F.mse_loss(prediction_v, v)

        with torch.enable_grad():
            grad_tuple = torch.autograd.grad(loss, tuple(memory_state.values()), retain_graph=True)
            grad_tuple = normalize_grad(grad_tuple, max_norm=1.0)

        with torch.no_grad():
            for i, (name, param) in enumerate(new_weights.items()):
                new_weights[name] = ((1 - alpha) * param + rho * past_surprise[i] - self.learning_rate * grad_tuple[
                    i]).detach().clone()
                torch.clamp_(new_weights[name], min=-100.0, max=100.0)

            for i, surprise_grad in enumerate(grad_tuple):
                clamped_surprise = torch.clamp(surprise_grad, min=-100.0, max=100.0)
                past_surprise[i] = clamped_surprise.detach().clone()

            return new_weights, past_surprise

    def retrieve(self, x: torch.Tensor, memory_state) -> torch.Tensor:
        """
        Извлекает информацию из памяти для входного тензора x.
        Args:
            x (torch.Tensor): Входной тензор, форма (seq_len, d_model).
            memory_state (dict): Словарь вида {name: torch.Tensor}
        """
        q = self.to_q(x)
        with torch.no_grad():
            output = functional_call(self.memory_model, memory_state, (q,))
            return output

    def reset_memory(self):
        """
        Сбрасывает состояние памяти (веса-буферы и 'удивление')
        к их изначальным значениям.
        """
        clean_memory_model = MemoryMLP(self.d_model, self.memory_model.depth).to(self.to_k.weight.device)

        clean_memory_model_params_dict = {name: torch.ones_like(p) / 10.0 for name, p in
                                          clean_memory_model.named_parameters()}
        clean_memory_model_surprise = [torch.zeros_like(p) for p in clean_memory_model.parameters()]

        return clean_memory_model_params_dict, clean_memory_model_surprise

    def store_batch(self, x, memory_states, past_surprises):
        """
        Реализует метод store для батча.
        Args:
            x (torch.Tensor): Входной тензор, форма (batch_size, seq_len, d_model)
            memory_states (list): список состояний памяти вида [{name: torch.Tensor}]
            past_surprises (list): список предыдущих удивлений вида [[torch.Tensor]]
        """
        for i, memory_state in enumerate(memory_states):
            memory_states[i], past_surprises[i] = self.store(x[i, :, :], memory_states[i], past_surprises[i])
        return memory_states, past_surprises

    def retrieve_batch(self, x, memory_states):
        """
        Реализует метод retrieve для батча.
        Args:
            x (torch.Tensor): Входной тензор, форма (batch_size, seq_len, d_model)
            memory_states (list): Список состояний памяти вида [{name: torch.Tensor}]
        """
        output = x.clone().detach()

        for i, memory_state in enumerate(memory_states):
            output[i, :, :] = self.retrieve(x[i, :, :], memory_states[i])

        return output

    def reset_memory_batch(self, batch_size: int):
        """
        Реализует метод reset_memory для батча.
        Args:
            batch_size (int): размер батча
        """
        surprises_list = []
        memory_states_list = []

        for i in range(batch_size):
            new_memory_state, new_surprise = self.reset_memory()
            surprises_list.append(new_surprise)
            memory_states_list.append(new_memory_state)

        return memory_states_list, surprises_list


class NeuralMemoryAsContextLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, depth: int, alpha: float = 0.1, rho: float = 0.01,
                 learning_rate: float = 0.001, dropout: float = 0.1):
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

        self.dropout = nn.Dropout(dropout)

    def store(self, x: torch.Tensor, memory_state, past_surprise):
        with torch.no_grad():
            Q_for_retrieve = self.q(x)
            retrieved = self.batch_retrieve(memory_state, Q_for_retrieve)
            x_with_memory = torch.cat([retrieved, x], dim=1)
            # 1. Projections with dropout

            K = self.k(x_with_memory)
            V = self.v(x_with_memory)

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
        parameters_dict = {name: torch.ones((batch_size,) + p.shape).to(device) / 10.0 for name, p in
                           self._memory_model.named_parameters()}
        surprises_list = {str(i): torch.zeros((batch_size,) + p.shape).to(device) for i, p in
                          enumerate(self._memory_model.parameters())}

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
        attn_res = self.dropout(attn_res)

        # 7. NeuralMemory update
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
        ff_output = self.dropout(ff_output)
        output = x_after_attn + ff_output

        return output, new_state, surprise

    def _forward_one(self, state_dict, k, v):
        pred = functional_call(self._memory_model, state_dict, (k,))
        loss = F.mse_loss(pred, v)
        return loss


class NeuralMemoryAsContextLayerWithResidual(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, depth: int, alpha: float = 1e-3, rho: float = 5e-4,
                 learning_rate: float = 2e-5, dropout: float = 0.1):
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
        self.x_with_retrieved_norm = nn.LayerNorm(d_model)
        self.K_norm = nn.LayerNorm(d_model)

        self.alpha = alpha
        self.rho = rho
        self.memory_lr = learning_rate

        self.batch_retrieve = vmap(
            lambda state, q: functional_call(self._memory_model, state, (q,)),
        )

        self.grad_one = vmap(grad(self._forward_one, argnums=0),  # argnums=0 → grad w.r.t. state_dict
                             in_dims=({f'weights.{i}': 0 for i in range(self.memory_depth)}, 0, 0))

        self.dropout = nn.Dropout(dropout)

    def _prepare_attention_mask(self, mask, n_heads):
        """
        Подготавливает маску для MultiHeadAttention.

        Args:
            mask: Tensor формы (batch_size, seq_len) или (batch_size, seq_len, seq_len)
            n_heads: количество голов внимания

        Returns:
            mask: Tensor формы (batch_size, n_heads, seq_len, seq_len)
        """
        if mask.dim() == 2:
            # (B, seq_len) -> (B, 1, 1, seq_len) -> (B, n_heads, seq_len, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(-1, n_heads, -1, -1)
        elif mask.dim() == 3:
            # (B, seq_len, seq_len) -> (B, 1, seq_len, seq_len) -> (B, n_heads, seq_len, seq_len)
            mask = mask.unsqueeze(1)
            mask = mask.expand(-1, n_heads, -1, -1)

        return mask

    def store(self, x: torch.Tensor, memory_state, past_surprise):
        with torch.no_grad():
            x_norm = self.norm_1(x)

            retrieved = self.batch_retrieve(memory_state, x_norm)

            x_with_memory = x + retrieved
            x_with_memory = self.x_with_retrieved_norm(x_with_memory)

            # 1. Projections with dropout
            K = self.k(x_with_memory)
            V = self.v(x_with_memory)

            K_norm = self.K_norm(K)

            memory_output = self.batch_retrieve(memory_state, K_norm)
            grads = self.grad_one(memory_state, memory_output, V)
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
        parameters_dict = {name: torch.ones((batch_size,) + p.shape).to(device) / 10.0 for name, p in
                           self._memory_model.named_parameters()}
        surprises_list = {str(i): torch.zeros((batch_size,) + p.shape).to(device) for i, p in
                          enumerate(self._memory_model.parameters())}

        return parameters_dict, surprises_list

    def forward(self, x: torch.Tensor, memory_state, past_surprise, mask=None):
        batch_size = x.shape[0]

        # 0. Retrieve context from memory
        with torch.no_grad():
            x_norm = self.norm_1(x)
            retrieved = self.batch_retrieve(memory_state, x_norm)
            x_with_memory = x + retrieved
            x_with_memory = self.x_with_retrieved_norm(x_with_memory)

        # 1. Projections with dropout
        Q = self.q(x_with_memory)
        K = self.k(x_with_memory)
        V = self.v(x_with_memory)

        # 2. Split into heads
        Q = Q.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 3. Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            # print(mask.shape, scores.shape)
            if mask.shape != scores.shape:
                mask = self._prepare_attention_mask(mask, self.n_heads)
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Attention dropout (optional)
        attention_weights = torch.softmax(scores, dim=-1)

        # 5. Context computation
        attn_res = torch.matmul(attention_weights, V)

        # 6. Combine heads
        attn_res = attn_res.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # attn_res = self.norm_2(attn_res)
        attn_res = self.fc_out(attn_res)
        attn_res = self.dropout(attn_res)

        # 8. DecoderBlock Logic
        x_after_attn = x_with_memory + attn_res

        norm_x_after_attn = self.norm_2(x_after_attn)
        ff_output = self.feed_forward(norm_x_after_attn)
        ff_output = self.dropout(ff_output)
        output = x_after_attn + ff_output

        return output, memory_state, past_surprise

    def _forward_one(self, state_dict, k, v):
        pred = functional_call(self._memory_model, state_dict, (k,))
        loss = F.mse_loss(pred, v)
        return loss
