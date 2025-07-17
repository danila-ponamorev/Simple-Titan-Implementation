import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F
from typing import Dict, List

from .memory_models import MemoryMLP
from utils.helpers import normalize_grad

class NeuralMemoryFixedRhoAlpha(nn.Module):
    """
    Слой нейронной памяти.
    Важно: Этот модуль не является дифференцируемым сквозь основной граф вычислений.
    Он обновляет веса (передаваемые вручную) с помощью собственного правила
    обновления на основе градиентов от локальной MSE-потери.
    """

    def __init__(self, d_model: int, depth: int, aplha: float = 0.5, rho: float = 0.4,  learning_rate: float = 0.001):
        super().__init__()
        self.d_model = d_model
        self.learning_rate = learning_rate

        self.to_k = nn.Linear(d_model, d_model)
        self.to_q = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)

        self.alpha = aplha
        self.rho = rho

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

        prediction_v = functional_call(self.memory_model, memory_state, (k,))
        loss = F.mse_loss(prediction_v, v)

        with torch.enable_grad():
            grad_tuple = torch.autograd.grad(loss, tuple(memory_state.values()), retain_graph=True)
            grad_tuple = normalize_grad(grad_tuple, max_norm=1.0)

        with torch.no_grad():
            for i, (name, param) in enumerate(new_weights.items()):
                new_weights[name] = ((1 - self.alpha) * param + self.rho * past_surprise[i] - self.learning_rate * grad_tuple[i]).detach().clone()
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