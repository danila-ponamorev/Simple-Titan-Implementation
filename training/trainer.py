from typing import List, Dict, Optional, Callable, Any, Tuple
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from abc import ABC
from .callbacks import Callback

class BaseTrainer(ABC):
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: Optimizer,
            loss_fn: Callable,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            device: str = "cuda",
            grad_accumulation_steps: int = 1,
            callbacks: Optional[List[Callback]] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.grad_accumulation_steps = grad_accumulation_steps
        self.callbacks = callbacks or []

        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False
        self.metrics = {}
        self._log_history = {
            'train': [],
            'val': [],
            'custom': []
        }
        self._current_logs = {}
        self._log_buffer = {}

    def _run_callbacks(self, hook_name, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, hook_name):
                method = getattr(callback, hook_name)
                method(self, **kwargs)

    def log(self, metric_name: str, value: float, phase: str = 'train'):
        """Логирование метрик.

        Args:
            metric_name: Название метрики (например, 'loss')
            value: Значение метрики
            phase: Фаза ('train', 'val' или 'custom')
        """
        if phase not in self._log_history:
            raise ValueError(f"Неизвестная фаза логирования: {phase}. Допустимо: 'train', 'val', 'custom'")

        self._current_logs[metric_name] = value

        self._log_history[phase].append({
            'step': self.global_step,
            'epoch': self.current_epoch,
            metric_name: value
        })

        if metric_name not in self._log_buffer:
            self._log_buffer[metric_name] = []
        self._log_buffer[metric_name].append(value)

    def get_logs(self, phase: str = 'train', last_n: int = None) -> list:
        """Возвращает историю логирования."""
        logs = self._log_history.get(phase, [])
        return logs[-last_n:] if last_n else logs

    def _flush_logs(self):
        """Очищает текущие логи после шага обучения/валидации."""
        self._current_logs = {}
        self._log_buffer = {}

    def train(self, num_epochs: int):
        self._run_callbacks('on_train_start', num_epochs=num_epochs)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self._run_callbacks('on_epoch_start', epoch=epoch)

            self.model.train()
            for batch_idx, batch in enumerate(self.train_loader):
                self._handle_iteration(batch, batch_idx)
                if self.should_stop:
                    break
            if self.val_loader:
                self.validate()

            self._run_callbacks('on_epoch_end')
            if self.should_stop:
                break

        self._run_callbacks('on_train_end')

    def _handle_iteration(self, batch: Dict, batch_idx: int):
        self._run_callbacks('on_iteration_start')

        total_loss = 0
        self.optimizer.zero_grad()

        for accumulation_step in range(self.grad_accumulation_steps):
            micro_batch = self._slice_batch(batch, accumulation_step)
            self._run_callbacks('on_batch_start', batch=micro_batch)

            loss = self._forward_pass(micro_batch)["loss"]
            loss.backward()

            total_loss += loss.item()
            self._run_callbacks(
            'on_batch_end',
                    batch=micro_batch,
                    loss=loss,
                    batch_idx=batch_idx,
                    global_step=self.global_step,
            )

            self.global_step += 1

        if (batch_idx + 1) % self.grad_accumulation_steps == 0:
            self._run_callbacks('on_before_optimizer_step')
            self.optimizer.step()
            self._run_callbacks('on_after_optimizer_step')
            self._run_callbacks('on_iteration_end', batch_idx=batch_idx)

        return total_loss / self.grad_accumulation_steps

    def validate(self):
        self.model.eval()
        self._run_callbacks('on_validation_start')

        val_metrics = {}
        total_loss = 0
        val_len = len(self.val_loader)
        for batch_idx, batch in enumerate(self.val_loader):
            self._run_callbacks('on_validation_batch_start', batch=batch)
            with torch.no_grad():
                outputs = self._forward_pass(batch)

            val_metrics = self._update_metrics(val_metrics, outputs)

            total_loss += outputs['loss'].item()
            self._run_callbacks(
                'on_validation_batch_end',
                        batch=batch,
                        outputs=outputs,
                        batch_idx=batch_idx,
                        loss=outputs['loss'],
            )
        avg_loss = total_loss / val_len
        self.metrics.update({'val_loss': avg_loss})
        self._run_callbacks('on_validation_end', loss=avg_loss)

    def _forward_pass(self, batch) -> Dict:
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)

        return {
            "loss": loss,
            "outputs": outputs,
            "targets": targets,
        }

    def _update_metrics(self, metrics_dict: Dict, outputs: Dict):
        pass

    def _slice_batch(self, batch, accumulation_step):
        """Разделение batch'а для gradient accumulation."""
        return batch


class DecoderOnlyMACTitanBaseTrainer(BaseTrainer):
    def __init__(self, padding_token: int, chunk_size: int = 16, **kwargs):
        self.padding_token = padding_token
        self.chunk_size = chunk_size
        super().__init__(**kwargs)

    def _forward_pass(self, batch):
        """Выполняет forward pass с выравниванием входных последовательностей и обновлением памяти.

        Args:
            batch: Кортеж (inputs, targets), где:
                inputs - тензор/список тензоров с входными последовательностями
                targets - тензор с целевыми значениями

        Returns:
            Словарь с:
                loss - значение функции потерь
                outputs - выходы модели
                targets - целевые значения
        """
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        prepared_inputs = self._prepare_sequences(inputs, max_len=self.model.window_size, pad_token=self.padding_token)
        prepared_targets = self._prepare_sequences(targets, max_len=self.model.window_size,pad_token=self.padding_token)

        batch_size = inputs.size(0)

        memory_state, past_surprise = self.model.neural_memory.new_states_for_batch(batch_size, self.device)
        memory_inputs = self._prepare_sequences_for_memory(inputs, inputs.shape[1], 16)
        memory_state, past_surprise = self.model.store(
                memory_inputs,
                memory_state,
                past_surprise,
                chunk_size=self.chunk_size
            )

        outputs, _, _ = self.model(prepared_inputs, memory_state, past_surprise)

        # generated = torch.Tensor([torch.argmax(outputs[0, i, :].view(-1)) for i in range(outputs.size(1))])

        outputs = outputs.view(-1, outputs.size(-1))
        prepared_targets = prepared_targets.view(-1)

        loss = self.loss_fn(outputs, prepared_targets)

        return {
            "loss": loss,
            "outputs": outputs,
            "targets": prepared_targets,
            "aligned_inputs": inputs
        }

    def _prepare_sequences(self, tokens, max_len, pad_token=0):
        """
        Альтернативная версия с использованием torch.narrow
        """
        batch_size, seq_len = tokens.shape

        is_pad = (tokens == pad_token)
        seq_lengths = is_pad.int().argmax(dim=1)
        no_pad_mask = (seq_lengths == 0) & (tokens[:, 0] != pad_token)
        seq_lengths[no_pad_mask] = seq_len

        start_indices = torch.clamp(seq_lengths - max_len, min=0)

        prepared_sequences = torch.full((batch_size, max_len), pad_token,
                                        dtype=tokens.dtype, device=tokens.device)

        for i in range(batch_size):
            start_idx = start_indices[i].item()
            actual_length = min(seq_lengths[i].item() - start_idx, max_len)

            if actual_length > 0:
                prepared_sequences[i, :actual_length] = tokens[i, start_idx:start_idx + actual_length]

        return prepared_sequences

    def _prepare_sequences_for_memory(self, tokens, max_len, chunk_size, pad_token=0):
        """
        Переписать для более эффективного использования.
        """
        batch_size, seq_len = tokens.shape

        is_pad = (tokens == pad_token)
        seq_lengths = is_pad.int().argmax(dim=1)
        no_pad_mask = (seq_lengths == 0) & (tokens[:, 0] != pad_token)
        seq_lengths[no_pad_mask] = seq_len

        start_indices = torch.clamp(seq_lengths - seq_len, min=0)

        prepared_sequences = torch.full((batch_size, max_len), pad_token,
                                        dtype=tokens.dtype, device=tokens.device)

        for i in range(batch_size):
            start_idx = start_indices[i].item()
            actual_length = min(seq_lengths[i].item() - start_idx, max_len)

            if actual_length > 0:
                actual_length = actual_length - actual_length % chunk_size
                prepared_sequences[i, :actual_length] = tokens[i, start_idx:start_idx + actual_length]

        return prepared_sequences

    def _align_sequences(self, sequences, pad_value=0, dim=0):
        """Выравнивает последовательности добавлением pad_value в начало.

        Args:
            sequences: Список тензоров формы [seq_len] или [batch, seq_len]
            pad_value: Значение для padding
            dim: Измерение для выравнивания

        Returns:
            Выровненный тензор формы [batch, max_seq_len]
        """
        max_len = max(s.size(dim) for s in sequences)

        aligned = []

        for seq in sequences:
            pad_size = max_len - seq.size(dim)

            if pad_size > 0:
                pad_shape = list(seq.shape)
                pad_shape[dim] = pad_size
                padding = torch.full(
                    pad_shape,
                    pad_value,
                    dtype=seq.dtype,
                    device=seq.device
                )

                aligned_seq = torch.cat([seq, padding], dim=dim)
            else:
                aligned_seq = seq

            aligned.append(aligned_seq)

        return torch.stack(aligned)

