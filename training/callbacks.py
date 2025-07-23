from abc import ABC #, abstractmethod
import torch
import time
from typing import List


"""
Проверить на корректность и совместимость с BaseTrainer
Написать логику для hook'ов DecoderOnlyMACTitan
"""

class Callback(ABC):
    """
    Абстрактный базовый класс для всех callback'ов.
    """
    # @abstractmethod
    def on_train_start(self, trainer, **kwargs):
        """Вызывается перед началом обучения."""
        pass

    def on_epoch_start(self, trainer, **kwargs):
        """Вызывется в начале каждой эпохи."""
        pass

    def on_iteration_start(self, trainer, **kwargs):
        """Вызывается перед началом каждой итерации (перед первым batch'ем)."""
        pass

    def on_batch_start(self, trainer, **kwargs):
        """Вызывается перед обработкой batch'а."""
        pass

    def on_batch_end(self, trainer, **kwargs):
        """Вызывается после обработки каждого batch'а."""
        pass

    def on_before_optimizer_step(self, trainer, **kwargs):
        """Вызывается до шага оптимизатора"""
        pass

    def on_after_optimizer_step(self, trainer, **kwargs):
        """Вызывается после шага оптимизатора"""
        pass

    def on_iteration_end(self, trainer, **kwargs):
        """Вызывается после завершения итерации (после optimizer.step())."""
        pass

    def on_validation_start(self, trainer, **kwargs):
        """Вызывается перед валидацией."""
        pass

    def on_validation_batch_start(self, trainer, **kwargs):
        """Вызывается перед batch'ем валидации"""
        pass

    def on_validation_batch_end(self, trainer, **kwargs):
        """Вызывается после batch'а валидации"""
        pass

    def on_validation_end(self, trainer, **kwargs):
        """Вызывается после валидации."""
        pass

    def on_epoch_end(self, trainer, **kwargs):
        """Вызывается в конце каждой эпохи."""
        pass

    def on_train_end(self, trainer, **kwargs):
        """Вызывается после обучения."""
        pass


class GradientClipping(Callback):
    """Градиентный клиппинг."""
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm

    def on_before_optimizer_step(self, trainer, **kwargs):
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(),
            self.max_norm
        )


class EarlyStopping(Callback):
    """Ранняя остановка при отсутствии улучшений."""
    def __init__(self, monitor='val_loss', patience=5, min_delta=0.01):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float('inf')

    def on_epoch_end(self, trainer, **kwargs):
        current = trainer.metrics[self.monitor]
        if current < self.best_value - self.min_delta:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.should_stop = True


class ModelCheckpoint(Callback):
    """Сохранение лучших весов."""
    def __init__(self, filepath, monitor='val_loss'):
        self.filepath = filepath
        self.monitor = monitor
        self.best_value = float('inf')

    def on_epoch_end(self, trainer, **kwargs):
        torch.save(trainer.model.state_dict(), self.filepath)
        current = trainer.metrics.get(self.monitor)
        if current < self.best_value:
            self.best_value = current
            torch.save(trainer.model.state_dict(), f"{self.filepath.replace('.pt', '')}_best.pt")


class LRScheduler(Callback):
    """Обертка для стандартных LR шедулеров."""
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self, trainer, **kwargs):
        self.scheduler.step()


class WarmupScheduler(Callback):
    """Линейный warmup learning rate."""
    def __init__(self, warmup_steps: int = 1000):
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def on_batch_start(self, trainer, **kwargs):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = trainer.initial_lr * (self.current_step / self.warmup_steps)
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = lr




class MetricsLogger(Callback):
    """Логирование метрик в консоль."""
    def __init__(self):
        super().__init__()
    def on_epoch_end(self, trainer, **kwargs):
        print(f"Epoch {trainer.current_epoch}:")
        for k, v in trainer.metrics.items():
            print(f"  {k}: {v:.4f}")


class MetricLogger(Callback):
    def __init__(self):
        super().__init__()

    def on_batch_end(self, trainer, **kwargs):
        loss = kwargs.get('loss')
        # batch = kwargs.get('batch')

        trainer.log('batch_loss', loss.item())
        # print(f"loss: {loss.item():.4f}")
    def on_validation_batch_end(self, trainer, **kwargs):

        loss = kwargs.get('loss')

        trainer.log('val_batch_loss', loss.item())

    def on_epoch_end(self, trainer, **kwargs):
        avg_loss = torch.mean(torch.tensor(trainer._log_buffer['batch_loss']))
        trainer.log('epoch_loss', avg_loss, phase='val')
        # print(f"loss: {avg_loss:.4f}")
        trainer._log_buffer['batch_loss'] = []

    def on_validation_end(self, trainer, **kwargs):
        avg_loss = torch.mean(torch.tensor(trainer._log_buffer['val_batch_loss']))
        trainer.log('val_loss', avg_loss, phase='val')
        # print(f"loss: {avg_loss:.4f}")
        trainer._log_buffer['val_batch_loss'] = []


class LogPrinter(Callback):
    def __init__(self, log_interval: int = 10):
        super().__init__()
        self.log_interval = log_interval
        self.global_start_time = None
        self.last_time = None
        self.start_time = None
        self.num_epochs = None
        self.current_epoch = None

    def on_train_start(self, trainer, **kwargs):
        num_epochs = kwargs.get('num_epochs')
        self.global_start_time = time.time()
        self.num_epochs = num_epochs

    def on_epoch_start(self, trainer, **kwargs):
        epoch = kwargs.get('epoch')
        self.current_epoch = epoch
        self.start_time = time.time()

    def on_batch_start(self, trainer, **kwargs):
        if self.last_time is None:
            self.last_time = time.time()

    def on_batch_end(self, trainer, **kwargs):
        batch_idx = kwargs.get('batch_idx')
        global_step = kwargs.get('global_step')

        if (batch_idx + 1) % self.log_interval == 0:
            print(f"Epoch: {self.current_epoch}/{self.num_epochs},\n"
                  f" Global Step: {global_step + 1}/{len(trainer.train_loader) * self.num_epochs},\n"
                  f" Batch: {batch_idx + 1}/{len(trainer.train_loader)},\n"
                  f" Loss: {trainer._log_buffer['batch_loss'][-1]:.4f},\n"
                  f" Time: {time.time() - self.last_time:.2f} sec,\n"
                  f" Overall Time: {(time.time() - self.global_start_time)/60:.2f} min")
            self.last_time = time.time()


    def on_epoch_end(self, trainer, **kwargs):
        print(f"Epoch Ended!\n"
              f"Epoch: {self.current_epoch + 1}/{self.num_epochs},\n"
              f" Loss: {trainer._log_buffer['epoch_loss'][-1]:.4f},\n"
              f" Time: {(time.time() - self.start_time)/60:.2f} min")
        self.last_time = time.time()

    def on_validation_start(self, trainer, **kwargs):
        self.start_time = time.time()

    def on_validation_batch_end(self, trainer, **kwargs):
        batch_idx = kwargs.get('batch_idx')

        if (batch_idx + 1) % self.log_interval == 0:
            val_loader_len = 0
            if trainer.val_loader:
                val_loader_len = len(trainer.val_loader)
            print(f"Val Batch: {batch_idx + 1}/{val_loader_len},\n"
                  f" Loss: {trainer._log_buffer['val_batch_loss'][-1]:.4f},\n"
                  f" Time: {time.time() - self.last_time:.2f} sec,\n"
                  f" Overall Time: {(time.time() - self.global_start_time)/60:.2f} min")
            self.last_time = time.time()

    def on_validation_end(self, trainer, **kwargs):
        # val_loss = kwargs.get('avg_loss')
        print(f'Validation end!\n'
              f' Avg. Loss: {trainer._log_buffer['val_loss'][-1]:.4f},\n'
              f' Time: {(time.time() - self.start_time)/60:.2f} min')



class TensorBoardLogger(Callback):
    """Логирование в TensorBoard."""
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)

    def on_epoch_end(self, trainer, **kwargs):
        for k, v in trainer.metrics.items():
            self.writer.add_scalar(k, v, trainer.current_epoch)


class MemoryConsolidator(Callback):
    """Периодическая консолидация памяти."""
    def __init__(self, interval: int = 100):
        self.interval = interval

    def on_iteration_end(self, trainer, **kwargs):
        if trainer.global_step % self.interval == 0:
            if hasattr(trainer.model, 'consolidate_memory'):
                trainer.model.consolidate_memory()


class GPUMemoryLogger(Callback):
    def on_batch_end(self, trainer, **kwargs):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**2  # MB
            reserved = torch.cuda.memory_reserved() / 1024**2
            trainer.log('memory/allocated_mb', alloc, phase='custom')
            trainer.log('memory/reserved_mb', reserved, phase='custom')

class NanDetector(Callback):
    """Детектирование NaN в градиентах."""

    def on_before_optimizer_step(self, trainer, **kwargs):
        for name, param in trainer.model.named_parameters():
            if param.grad is None:
                print(param.device)
                raise ValueError(f"None в градиентах {name}")
            # print("Параметр:", name)
            if torch.isnan(param.grad).any():
                raise ValueError(f"NaN в градиентах {name}")


class GradientStats(Callback):
    """Статистика градиентов."""

    def on_before_optimizer_step(self, trainer, **kwargs):
        grads = []
        for param in trainer.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.norm())

        if grads:
            avg_grad = torch.mean(torch.stack(grads))
            trainer.log('grad/avg_norm', avg_grad)


class FreezeUnusedParameters(Callback):
    """Заморозка неиспользуемых параметров."""
    def on_train_start(self, trainer, **kwargs):
        for name, param in trainer.model.named_parameters():
            if 'memory' not in name:
                param.requires_grad = False


class BatchTimeProfiler(Callback):
    """Профилирование времени обработки батча."""
    def __init__(self):
        self.start_time = None

    def on_batch_start(self, trainer, **kwargs):
        self.start_time = time.time()

    def on_batch_end(self, trainer, **kwargs):
        duration = time.time() - self.start_time
        trainer.log('time/batch_sec', duration)

class AttentionVisualizer(Callback):
    """Визуализация attention heads."""
    def __init__(self, layer_idx: int = 0):
        self.layer_idx = layer_idx

    def on_validation_batch_end(self, trainer, **kwargs):
        if hasattr(trainer.model, 'get_attention_maps'):
            attn_maps = trainer.model.get_attention_maps()
            # Визуализация через matplotlib/TensorBoard

class CompositeCallback(Callback):
    """Группировка нескольких callback'ов."""
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks

    def __getattr__(self, name):
        """Делегирование вызовов всем дочерним callback'ам."""
        def handler(*args, **kwargs):
            for cb in self.callbacks:
                if hasattr(cb, name):
                    getattr(cb, name)(*args, **kwargs)
        return handler
