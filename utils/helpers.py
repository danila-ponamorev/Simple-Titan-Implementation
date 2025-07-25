from typing import List, Union, Dict

import torch
from torch import nn

def normalize_grad(grad, max_norm: float = 1.0):
    """
    Нормализует градиенты весов
    Args:
        grad (tuple): кортеж градиентов весов вида (torch.Tensor)
        max_norm (float): Максимальная норма
    """
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grad]), 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    grad = [g * clip_coef if clip_coef < 1 else g for g in grad]
    return grad

def normalize_grad_fast(grad: Dict, max_norm: float = 1.0):
    """
    Нормализует градиенты весов
    Args:
        grad (tuple): кортеж градиентов весов вида (torch.Tensor)
        max_norm (float): Максимальная норма
    """
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grad.values()]), 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    grad = {n: g * clip_coef if clip_coef < 1 else g for n, g in grad.items()}
    return grad

def debug_tensor(tensor: torch.Tensor, name: str):
    """
    Проверяет тензор на наличие NaN/Inf и печатает его статистику.
    Args:
        tensor (torch.Tensor): Входной тензор
        name (str): Отображаемое имя
    """
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    if has_nan or has_inf:
        print(f"!!! ALERT: NaN or Inf found in tensor '{name}' !!!")
        print(f"    Has NaN: {has_nan}, Has Inf: {has_inf}")

        raise RuntimeError(f"NaN/Inf detected in {name}")
    else:
        # Если все хорошо, печатаем полезную статистику
        print(
            f"  DEBUG: Tensor '{name}' | "
            f"Shape: {tensor.shape} | "
            f"Mean: {tensor.mean().item():.4f} | "
            f"Std: {tensor.std().item():.4f} | "
            f"Min: {tensor.min().item():.4f} | "
            f"Max: {tensor.max().item():.4f}"
        )

def init_weights(m):
    """
    Инициализирует веса слоя nn.Embedding или nn.Linear
    Args:
        m (nn.Linear or nn.Embedding): Слой nn.Embedding или nn.Linear
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.02)


def align_tensors_by_length(
        tensor_list: List[torch.Tensor],
        align_value : Union[int, float]= 0,
        value_type: torch.dtype = torch.float32,
        dim: int = 1):
    """
     Выравнивает тензоры по максимальной длине заданного измерения измерения и заполняет оставшуюся чать align_value.
     Ожидается, что все измерения, кроме указанного dim, уже выравнены, а также измерения тензоров сходятся. По умолчанию заполняет всё нулями.
     Полезно для выравнивания батча
     Args:
         tensor_list (dict): список входных тензоров вида Dict[torch.Tensor].
         align_value (int or float): Значения выравнивания.
         value_type (type): Тип значения входных тензоров.
         dim (int): Измерение по которому будет выполнено выравнивание.

    """
    if not tensor_list:
        return []

    # Проверка согласованности размерностей
    ref_shape = list(tensor_list[0].shape)
    del ref_shape[dim]  # Игнорируем измерение для выравнивания

    for tensor in tensor_list[1:]:
        current_shape = list(tensor.shape)
        del current_shape[dim]
        if current_shape != ref_shape:
            raise ValueError(
                f"Несогласованные размеры тензоров. Ожидается {ref_shape} без dim {dim}, получено {current_shape}")

    max_length = max(tensor.shape[dim] for tensor in tensor_list)
    aligned_tensors = []

    for tensor in tensor_list:
        pad_size = max_length - tensor.shape[dim]
        if pad_size > 0:
            # Создаем padding tensor с правильными размерами
            pad_shape = list(tensor.shape)
            pad_shape[dim] = pad_size
            padding = torch.full(
                pad_shape,
                align_value,
                dtype=value_type,
                device=tensor.device
            )

            # Конкатенируем в правильном измерении
            aligned_tensor = torch.cat([padding, tensor], dim=dim)
        else:
            aligned_tensor = tensor

        aligned_tensors.append(aligned_tensor)

    return aligned_tensors

def read_files(files: List[str], encoding='utf-8'):
    """
    Читает файлы в формате .txt из списка и возращает их содержимое
    Args:
        files (list): список путей к файлам
    """
    text = []
    for file in files:
        with open(file, 'r', encoding=encoding) as f:
            text.append(f.read())

    return text

def align_sequences(sequences, pad_value=0, dim=0):
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

            aligned_seq = torch.cat([padding, seq], dim=dim)
        else:
            aligned_seq = seq

        aligned.append(aligned_seq)

    return torch.stack(aligned)







