from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from typing import Any, List, Optional, Union
import torch

from .iterators import PaddedDataLoaderIter

class MixedDataLoader(IterableDataset):
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        its = [iter(loader) for loader in self.loaders]
        while True:
            try:
                # Поочередно берем батчи из каждого загрузчика
                for it in its:
                    yield next(it)
            except StopIteration:
                return

class ParallelLoader:
    """
    Параллельный загрузчик данных из нескольких DataLoader'ов.
    Позволяет одновременно итерироваться по нескольким даталоадерам,
    возвращая батчи из всех источников в виде кортежа.
    """
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        for batches in zip(*self.loaders):
            yield batches


class PaddedDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, pad_token=0, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.pad_token = pad_token
        self.shuffle = shuffle

    def _get_iterator(self):
        return PaddedDataLoaderIter(self, pad_token=self.pad_token)