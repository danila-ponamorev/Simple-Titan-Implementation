import torch
import random
from torch.utils.data.dataloader import _BaseDataLoaderIter, DataLoader
from typing import Tuple

class PaddedDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader, pad_token=0):
        super().__init__(loader)
        self.pad_token = pad_token
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.shuffle = loader.shuffle

        # Initialize indices
        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(self.indices)

        self.current_idx = 0

    def _pad_collate(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Unpack the batch (assuming each item is a tuple of tensors)
        items1, items2 = zip(*batch)

        # Pad first sequence
        max_len1 = max(item.size(0) for item in items1)
        padded1 = []
        for item in items1:
            pad_size = max_len1 - item.size(0)
            if pad_size > 0:
                padding = torch.full((pad_size,) + item.shape[1:],
                                     self.pad_token,
                                     dtype=item.dtype,
                                     device=item.device)
                padded_item = torch.cat([item, padding], dim=0)
            else:
                padded_item = item
            padded1.append(padded_item)

        # Pad second sequence
        max_len2 = max(item.size(0) for item in items2)
        padded2 = []
        for item in items2:
            pad_size = max_len2 - item.size(0)
            if pad_size > 0:
                padding = torch.full((pad_size,) + item.shape[1:],
                                     self.pad_token,
                                     dtype=item.dtype,
                                     device=item.device)
                padded_item = torch.cat([item, padding], dim=0)
            else:
                padded_item = item
            padded2.append(padded_item)

        # Stack padded sequences
        return torch.stack(padded1), torch.stack(padded2)

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.current_idx >= len(self.indices):
            raise StopIteration

        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size

        batch = [self.dataset[i] for i in batch_indices]
        return self._pad_collate(batch)

    def __iter__(self):
        return self

    def reset(self):
        self.current_idx = 0
        if self.shuffle:
            random.shuffle(self.indices)
