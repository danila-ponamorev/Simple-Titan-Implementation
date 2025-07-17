import torch
from torch import nn
import math

from .positional_encoding import PositionalEncoding
from .base_components import DecoderBlock

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float = 0.1,
                 max_len: int = 512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def _generate_square_subsequent_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Генерирует каузальную маску (для декодера).

        Args:
            size (int): Размер последовательности
            device (str): Текущее устройство
        """
        return nn.Transformer.generate_square_subsequent_mask(size, device=device)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): Тензор, форма (batch_size, seq_len)
        """
        batch_size, seq_len = src.shape
        device = src.device

        src_mask = self._generate_square_subsequent_mask(seq_len, device)

        embedded_src = self.token_embedding(src) * math.sqrt(self.d_model)
        src_pos = self.positional_encoding(embedded_src)

        output = src_pos
        for layer in self.decoder_layers:
            output = layer(output, src_mask)

        return self.fc_out(output)