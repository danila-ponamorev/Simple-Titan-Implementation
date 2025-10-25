import torch
from torch import nn
from torch.func import functional_call
import math

from .positional_encoding import PositionalEncoding
from .base_components import DecoderBlock
from .neural_memory import NeuralMemoryAsContextLayer, NeuralMemoryAsContextLayerWithResidual
from .neural_memory_exp import FastNeuralMemoryAsContextLayerWithResidualEXP
class DecoderOnlyMACTitan(nn.Module):
    """
    Простейшая реализвация Titan MemoryAsContext без PersistentMemory.
    """
    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            n_heads: int,
            n_layers: int,
            d_ff: int,
            memory_depth: int,
            memory_lr: float = 1e-5,
            dropout: float = 0.1,
            max_len: int = 512
    ):
        super().__init__()
        self.d_model = d_model
        self.window_size = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.neural_memory = FastNeuralMemoryAsContextLayerWithResidualEXP(d_model, d_ff, n_heads, memory_depth, memory_lr, dropout=dropout)

        self.norm_memory = nn.LayerNorm(d_model)
        self.norm_final = nn.LayerNorm(vocab_size)

        self.dropout_embed = nn.Dropout(dropout)
        self.dropout_pos = nn.Dropout(dropout)
        self.dropout_memory = nn.Dropout(dropout)

    def mask_to_attention_mask(self, mask):
        """
        Конвертирует маску паддинга в маску для внимания.
        """
        batch_size, seq_len = mask.shape
        device = mask.device
        causal_mask = torch.tril(
            torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)).to(device)

        expanded_mask = mask.unsqueeze(1) & mask.unsqueeze(2)

        causal_mask = causal_mask & expanded_mask
        return causal_mask

    def store(self, src, memory_state, past_surprise, chunk_size: int = 256):
        """
        Корректирует веса нейронной памяти в соответствии с текущими входами и предыдущими состояниями

        Args:
            src (torch.Tensor): Входной тензор, форма (batch_size, seq_len)
            memory_state (dict): Словарь вида {name: torch.Tensor}.
            past_surprise (list): Список, вида [torch.Tensor]
            chunk_size (int): Число, размер чанка (разбиения)
        """
        batch_size, seq_len = src.shape
        device = src.device

        new_state, surprise = memory_state, past_surprise
        with torch.no_grad():
            token_embedding_weights_dict = {name: p.clone().detach().to(device) for name, p in self.token_embedding.named_parameters()}
            positional_encoding_weights_dict = {name: p.clone().detach().to(device) for name, p in self.positional_encoding.named_parameters()}
        for i in range(0, seq_len, chunk_size):
            end_index = min(i + chunk_size, seq_len)
            chunk = src[:, i:end_index].to(device)
            _, chunk_len = chunk.shape
            # mask = self._generate_square_subsequent_mask(chunk_len * 2, device)
            embedded_src = functional_call(self.token_embedding, token_embedding_weights_dict, (chunk,))
            src_pos = functional_call(self.positional_encoding, positional_encoding_weights_dict, (embedded_src,))
            new_state, surprise = self.neural_memory.store(src_pos, new_state, surprise)

        return new_state, surprise

    def forward(self, src: torch.Tensor, memory_state, past_surprise) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): Входной тензор, форма (batch_size, seq_len)
            memory_state (dict): Словарь вида {name: torch.tensor}.
            past_surprise (dict): past_surprise (list): Список, вида [torch.Tensor]
        """
        batch_size, seq_len = src.shape
        device = src.device
        mask = src != 0
        src_mask = self.mask_to_attention_mask(mask)

        # 1. Embeddings + positional encodings
        embedded_src = self.token_embedding(src) * math.sqrt(self.d_model)
        embedded_src = self.dropout_embed(embedded_src)

        src_pos = self.positional_encoding(embedded_src)
        src_pos = self.dropout_pos(src_pos)

        # 2. Neural Memory
        mem_res, new_state, surprise = self.neural_memory(src_pos, memory_state, past_surprise, src_mask)
        # mem_res = self.norm_memory(mem_res)
        output = mem_res
        # 3. Decoder Layers
        for layer in self.decoder_layers:
            output = layer(output, src_mask)

        # 4. Final layer
        # output = output[:, seq_len:, :]
        output = self.fc_out(output)

        return output, new_state, surprise
