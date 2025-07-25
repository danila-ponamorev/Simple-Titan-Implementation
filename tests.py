"""Используется как тестовый полигон."""
import time

import torch
from models.neural_memory import NeuralMemoryAsContextLayer
from models.neural_memory_exp import FastNeuralMemoryAsContextLayer

fast_neural_memory = FastNeuralMemoryAsContextLayer(256, 3072, 8, 4, learning_rate=1e-5)
neural_memory = NeuralMemoryAsContextLayer(256, 3072, 8, 4, learning_rate=1e-5)

BATCH_SIZE = 32
D_MODEL = 256
seq_len = 128

input_tensor = torch.randn(BATCH_SIZE, seq_len, D_MODEL).to('cuda')

neural_memory.to('cuda')
mem_state, surprise = neural_memory.reset_memory_batch(BATCH_SIZE)
start_time = time.time()
for i in range(100):
    mem_state, surprise = neural_memory.store(input_tensor, mem_state, surprise)
print(f"Overal time = {time.time() - start_time}")
del neural_memory
torch.cuda.empty_cache()

print('sleeping')
time.sleep(10)
print('continuing')

fast_neural_memory.to('cuda')
mem_state, surprise = fast_neural_memory.new_states_for_batch(BATCH_SIZE, 'cuda')
start_time = time.time()
for i in range(100):
    mem_state, surprise = fast_neural_memory.store(input_tensor, mem_state, surprise)
print(f"Overal time = {time.time() - start_time}")
del fast_neural_memory
torch.cuda.empty_cache()

