import random

import torch
from models.titan import DecoderOnlyMACTitan
from models.titan_hrm import DecoderOnlyMACTitanHRM

def generate(model: DecoderOnlyMACTitan, tokens: torch.Tensor, max_len: int = 512, chunk_size: int = 16, top_k: int = 10, temperature = 0.7):
    seq_len = tokens.shape[-1]
    input_tokens = tokens.view(-1, seq_len)
    memory_state, past_surprise = model.neural_memory.new_states_for_batch(1, "cuda")
    memory_state, past_surprise = model.store(tokens, memory_state, past_surprise, 16)

    generated_tokens = []

    for i in range(max_len):
        logits, new_state, surprise = model(input_tokens[:, max(seq_len - model.window_size, 0):], memory_state, past_surprise)
        next_token = torch.argmax(logits[:, -1, :].view(-1))
        generated_tokens.append(next_token.item())

        input_tokens = torch.cat([input_tokens, next_token.view(-1, 1)], dim=1)
        seq_len += 1

        memory_state, past_surprise = model.neural_memory.new_states_for_batch(1, "cuda")
        memory_state, past_surprise = model.store(input_tokens, memory_state, past_surprise, chunk_size)


    return generated_tokens

#Пример Использования
if __name__ == '__main__':
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    VOCAB_SIZE = len(tokenizer)
    D_MODEL = 384
    N_HEADS = 6
    N_LAYERS = 8
    D_FF = 1024
    MEMORY_DEPTH = 6
    MEMORY_LR = 1e-5
    DROPOUT = 0.1
    WINDOW_SIZE = 128

    model = DecoderOnlyMACTitan(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        memory_depth=MEMORY_DEPTH,
        memory_lr=MEMORY_LR,
        dropout=DROPOUT,
        max_len=WINDOW_SIZE,
    ).to(DEVICE)

    model.load_state_dict(torch.load('../checkpoints/MACTitan_A_384d_nightly_200_scratch_best_val.pt', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    text = """<|prompter|> Hello, can you"""
    tokens = tokenizer.encode(text, return_tensors='pt').to(DEVICE)
    print(tokens)
    generated_text = generate(model, tokens, 128)
    print(tokenizer.decode(generated_text))
