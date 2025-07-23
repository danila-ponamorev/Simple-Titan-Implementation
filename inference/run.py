import random

import torch
from models.titan import DecoderOnlyMACTitan

def generate(model: DecoderOnlyMACTitan, tokens: torch.Tensor, max_len: int = 512, top_k: int = 2, temperature = 0.1):
    seq_len = tokens.shape[-1]
    input_tokens = tokens.view(-1, seq_len)
    memory_state, past_surprise = model.neural_memory.reset_memory_batch(1)

    memory_state, past_surprise = model.store(tokens, memory_state, past_surprise, 64)

    generated_tokens = []

    for i in range(max_len):
        logits = model(input_tokens[:, max(seq_len - model.window_size, 0):], memory_state)
        next_token = torch.argmax(logits[:, -1, :].view(-1))
        # print(next_token.item())
        # break
        # probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        # top_probs, top_indices = torch.topk(probs, top_k)
        # next_token = torch.multinomial(top_probs, num_samples=1)
        # # next_token = next_token.view(-1, 1)
        # # print(next_token.shape)
        generated_tokens.append(next_token.item())

        input_tokens = torch.cat([input_tokens, next_token.view(-1, 1)], dim=1)

        memory_state, past_surprise = model.store(next_token.view(-1, 1), memory_state, past_surprise)
        seq_len += 1

    return generated_tokens

if __name__ == '__main__':
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    VOCAB_SIZE = len(tokenizer)
    D_MODEL = 768
    N_HEADS = 8
    N_LAYERS = 8
    D_FF = 3072
    MEMORY_DEPTH = 8
    MEMORY_LR = 1e-5
    DROPOUT = 0.1
    WINDOW_SIZE = 256

    model = DecoderOnlyMACTitan(
        VOCAB_SIZE,
        D_MODEL,
        N_HEADS,
        N_LAYERS,
        D_FF,
        MEMORY_DEPTH,
        MEMORY_LR,
        DROPOUT,
        WINDOW_SIZE,
    )
    model.load_state_dict(torch.load('../checkpoints/MACTitan_best.pt', map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    text = """Том 14 Глава 1 - Божественная кара для этих святых доспехов!
Часть 1
На днях кандидат в генералы Короля демонов, смазливый падший ангел Дюк, был уничтожен.

Но перед этим он успел разбить Виз сердце, из-за чего она теперь безвылазно сидела в магазине. Аква каждый день ходила её утешать.

И сегодня было то же самое.""".replace('\n\n', '\n')
    tokens = tokenizer.encode(text, return_tensors='pt').to(DEVICE)
    generated_text = generate(model, tokens, 32)
    print(tokenizer.decode(generated_text))
