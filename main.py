import random
import json

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import GPT2Tokenizer

import config
from models.titan_hrm import DecoderOnlyMACTitanHRM
from models.titan import DecoderOnlyMACTitan
from training.trainer import DecoderOnlyMACTitanBaseTrainer
from utils.dataset import TextDatasetWithTokenizer
from training.callbacks import MetricLogger, NanDetector, GPUMemoryLogger, LogPrinter, ModelCheckpoint, \
    ValidationCallback, LRScheduler, EarlyStopping
from utils.helpers import init_weights, read_files, \
    create_linear_forward_patcher, silu_weight_transpose_forward, norm_weight_transpose_forward
from utils.loaders import PaddedDataLoader


torch.set_printoptions(threshold=100000)

tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
VAL_BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 1e-4

VOCAB_SIZE = len(tokenizer)
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 8
D_FF = 768
MEMORY_DEPTH = 6
MEMORY_LR = 1e-5
DROPOUT = 0.1
WINDOW_SIZE = 128

H_depth = 5
L_depth = 2
H_cycles = 3
L_cycles = 3

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


# model.load_state_dict(torch.load('checkpoints/MACTitan_A_384d_nightly_best_val.pt', 'cuda'))
model.apply(init_weights)
model.apply(create_linear_forward_patcher(norm_weight_transpose_forward))

print(f"Number of trainable parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
# scheduler = CosineAnnealingLR(optimizer, 250_000)


from data.prepare_datasets import load_and_prepare_webgpt_dataset
text = load_and_prepare_webgpt_dataset('train')


filtered_text = []

for text_fragment in text:
    if len(text_fragment.split(' ')) >= WINDOW_SIZE:
        filtered_text.append(text_fragment)

print(len(filtered_text))

filtered_text = filtered_text[:10000]

random.shuffle(filtered_text)

print("Processing micro-datasets...")

mini_datasets = [TextDatasetWithTokenizer(
    text_fragment,
    tokenizer,
    seq_len=WINDOW_SIZE,
    min_len=8,
    ) for text_fragment in filtered_text]

print("Datasets are done!")

train_dataset = ConcatDataset(mini_datasets[:-100])
val_dataset = ConcatDataset(mini_datasets[-50:])


train_loader = PaddedDataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    pad_token=0,
    shuffle=False,
)

val_loader = PaddedDataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    pad_token=0,
    shuffle=False,
)

user_id = config.USER_ID


callbacks = [
    MetricLogger(),
    LogPrinter(user_id, 50),
    # NanDetector(),
    GPUMemoryLogger(),
    ModelCheckpoint('checkpoints/MACTitan_A_384d_030925_scratch.pt'),
    ValidationCallback(val_interval=1000),
    EarlyStopping(patience=8)
]


trainer = DecoderOnlyMACTitanBaseTrainer(
    padding_token=0,
    chunk_size=16,
    model=model,
    optimizer=optimizer,
    loss_fn=criterion,
    train_loader=train_loader,
    val_loader=val_loader,
    device=DEVICE,
    grad_accumulation_steps=1,
    callbacks=callbacks
)

trainer.train(EPOCHS)

print(trainer._log_buffer)