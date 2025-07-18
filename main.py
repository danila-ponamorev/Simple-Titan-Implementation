import torch
from torch.utils.data import DataLoader, ConcatDataset
from models import DecoderOnlyMACTitan
from training.trainer import DecoderOnlyMACTitanBaseTrainer
from utils.dataset import TextDatasetWithTokenizer
from transformers import GPT2Tokenizer
from training.callbacks import MetricLogger, NanDetector, GPUMemoryLogger, LogPrinter, ModelCheckpoint
from utils.helpers import init_weights, read_files
from utils.loaders import PaddedDataLoader

tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-5

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
).to(DEVICE)

model.apply(init_weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
criterion = torch.nn.CrossEntropyLoss()

text = read_files(['data/text1.txt', 'data/text2.txt', 'data/text3.txt', 'data/test.txt'], encoding='utf-8')
split_text = list(map(lambda x: x.replace('\n\n', '\n').split('Часть '), text))


processed_text = []
for text_fragment in split_text:
    processed_text.extend(text_fragment)

filtered_text = []
for text_fragment in processed_text:
    if len(text_fragment.split(' ')) >= WINDOW_SIZE:
        filtered_text.append(text_fragment)
# print(processed_text)
mini_datasets = [TextDatasetWithTokenizer(
    text_fragment,
    tokenizer,
    seq_len=WINDOW_SIZE,
    min_len=WINDOW_SIZE,
    ) for text_fragment in filtered_text]

# combined_dataset = ConcatDataset(mini_datasets)

train_dataset = ConcatDataset(mini_datasets[:-2])
val_dataset = ConcatDataset(mini_datasets[-2:])

train_loader = PaddedDataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    pad_token=0,
    shuffle=True,
)

val_loader = PaddedDataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    pad_token=0,
    shuffle=True,
)

callbacks = [
    MetricLogger(),
    LogPrinter(100),
    # NanDetector(),
    GPUMemoryLogger(),
    ModelCheckpoint('checkpoints/MACTitan.pt'),
]


trainer = DecoderOnlyMACTitanBaseTrainer(
    padding_token=tokenizer.pad_token_id,
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