import torch
from torch.utils.data import Dataset

class SimpleTextDatasetWithTokenizerFromFile(Dataset):
    """
    Датасет для текста с токенизацией на лету.
    Принимает путь к файлу, токенизатор, и длину последовательности
    """
    def __init__(self, file_path, tokenizer, seq_len=32):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        self.tokens = tokenizer.encode(text)
        if len(self.tokens) <= seq_len:
            raise IndexError(f"Недостаточная длина последовательности токенов. Минимум: {seq_len}, Дано: {len(self.tokens)}")


        self.num_samples = len(self.tokens) - seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.tokens[0: idx + self.seq_len]

        y = self.tokens[idx + 1: idx + self.seq_len + 1]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class TextDatasetWithTokenizer(Dataset):
    """
    Датасет для текста с токенизацией на лету.
    Принимает путь текст, токенизатор, и длину последовательности
    Предназначен только для Titan, так как x != y (для обработки остальной части памятью).
    """
    def __init__(self, text: str, tokenizer, seq_len: int = 32, min_len: int =32):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.min_len = min_len

        self.tokens = tokenizer.encode(text)
        if len(self.tokens) <= min_len:
            raise IndexError(f"Недостаточная длина последовательности токенов. Минимум: {self.min_len}, Дано: {len(self.tokens)}")

        self.num_samples = len(self.tokens) - min_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.tokens[0: idx + self.min_len]

        y = self.tokens[max(idx + 1 + self.min_len - self.seq_len, 1): idx + self.min_len + 1]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
