from transformers import GPT2Tokenizer
from datasets import load_dataset
from huggingface_hub import HfFolder

try:
    from config import HF_TOKEN
    print("Hugging Face token found in config.py")
    HfFolder.save_token(HF_TOKEN)
    print("Login to Hugging Face Hub successful.")
except:
    print("Hugging Face token was not found.")

tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    texts = [t + (tokenizer.eos_token or "") for t in examples["text"]]
    return tokenizer(texts, add_special_tokens=False)

def load_and_prepare_webgpt_dataset(typo: str = 'train'):
    print("loading dataset...")
    raw_datasets = load_dataset("reciprocate/oasst_hh_shp_hellaswag_webgpt_rm_dataset")
    print("dataset loaded!")
    print("preparing dataset...")
    text = []
    for examples in raw_datasets[typo]:
        for example in examples["replies"]:
            text.append(examples["prompt"] + example)
    print(f"dataset prepared! total len: {len(text)}")
    return text