import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import AutoTokenizer
from .config_miniLM_qqp import MODEL_NAME, MAX_LENGTH, DATASET_NAME

def get_datasets():
    ds = load_dataset("glue", DATASET_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        q1 = examples["question1"]
        q2 = examples["question2"]
        tok = tokenizer(
            q1,
            q2,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tok["labels"] = examples["label"]
        return tok

    tokenized = ds.map(preprocess, batched=True, remove_columns=["idx", "question1", "question2"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized

if __name__ == '__main__':
    dataset = get_datasets()
    print(dataset)