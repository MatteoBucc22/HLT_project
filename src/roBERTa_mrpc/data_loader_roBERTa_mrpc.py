import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import AutoTokenizer
from .config_roBERTa_mrpc import MODEL_NAME, MAX_LENGTH, DATASET_NAME

def get_datasets():
    ds = load_dataset("glue", DATASET_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        q1 = examples["sentence1"]
        q2 = examples["sentence2"]

        tok = tokenizer(
            q1,
            q2,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tok["labels"] = examples["label"]
        return tok


    tokenized = ds.map(preprocess, batched=True, remove_columns=["sentence1", "sentence2", "label", "idx"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized

if __name__ == '__main__':
    dataset = get_datasets()
    print(dataset)