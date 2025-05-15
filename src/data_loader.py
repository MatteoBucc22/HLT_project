import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import InputExample
from config import MODEL_NAME, DATASET_NAME, MAX_LENGTH

def get_dataset():
    ds = load_dataset("glue", DATASET_NAME)
    print("DATASET SPLITS:", ds.keys())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def normalize_pair(q1, q2):
        return tuple(sorted((q1.strip().lower(), q2.strip().lower())))

    train_pairs = set(
        normalize_pair(q1, q2)
        for q1, q2 in zip(ds["validation"]["sentence1"], ds["train"]["sentence2"])
    )
    test_pairs = set(
        normalize_pair(q1, q2)
        for q1, q2 in zip(ds["validation"]["sentence1"], ds["test"]["sentence2"])
    )

    duplicates = train_pairs & test_pairs
    print(f"DUPLICATI TRA TRAIN E VALIDATION: {len(duplicates)}")

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

    tokenized = ds.map(preprocess, batched=True, remove_columns=["idx", "sentence1", "sentence2"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    labels = ds["train"]["label"]
    print("VALORI UNICI DELLE ETICHETTE:", set(labels))
    return tokenized

if __name__ == '__main__':
    get_dataset()
