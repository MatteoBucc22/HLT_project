import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH, DATASET_NAME

def get_datasets():
    # Carica il dataset Quora con le suddivisioni predefinite
    ds = load_dataset("glue", DATASET_NAME)
    print("DATASET CARICATO:", ds.keys())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Controllo per duplicati tra train e test
    def normalize_pair(q1, q2):
        return tuple(sorted((q1.strip().lower(), q2.strip().lower())))

    train_pairs = set(
        normalize_pair(q1, q2)
        for q1, q2 in zip(ds["validation"]["question1"], ds["train"]["question2"])
    )
    test_pairs = set(
        normalize_pair(q1, q2)
        for q1, q2 in zip(ds["validation"]["question1"], ds["test"]["question2"])
    )

    duplicates = train_pairs & test_pairs
    print(f"DUPLICATI TRA TRAIN E VALIDATION: {len(duplicates)}")

    def preprocess(examples):
        # Access question texts directly from "question1" and "question2"
        q1 = examples["question1"]
        q2 = examples["question2"]

        tok = tokenizer(
            q1,
            q2,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tok["labels"] = examples["label"]  # Use "label" for is_duplicate
        return tok

    # Applica la tokenizzazione a tutte le suddivisioni
    tokenized = ds.map(preprocess, batched=True, remove_columns=["idx", "question1", "question2"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    labels = ds["train"]["label"]
    print("VALORI UNICI DELLE ETICHETTE:", set(labels))
    return tokenized

if __name__ == '__main__':
    dataset = get_datasets()
    print(dataset)
