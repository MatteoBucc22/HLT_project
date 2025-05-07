import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH

def get_datasets():
    # Carica il dataset Quora con le suddivisioni predefinite
    ds = load_dataset("glue", "quora")
    print("DATASET CARICATO:", ds.keys())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        q1 = [q["text"][0] for q in examples["questions"]]
        q2 = [q["text"][1] for q in examples["questions"]]

        tok = tokenizer(
            q1,
            q2,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tok["labels"] = [int(x) for x in examples["is_duplicate"]]
        return tok



    # Applica la tokenizzazione a tutte le suddivisioni
    tokenized = ds.map(preprocess, batched=True)

    return tokenized

