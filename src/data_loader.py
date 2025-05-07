import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH

def get_datasets():
    # Carica il dataset Quora con le suddivisioni predefinite
    ds = load_dataset("quora")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        # Usa direttamente le due colonne
        q1 = examples["questions"]["text1"]
        q2 = examples["questions"]["text2"]

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

