import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH

def get_datasets():
    # Carica il CSV
    df = pd.read_csv("SimilEx_dataset.csv")

    # Tieni solo le colonne di interesse e rimuovi righe senza etichetta
    df = df[["Sentence_1", "Sentence_2", "Stud_2"]].dropna(subset=["Stud_2"])

    # Rinomina per compatibilità col preprocess
    df = df.rename(columns={"Sentence_1": "question1", "Sentence_2": "question2", "Stud_2": "label"})

    # Binarizza le etichette (es: >=4 → 1, altrimenti 0)
    df["label"] = df["label"].apply(lambda x: 1 if x >= 4 else 0)

    # Suddivisione train/val
    split_idx = int(0.9 * len(df))
    ds = DatasetDict({
        "train": Dataset.from_pandas(df[:split_idx].reset_index(drop=True)),
        "validation": Dataset.from_pandas(df[split_idx:].reset_index(drop=True)),
    })

    print("DATASET CARICATO:", ds.keys())
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

    # Tokenizzazione
    tokenized = ds.map(preprocess, batched=True, remove_columns=["question1", "question2", "__index_level_0__"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    print("VALORI UNICI DELLE ETICHETTE:", set(df["label"]))
    return tokenized

if __name__ == '__main__':
    dataset = get_datasets()
    print(dataset)
