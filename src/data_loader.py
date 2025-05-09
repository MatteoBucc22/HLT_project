import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH
from sklearn.model_selection import train_test_split

def get_datasets():
    # Carica il CSV
    df = pd.read_csv("/kaggle/working/HLT_project/src/SimilEx_dataset.csv")

    # Calcola la media delle annotazioni A1–A6
    annot_cols = ["A1", "A2", "A3", "A4", "A5", "A6"]
    df["label"] = df[annot_cols].mean(axis=1)

    # Rinomina le colonne
    df = df.rename(columns={"Sentence_1": "question1", "Sentence_2": "question2"})

    # Rimuove righe con valori mancanti
    df = df[["question1", "question2", "label"]].dropna()

    # Binarizza la media: >= 4 → 1 (parafrasi), altrimenti 0
    df["label"] = df["label"].apply(lambda x: 1 if x >= 4 else 0)

    # Split 70% train, 10% val, 20% test
    df_train, df_temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
    df_val, df_test = train_test_split(df_temp, test_size=2/3, random_state=42, stratify=df_temp["label"])

    # DatasetDict Hugging Face
    ds = DatasetDict({
        "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
        "validation": Dataset.from_pandas(df_val.reset_index(drop=True)),
        "test": Dataset.from_pandas(df_test.reset_index(drop=True)),
    })

    print("DATASET CARICATO:", ds.keys())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        tok = tokenizer(
            examples["question1"],
            examples["question2"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tok["labels"] = examples["label"]
        return tok

    # Tokenizzazione
    tokenized = ds.map(preprocess, batched=True, remove_columns=["question1", "question2"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    print("VALORI UNICI DELLE ETICHETTE:", set(df["label"]))
    return tokenized

if __name__ == '__main__':
    dataset = get_datasets()
    print(dataset)
