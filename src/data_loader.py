import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH
from sklearn.model_selection import train_test_split


def get_datasets(
    txt_path: str = "/kaggle/working/HLT_project/src/SimilEx_dataset.txt",
    t0: float = 0.8,
    beta: float = 0.1
) -> DatasetDict:
    """
    Carica un file .txt tab-delimitato con colonne:
      - Sentence_1
      - Sentence_2
      - Cosine_Similarity
      - Confidence
    e restituisce un DatasetDict con split train/validation/test (70/10/20)
    usa soglia adattiva basata su confidence:
      threshold = t0 + beta * (1 - confidence)
      label = cosine_similarity > threshold
    Filtra solo righe con confidence > 0.9.
    """
    # 1) Carica il dataset (salta righe malformate)
    df = pd.read_csv(
        txt_path,
        sep="\t",
        engine="python",
        on_bad_lines="skip",
        usecols=["Sentence_1", "Sentence_2", "Cosine_Similarity", "Confidence"]
    )

    # 2) Converte in numerico e rimuove righe malformate
    df["Cosine_Similarity"] = pd.to_numeric(df["Cosine_Similarity"], errors="coerce")
    df["Confidence"]       = pd.to_numeric(df["Confidence"], errors="coerce")
    df = df.dropna(subset=["Cosine_Similarity", "Confidence"]).reset_index(drop=True)

    # 3) Filtra solo righe con confidence > 0.9
    df = df[df["Confidence"] > 0.9].reset_index(drop=True)

    # Stampa numero esempi originali
    original_count = len(df)
    print(f"Numero esempi originali (confidence>0.9): {original_count}")

    # 4) Rinomina colonne
    df = df.rename(
        columns={
            "Sentence_1": "question1",
            "Sentence_2": "question2",
            "Cosine_Similarity": "cosine_similarity",
            "Confidence": "confidence"
        }
    )

    # 5) Calcola soglia adattiva e binarizza
    df["threshold"] = t0 + beta * (1 - df["confidence"])
    df["label"]     = (df["cosine_similarity"] > df["threshold"]).astype(int)

    # Stampa esempi con label=1
    pos_count = int(df["label"].sum())
    print(f"Numero esempi con label=1 (parafrasi): {pos_count}")

    # 6) Seleziona colonne utili
    df = df[["question1", "question2", "label"]]

    # 7) Split stratificato 70/10/20
    df_train, df_temp = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=2/3, random_state=42, stratify=df_temp["label"]
    )

    # 8) Crea DatasetDict
    ds = DatasetDict({
        "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
        "validation": Dataset.from_pandas(df_val.reset_index(drop=True)),
        "test": Dataset.from_pandas(df_test.reset_index(drop=True)),
    })

    # Shuffle e stampa conteggi
    ds = ds.shuffle(seed=42)
    print("Dataset splits:", {k: len(v) for k, v in ds.items()})

    # 9) Tokenizzazione
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def preprocess(examples):
        tokens = tokenizer(
            examples["question1"],
            examples["question2"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokens["labels"] = examples["label"]
        return tokens

    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=["question1", "question2"]
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 10) Debug: distribuzione etichette
    print("Distribuzione label nel train split:", tokenized["train"].unique("labels"))
    return tokenized


if __name__ == '__main__':
    dataset = get_datasets()
    print(dataset)
