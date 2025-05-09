import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH
from sklearn.model_selection import train_test_split


def get_datasets(
    txt_path: str = "/kaggle/working/HLT_project/src/PACCSS-IT.txt",
    t0: float = 0.8,
    beta: float = 0.1
) -> DatasetDict:
    """
    Carica un file .txt tab-delimitato con colonne:
      - Sentence_1
      - Sentence_2
      - Cosine_Similarity
      - Confidence
      - Readability_1
      - Readability_2
      - (Readability_1-Readability_2)
    e restituisce un DatasetDict con split train/validation/test (70/10/20)
    usando come label binaria con soglia adattiva:
      threshold = t0 + beta * (1 - confidence)
    e label = cosine_similarity > threshold.
    """
    # 1) Carica il dataset (salta righe malformate)
    df = pd.read_csv(
        txt_path,
        sep="\t",
        engine="python",
        on_bad_lines="skip",
        usecols=["Sentence_1", "Sentence_2", "Cosine_Similarity", "Confidence"]
    )

    # 2) Converto in numerico
    df["Cosine_Similarity"] = pd.to_numeric(df["Cosine_Similarity"], errors="coerce")
    df["Confidence"]       = pd.to_numeric(df["Confidence"], errors="coerce")
    df = df.dropna(subset=["Cosine_Similarity", "Confidence"]).reset_index(drop=True)

    # 3) Rinomino
    df = df.rename(
        columns={
            "Sentence_1": "question1",
            "Sentence_2": "question2",
            "Cosine_Similarity": "cosine_similarity",
            "Confidence": "confidence"
        }
    )

    # 4) Soglia adattiva e binarizzazione
    df["threshold"] = t0 + beta * (1 - df["confidence"])
    df["label"]     = (df["cosine_similarity"] > df["threshold"]).astype(int)

    # 5) Preparo le colonne
    df = df[["question1", "question2", "label"]]

    # 6) Split 70/10/20 stratificato
    df_train, df_temp = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=2/3, random_state=42, stratify=df_temp["label"]
    )

    # 7) Costruzione DatasetDict
    ds = DatasetDict({
        "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
        "validation": Dataset.from_pandas(df_val.reset_index(drop=True)),
        "test": Dataset.from_pandas(df_test.reset_index(drop=True)),
    })

    # 8) Shuffle
    ds = ds.shuffle(seed=42)
    print("DATASET CARICATO E MISCELATO:", ds.keys())

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

    return tokenized


if __name__ == '__main__':
    dataset = get_datasets()
    print(dataset)
