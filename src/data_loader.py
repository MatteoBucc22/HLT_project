import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH
from sklearn.model_selection import train_test_split


def get_datasets(
    txt_path: str = "/kaggle/working/HLT_project/src/SimilEx_dataset.txt",
    cos_thresh: float = 0.8,
    conf_thresh: float = 0.75
) -> DatasetDict:
    """
    Carica un file .txt tab-delimitato con colonne:
      - Sentence_1
      - Sentence_2
      - Cosine_Similarity
      - Confidence
    e restituisce un DatasetDict con split train/validation/test (70/10/20)
    usando come label binaria con tecnica AND:
      label = (cosine_similarity > cos_thresh) AND (confidence > conf_thresh)
    """
    # 1) Carica il dataset (salta righe malformate)
    df = pd.read_csv(
        txt_path,
        sep="\t",
        engine="python",
        on_bad_lines="skip",
        usecols=["Sentence_1", "Sentence_2", "Cosine_Similarity", "Confidence"]
    )

    # 2) Converto in numerico e scarto righe malformate
    df["Cosine_Similarity"] = pd.to_numeric(df["Cosine_Similarity"], errors="coerce")
    df["Confidence"]       = pd.to_numeric(df["Confidence"], errors="coerce")
    df = df.dropna(subset=["Cosine_Similarity", "Confidence"]).reset_index(drop=True)

    # 3) Rinomino colonne per compatibilità
    df = df.rename(
        columns={
            "Sentence_1": "question1",
            "Sentence_2": "question2",
            "Cosine_Similarity": "cosine_similarity",
            "Confidence": "confidence"
        }
    )

    # Stampa numero esempi originali (dopo dropna)
    original_count = len(df)
    print(f"Numero esempi originali: {original_count}")

    # 4) Binarizzazione con tecnica AND
    df["label"] = (
        (df["cosine_similarity"] > cos_thresh) &
        (df["confidence"]       > conf_thresh)
    ).astype(int)

    # Stampa quanti esempi etichettati come paraphrase
    pos_count = int(df["label"].sum())
    print(f"Numero esempi con label=1 (parafrasi): {pos_count}")
    df["label"] = (
        (df["cosine_similarity"] > cos_thresh) &
        (df["confidence"]       > conf_thresh)
    ).astype(int)

    # 5) Seleziono solo colonne utili
    df = df[["question1", "question2", "label"]]

    # 6) Split stratificato 70/10/20
    df_train, df_temp = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=2/3, random_state=42, stratify=df_temp["label"]
    )

    # 7) Crea DatasetDict Hugging Face
    ds = DatasetDict({
        "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
        "validation": Dataset.from_pandas(df_val.reset_index(drop=True)),
        "test": Dataset.from_pandas(df_test.reset_index(drop=True)),
    })

    # 8) Shuffle
    ds = ds.shuffle(seed=42)
    print("DATASET CARICATO E MISCELATO:", ds.keys())

    # Stampare il numero di esempi per split
    print(f"Numero esempi - train: {len(ds['train'])}, validation: {len(ds['validation'])}, test: {len(ds['test'])}")

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

    # 10) Debug: class distribution
    print("VALORI UNICI DELLE ETICHETTE:", tokenized["train"].unique("labels"))
    return tokenized


if __name__ == '__main__':
    dataset = get_datasets()
    print(dataset)
