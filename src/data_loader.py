import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH
from sklearn.model_selection import train_test_split


def get_datasets(txt_path: str = "/kaggle/working/HLT_project/src/PACCSS-IT.txt") -> DatasetDict:
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
    usando come label binaria: Cosine_Similarity > 0.8.
    Gestisce righe malformate saltandole e converte la similarità in numerico.
    """
    # 1) Carica il dataset dal file .txt (salta righe malformate)
    df = pd.read_csv(
        txt_path,
        sep="\t",
        engine="python",
        on_bad_lines="skip",
        usecols=["Sentence_1", "Sentence_2", "Cosine_Similarity"]
    )

    # 2) Converte Cosine_Similarity in valore numerico, righe con errori diventano NaN
    df["Cosine_Similarity"] = pd.to_numeric(df["Cosine_Similarity"], errors="coerce")

    # 3) Rimuove righe con Cosine_Similarity NaN
    df = df.dropna(subset=["Cosine_Similarity"]).reset_index(drop=True)

    # 4) Rinominazione colonne
    df = df.rename(
        columns={
            "Sentence_1": "question1",
            "Sentence_2": "question2",
            "Cosine_Similarity": "cosine_similarity"
        }
    )

    # 5) Creazione della label binaria: parafrasi se cosine_similarity > 0.8
    df["label"] = (df["cosine_similarity"] > 0.8).astype(int)

    # 6) Filtra e rimuove righe con campi mancanti
    df = df[["question1", "question2", "label"]].dropna().reset_index(drop=True)

    # 7) Split in train (70%), validation (10%), test (20%) stratificato per label
    df_train, df_temp = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=2/3, random_state=42, stratify=df_temp["label"]
    )

    # 8) Creazione del DatasetDict
    ds = DatasetDict({
        "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
        "validation": Dataset.from_pandas(df_val.reset_index(drop=True)),
        "test": Dataset.from_pandas(df_test.reset_index(drop=True)),
    })

    # 9) Shuffle dei split per garantire ordine casuale
    ds = ds.shuffle(seed=42)
    print("DATASET CARICATO E MISCELATO:", ds.keys())

    # 10) Tokenizzazione
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

    # 11) Debug: mostra valori unici delle label
    print("VALORI UNICI DELLE ETICHETTE:", tokenized["train"].unique("labels"))
    return tokenized


if __name__ == '__main__':
    dataset = get_datasets()
    print(dataset)
