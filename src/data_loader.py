import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH
from sklearn.model_selection import train_test_split


def get_datasets(
    data_path: str = "/kaggle/working/HLT_project/src/PACCSS-IT-final.csv",
    t0: float = 0.8,
    beta: float = 0.1
) -> DatasetDict:
    """
    Carica un file .csv tab-delimitato con colonne:
      - Sentence_1
      - Sentence_2
      - label
    """
    # 1) Carica il dataset (salta righe malformate)
    df = pd.read_csv(
        data_path,
        engine="python",
        on_bad_lines="skip",
        usecols=["Sentence_1", "Sentence_2", "label"]
    )


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
            examples["Sentence_1"],
            examples["Sentence_2"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokens["labels"] = examples["label"]
        return tokens

    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=["Sentence_1", "Sentence_2"]
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 10) Debug: distribuzione etichette
    print("Distribuzione label nel train split:", tokenized["train"].unique("labels"))
    return tokenized


if __name__ == '__main__':
    dataset = get_datasets()
    print(dataset)
