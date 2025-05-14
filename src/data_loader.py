import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH
from sklearn.model_selection import train_test_split


def get_datasets(
    train_path: str = "/kaggle/working/HLT_project/src/train.csv",
    test_path: str = "/kaggle/working/HLT_project/src/test.csv",
    val_path: str = "/kaggle/working/HLT_project/src/val.csv",
    t0: float = 0.8,
    beta: float = 0.1
) -> DatasetDict:
    
    ds = DatasetDict({
        "train": Dataset.from_pandas(pd.read_csv(train_path)),
        "validation": Dataset.from_pandas(pd.read_csv(val_path)),
        "test": Dataset.from_pandas(pd.read_csv(test_path))
    })


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
