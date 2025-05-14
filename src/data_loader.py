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

    # Rimozione overlap tra split
    def get_sentences(df):
        return set(df["Sentence_1"]).union(set(df["Sentence_2"]))

    # Rimuovi da validation tutte le righe che contengono frasi già nel train
    train_sentences = get_sentences(df_train)
    mask_val = ~(
        df_val["Sentence_1"].isin(train_sentences) |
        df_val["Sentence_2"].isin(train_sentences)
    )
    removed_val = (~mask_val).sum()
    if removed_val > 0:
        print(f"[INFO] Rimosse {removed_val} righe da validation per overlap con train")
    df_val = df_val[mask_val]

    # Aggiorna set frasi validation
    val_sentences = get_sentences(df_val)
    all_train_val_sentences = train_sentences.union(val_sentences)

    # Rimuovi da test tutte le righe che contengono frasi già in train o validation
    mask_test = ~(
        df_test["Sentence_1"].isin(all_train_val_sentences) |
        df_test["Sentence_2"].isin(all_train_val_sentences)
    )
    removed_test = (~mask_test).sum()
    if removed_test > 0:
        print(f"[INFO] Rimosse {removed_test} righe da test per overlap con train/validation")
    df_test = df_test[mask_test]

    # Ricampionamento se necessario (soglia: almeno 90% della dimensione desiderata)
    desired_val = int(0.10 * len(df))
    desired_test = int(0.20 * len(df))
    if len(df_val) < 0.9 * desired_val:
        print(f"[INFO] Ricampionamento validation: {len(df_val)} < 90% di {desired_val}")
        remaining = df_temp.drop(df_val.index)
        extra, _ = train_test_split(
            remaining,
            train_size=desired_val - len(df_val),
            stratify=remaining["label"],
            random_state=42
        )
        df_val = pd.concat([df_val, extra]).drop_duplicates().reset_index(drop=True)
    if len(df_test) < 0.9 * desired_test:
        print(f"[INFO] Ricampionamento test: {len(df_test)} < 90% di {desired_test}")
        remaining = df_temp.drop(df_test.index)
        extra, _ = train_test_split(
            remaining,
            train_size=desired_test - len(df_test),
            stratify=remaining["label"],
            random_state=42
        )
        df_test = pd.concat([df_test, extra]).drop_duplicates().reset_index(drop=True)

    # Controllo duplicati tra split
    def check_overlap(col):
        train_set = set(df_train[col])
        val_set = set(df_val[col])
        test_set = set(df_test[col])
        overlap_train_val = train_set & val_set
        overlap_train_test = train_set & test_set
        overlap_val_test = val_set & test_set
        if overlap_train_val:
            print(f"[WARNING] {col}: {len(overlap_train_val)} frasi in comune tra train e validation")
        if overlap_train_test:
            print(f"[WARNING] {col}: {len(overlap_train_test)} frasi in comune tra train e test")
        if overlap_val_test:
            print(f"[WARNING] {col}: {len(overlap_val_test)} frasi in comune tra validation e test")
        if not (overlap_train_val or overlap_train_test or overlap_val_test):
            print(f"[OK] Nessuna frase duplicata tra split per la colonna {col}")

    check_overlap("Sentence_1")
    check_overlap("Sentence_2")

    # Controllo duplicati a livello di riga intera
    def check_row_overlap(dfs, names):
        row_sets = {
            name: set(df[['Sentence_1','Sentence_2','label']].apply(tuple, axis=1))
            for name, df in zip(names, dfs)
        }
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if j <= i: continue
                common = row_sets[ni] & row_sets[nj]
                msg = f"{len(common)} righe identiche tra {ni} e {nj}"
                print(f"[OK] {msg}" if not common else f"[WARNING] {msg}")

    check_row_overlap(
        [df_train, df_val, df_test],
        ['train','validation','test']
    )

    # 8) Crea DatasetDict
    ds = DatasetDict({
        "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
        "validation": Dataset.from_pandas(df_val.reset_index(drop=True)),
        "test": Dataset.from_pandas(df_test.reset_index(drop=True)),
    })

    # Shuffle e stampa conteggi
    #ds = ds.shuffle(seed=42)
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
