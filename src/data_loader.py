import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH, DATASET_NAME  # Assicurati che DATASET_NAME = "qqp"

def get_datasets():
    # Carica il dataset QQP da GLUE
    ds = load_dataset("glue", DATASET_NAME)
    print("DATASET CARICATO:", ds.keys())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        # Le colonne corrette per QQP sono 'question1' e 'question2'
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

    # Tokenizza tutte le suddivisioni
    tokenized = ds.map(preprocess, batched=True, remove_columns=["idx", "question1", "question2", "label"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    labels = ds["train"]["label"]
    print("VALORI UNICI DELLE ETICHETTE:", set(labels))
    return tokenized

if __name__ == '__main__':
    dataset = get_datasets()
    print(dataset)
