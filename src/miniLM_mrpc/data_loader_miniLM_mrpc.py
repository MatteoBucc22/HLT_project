import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import InputExample
from config import MODEL_NAME, DATASET_NAME, MAX_LENGTH

def get_datasets():
    ds = load_dataset("glue", DATASET_NAME)
    print("DATASET SPLITS:", ds.keys())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


    def preprocess(examples):
        q1 = examples["sentence1"]
        q2 = examples["sentence2"]
        
        # Tokenizzazione ottimizzata per sequence classification
        tok = tokenizer(
            q1,
            q2,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_token_type_ids=True,  # Importante per distinguere le due frasi
            add_special_tokens=True,
        )
        tok["labels"] = examples["label"]
        return tok

    tokenized = ds.map(preprocess, batched=True, remove_columns=["idx", "sentence1", "sentence2"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
    labels = ds["train"]["label"]
    print("VALORI UNICI DELLE ETICHETTE:", set(labels))
    return tokenized

if __name__ == '__main__':
    get_datasets()