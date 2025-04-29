
from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH

def get_datasets():
    # 1) Carica e fai lo split
    raw = load_dataset("quora")
    ds = raw["train"].train_test_split(test_size=0.2)
    # rinominiamo 'test' per chiarezza
    ds["test"] = ds.pop("test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        # examples["questions"] è una lista di dict, ciascuno con "text": [q1, q2]
        q1 = [q["text"][0] for q in examples["questions"]]
        q2 = [q["text"][1] for q in examples["questions"]]

        # tokenizza le coppie
        tok = tokenizer(
            q1,
            q2,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        # converte boolean → int e lo aggiunge come colonna
        tok["labels"] = [int(x) for x in examples["is_duplicate"]]
        return tok

    # Applica la tokenizzazione e rimuovi le colonne annidate
    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds["train"].column_names  # toglie 'questions' e la vecchia 'is_duplicate'
    )

    return tokenized

