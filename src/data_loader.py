# src/data_loader.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import InputExample
from config import MODEL_NAME, DATASET_NAME, MAX_LENGTH

def get_examples():
    ds = load_dataset("glue", DATASET_NAME)
    print("DATASET SPLITS:", ds.keys())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def create_examples(split):
        examples = []
        for item in ds[split]:
            q1 = item["question1"]
            q2 = item["question2"]
            label = float(item["label"])  
            examples.append(InputExample(texts=[q1, q2], label=label))
        return examples

    train_examples = create_examples("train")
    dev_examples   = create_examples("validation")
    print(f"Loaded {len(train_examples)} train and {len(dev_examples)} dev examples.")
    return train_examples, dev_examples

if __name__ == '__main__':
    get_examples()