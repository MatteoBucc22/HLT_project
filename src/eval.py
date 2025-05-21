# src/eval.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
from peft import PeftModel
from data_loader import get_datasets
from model import get_model
from config import DEVICE, BATCH_SIZE, MODEL_NAME
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.util import cos_sim
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset

def evaluate(model, dataloader):

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)

            print("Logistic shape:", outputs.logits.shape)
            print("Labels dtype:", batch["labels"].dtype)
            print("Unique labels:", torch.unique(batch["labels"]))
            
            preds = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
    print(f"Validation Accuracy: {accuracy_score(all_labels, all_preds):.4f} | "
          f"F1 Score: {f1_score(all_labels, all_preds):.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path al file .pth (cross‑encoder) o alla cartella LORA adapter")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dataset_name", type=str, default="qqp")
    parser.add_argument("--element_name", type=str, default="question")
    args = parser.parse_args()
    
    dataset = load_dataset("glue", args.dataset_name)
    # Format evaluation data
    sentences1 = []
    sentences2 = []
    scores = []
    for example in dataset['validation']:
        sentences1.append(example[args.element_name+'1'])
        sentences2.append(example[args.element_name+'2'])
        scores.append(float(example['label']))
        if (example['label'] != 0 and example['label'] != 1): print(example)
    
    evaluator = BinaryClassificationEvaluator(sentences1, sentences2, scores)

    model = SentenceTransformer.load(args.checkpoint)
    
    results = evaluator(model)

    print(results)


if __name__ == "__main__":
    main()
