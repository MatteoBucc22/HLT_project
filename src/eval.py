# src/eval.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    default_data_collator
)
from peft import PeftModel
from data_loader import get_datasets
from model import get_model, MODEL_NAME
from config import DEVICE, BATCH_SIZE
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, digits=4))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(all_labels, all_preds))
    print(f"Overall Acc: {accuracy_score(all_labels, all_preds):.4f} | "
          f"F1 Score: {f1_score(all_labels, all_preds):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate paraphrase model")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to .pth file, LoRA adapter folder, HF model folder, or HF hub ID"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help="Batch size for evaluation"
    )
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Prepare test dataset
    ds = get_datasets()
    test_loader = DataLoader(
        ds["test"], batch_size=args.batch_size,
        shuffle=False, collate_fn=default_data_collator
    )
    print(f"✔️  Loaded test set: {len(ds['test'])} examples")

    # Determine model loading strategy
    ckpt = args.checkpoint
    base_model = get_model()

    if os.path.isdir(ckpt):
        # Directory: decide se è LoRA adapter (contiene adapter_config.json) oppure modello completo
        files = set(os.listdir(ckpt))
        if "adapter_config.json" in files:
            print(f"⚙️  Loading LoRA adapter from {ckpt}")
            model = PeftModel.from_pretrained(
                base_model, ckpt, safe_serialization=True
            )
        else:
            print(f"⚙️  Loading full model from local folder {ckpt}")
            model = AutoModelForSequenceClassification.from_pretrained(
                ckpt, local_files_only=True
            )
    elif ckpt.endswith(".pth"):

            print(f"⚙️  Loading full model from local folder {ckpt}")
            model = AutoModelForSequenceClassification.from_pretrained(
                ckpt, local_files_only=True
            )
    elif ckpt.endswith(".pth"):
        print(f"⚙️  Loading cross-encoder state_dict from {ckpt}")
        model = base_model
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        print(f"⚙️  Loading Hugging Face hub model '{ckpt}'")
        model = AutoModelForSequenceClassification.from_pretrained(ckpt)

    model.to(DEVICE)
    print("✔️  Model ready, starting evaluation...")
    evaluate(model, test_loader)


if __name__ == "__main__":
    main()
