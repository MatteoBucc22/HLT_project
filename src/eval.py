# src/eval.py
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
from peft import PeftModel
from data_loader import get_datasets
from model import get_model
from config import DEVICE, BATCH_SIZE, MODEL_NAME
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Validation Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Eval LoRA adapter")
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Cartella con adapter_model.safetensors e adapter_config.json"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE
    )
    args = parser.parse_args()

    print(">>> Device:", DEVICE)
    print(">>> Carico base model e adapter da", args.model_dir)
    base_model = get_model()
    model = PeftModel.from_pretrained(
        base_model,
        args.model_dir,
        safe_serialization=True  # forza safetensors
    )
    model.to(DEVICE)
    print(">>> Parametri addestrabili:", model.print_trainable_parameters())

    print(">>> Carico dataset di test")
    ds = get_datasets()
    print(">>> Dimensione test set:", len(ds["test"]))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    test_loader = DataLoader(
        ds["test"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=default_data_collator
    )

    print("🔎 Starting evaluation on test set...")
    evaluate(model, test_loader)

if __name__ == "__main__":
    main()
