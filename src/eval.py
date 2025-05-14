# src/eval.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    default_data_collator,
    AutoModelForSequenceClassification
)
from data_loader import get_datasets
from model import get_model
from config import DEVICE, BATCH_SIZE, MODEL_NAME
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
    cm = confusion_matrix(all_labels, all_preds)

    print(f"✅ Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print("📊 Confusion Matrix:\n", cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path al file .pth del modello o alla cartella contenente model.safetensors/config.json"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE
    )
    args = parser.parse_args()

    # prepara test set
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = get_datasets()
    test_loader = DataLoader(
        ds["validation"],  # o 'test' se preferisci
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=default_data_collator
    )
    print(f"✔️  Caricati {len(ds['validation'])} esempi di valutazione")

    # carica il modello
    if os.path.isdir(args.checkpoint):
        # cartella HuggingFace-style (model.safetensors, config.json, etc.)
        print("⚙️  Carico modello da directory:", args.checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.checkpoint,
            device_map={"": DEVICE} if DEVICE != "cpu" else None,
            local_files_only=True
        )
    elif args.checkpoint.endswith(".pth"):
        # file state_dict
        print("⚙️  Carico state_dict da file:", args.checkpoint)
        model = get_model()
        state = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        raise ValueError("Checkpoint non riconosciuto: fornisci un .pth o una directory di modello HF")

    model.to(DEVICE)
    print("✔️  Modello pronto, inizio evaluation…")
    evaluate(model, test_loader)


if __name__ == "__main__":
    main()
