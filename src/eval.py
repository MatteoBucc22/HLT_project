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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt  # per visualizzazione

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

    print(f"✅ Validation Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
    print("📊 Confusion Matrix:")
    print(cm)

    # Visualizzazione grafica opzionale
    try:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()
    except Exception as e:
        print("⚠️ Impossibile visualizzare la matrice di confusione:", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path al file .pth (cross‑encoder) o alla cartella LORA adapter")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # Carica tokenizer e dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = get_datasets()
    test_loader = DataLoader(
        ds["test"], batch_size=args.batch_size,
        shuffle=False, collate_fn=default_data_collator
    )
    print("✔️  Dataset test caricato:", len(ds["test"]), "esempi")

    # Caricamento modello
    ckpt = args.checkpoint
    base_model = get_model()
    if os.path.isdir(ckpt):
        print("⚙️  Carico LoRA adapter da:", ckpt)
        model = PeftModel.from_pretrained(base_model, ckpt, safe_serialization=True)
    else:
        print("⚙️  Carico Cross‑Encoder state_dict da:", ckpt)
        model = base_model
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)

    model.to(DEVICE)
    print("✔️  Modello pronto, inizio evaluation…")
    evaluate(model, test_loader)


if __name__ == "__main__":
    main()
