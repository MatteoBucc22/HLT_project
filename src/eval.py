# src/eval.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
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
        help="Path al file .pth del modello fine-tuned"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE
    )
    args = parser.parse_args()

    # tokenizer + dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = get_datasets()
    test_loader = DataLoader(
        ds["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=default_data_collator
    )
    print(f"✔️  Caricati {len(ds['validation'])} esempi di test")

    # modello base + caricamento weights
    model = get_model()
    if args.checkpoint.endswith(".pth"):
        state = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_state_dict(state)
        print("⚙️  Caricato state_dict da:", args.checkpoint)
    else:
        raise ValueError("Il checkpoint deve essere un file .pth")

    print("✔️  Modello pronto, inizio evaluation…")
    evaluate(model, test_loader)


if __name__ == "__main__":
    main()
