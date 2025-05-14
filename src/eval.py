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
from sklearn.metrics import accuracy_score, f1_score

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
    args = parser.parse_args()

    # Carica il tokenizer e il dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = get_datasets()
    test_loader = DataLoader(
        ds["validation"], batch_size=args.batch_size,
        shuffle=False, collate_fn=default_data_collator
    )
    print("✔️  Dataset validation caricato:", len(ds["validation"]), "esempi")

    # Decido se è un file .pth o una cartella
    ckpt = args.checkpoint
    base_model = get_model()
    if os.path.isdir(ckpt):
        # Modalità LoRA adapter
        print("⚙️  Carico LoRA adapter da:", ckpt)
        model = PeftModel.from_pretrained(base_model, ckpt, safe_serialization=True)
    else:
        # Modalità Cross‑Encoder .pth
        print("⚙️  Carico Cross‑Encoder state_dict da:", ckpt)
        model = base_model
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)

    model.to(DEVICE)
    print("✔️  Modello pronto, inizio evaluation…")
    evaluate(model, test_loader)

if __name__ == "__main__":
    main()
