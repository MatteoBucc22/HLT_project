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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
from tqdm.auto import tqdm

def evaluate(model, dataloader, dataset_name="validation"):
    """Valuta il modello con metriche dettagliate"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    
    with torch.no_grad():
        eval_loop = tqdm(dataloader, desc=f"Evaluating {dataset_name}")
        for batch in eval_loop:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)

            # Accumula la loss se disponibile
            if outputs.loss is not None:
                total_loss += outputs.loss.item()
            
            # Predictions e probabilità
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
            all_probs.extend(probs.cpu().numpy())
    
    # Calcola metriche dettagliate
    avg_loss = total_loss / len(dataloader) if total_loss > 0 else 0
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    print(f"\n📊 {dataset_name.upper()} RESULTS:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Combined Score: {0.7 * acc + 0.3 * f1:.4f}")
    
    # Classification report dettagliato
    print(f"\n📈 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Not Paraphrase', 'Paraphrase']))
    
    return {
        'loss': avg_loss,
        'accuracy': acc,
        'f1': f1,
        'combined_score': 0.7 * acc + 0.3 * f1,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': np.array(all_probs)
    }

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
        ds["test"], batch_size=args.batch_size,
        shuffle=False, collate_fn=default_data_collator
    )
    print("✔️  Dataset test caricato:", len(ds["test"]), "esempi")

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
