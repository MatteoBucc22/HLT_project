import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import default_data_collator
from tqdm.auto import tqdm
import time
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from data_loader import get_datasets
from model import get_model, MODEL_NAME
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, SAVE_DIR, DATASET_NAME
from hf_utils import save_to_hf


def train():
    # Pulizia del nome per il repo HF
    model_name_clean = MODEL_NAME.replace("/", "-")

    # Carica dataset e modello base
    dataset = get_datasets()
    model = get_model().to(DEVICE)

    # DataLoaders
    train_loader = DataLoader(
        dataset["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=default_data_collator
    )
    val_loader = DataLoader(
        dataset["validation"],
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        collate_fn=default_data_collator
    )

    # Ottimizzatore
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Mixed precision
    use_amp = True
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Metriche da tracciare
    losses, accuracies, f1_scores = [], [], []

    # Ciclo di training
    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
        for batch_idx, batch in enumerate(loop):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (batch_idx + 1))

        epoch_time = time.time() - start
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"\nEpoch {epoch+1} — Avg Train Loss: {avg_loss:.4f} — Time: {epoch_time:.1f}s")

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        accuracies.append(acc)
        f1_scores.append(f1)
        print(f"🧪 Validation — Accuracy: {acc:.4f} | F1 Score: {f1:.4f}\n")

        # Salvataggio checkpoint ogni 2 epoche
        if (epoch + 1) % 2 == 0:
            ckpt_dir = os.path.join(SAVE_DIR, f"{model_name_clean}-{DATASET_NAME}_epoch_{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            print(f"✔️ Modello (epoch {epoch+1}) salvato in: {ckpt_dir}")
            save_to_hf(
                ckpt_dir,
                repo_id=f"MatteoBucc/passphrase-identification-{model_name_clean}-{DATASET_NAME}-epoch-{epoch+1}"
            )

    # Directory finale
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_dir = os.path.join(SAVE_DIR, f"{model_name_clean}-{DATASET_NAME}_finetuned_{ts}")
    os.makedirs(final_dir, exist_ok=True)

    # Salvataggio del modello intero
    model.save_pretrained(final_dir)
    print(f"✔️ Modello fine-tuned salvato in: {final_dir}")

    # Upload full model su HF
    save_to_hf(
        final_dir,
        repo_id=f"MatteoBucc/passphrase-identification-{model_name_clean}-{DATASET_NAME}-final"
    )

    # Plottaggio metriche
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, losses, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, accuracies, marker="o", label="Acc")
    plt.plot(epochs_range, f1_scores, marker="x", label="F1")
    plt.title("Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, f"{model_name_clean}-{DATASET_NAME}_metrics.png")
    plt.savefig(plot_path)
    print(f"📊 Grafici salvati in {plot_path}")


if __name__ == "__main__":
    train()
