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
    model_name_clean = MODEL_NAME.replace("/", "-")
    dataset = get_datasets()
    model = get_model().to(DEVICE)

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

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    use_amp = True
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    losses, accuracies, f1_scores = [], [], []

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

        if (epoch + 1) % 2 == 0:
            model_dir_epoch = os.path.join(SAVE_DIR, f"{model_name_clean}-{DATASET_NAME}_epoch_{epoch+1}")
            os.makedirs(model_dir_epoch, exist_ok=True)
            model.save_pretrained(model_dir_epoch)
            print(f"✔️  Modello (epoch {epoch+1}) salvato in: {model_dir_epoch}")
            save_to_hf(model_dir_epoch, repo_id=f"MatteoBucc/passphrase-identification-{model_name_clean}-{DATASET_NAME}-epoch-{epoch+1}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir_final = os.path.join(SAVE_DIR, f"{model_name_clean}-{DATASET_NAME}_finetuned_{ts}")
    os.makedirs(model_dir_final, exist_ok=True)

    model.save_pretrained(model_dir_final)
    print(f"✔️  Modello fine-tuned finale salvato in: {model_dir_final}")

    pth_name = f"{model_name_clean}-{DATASET_NAME}_cross_encoder_fullfinetune_{ts}.pth"
    pth_path = os.path.join(SAVE_DIR, pth_name)
    torch.save(model.state_dict(), pth_path)
    print(f"✔️ Modello salvato come stato PyTorch in: {pth_path}")

    save_to_hf(model_dir_final, repo_id=f"MatteoBucc/passphrase-identification-{model_name_clean}-{DATASET_NAME}-final")

    # 🎨 Plot dei grafici
    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, losses, label="Train Loss", color="blue", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    # Accuracy + F1
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, accuracies, label="Val Accuracy", color="green", marker="o")
    plt.plot(epochs_range, f1_scores, label="Val F1 Score", color="orange", marker="x")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Accuracy / F1")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, f"{model_name_clean}-{DATASET_NAME}_metrics_plot.png")
    plt.savefig(plot_path)
    print(f"📊 Grafici salvati in {plot_path}")


if __name__ == "__main__":
    train()
