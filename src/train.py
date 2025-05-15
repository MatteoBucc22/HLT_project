import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import default_data_collator, get_scheduler
from tqdm.auto import tqdm
import time
import datetime
from sklearn.metrics import accuracy_score, f1_score
from data_loader import get_datasets
from model import get_model, MODEL_NAME
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, SAVE_DIR, DATASET_NAME, SEED
from hf_utils import save_to_hf

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_embeddings(model, dataloader, save_path):
    model.eval()
    all_embeddings, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔍 Generating Embeddings"):
            labels = batch["labels"]
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            cls_embeddings = outputs.hidden_states[-1][:, 0, :]
            all_embeddings.append(cls_embeddings.cpu())
            all_labels.extend(labels)

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.tensor(all_labels)

    os.makedirs(save_path, exist_ok=True)
    torch.save({"embeddings": all_embeddings, "labels": all_labels},
               os.path.join(save_path, "validation_embeddings.pt"))
    print(f"💾 Embedding di validazione salvati in: {save_path}/validation_embeddings.pt")

def train():
    set_seed(SEED)

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

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = EPOCHS * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0
    safe_model_name = MODEL_NAME.replace("/", "-")
    best_model_dir = os.path.join(SAVE_DIR, f"{safe_model_name}-{DATASET_NAME}-best")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        start = time.time()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
        for batch_idx, batch in enumerate(loop):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (batch_idx + 1))

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} — Avg Train Loss: {avg_loss:.4f} — Time: {time.time() - start:.1f}s")

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
        print(f"🧪 Validation — Accuracy: {acc:.4f} | F1 Score: {f1:.4f}\n")

        if acc > best_acc:
            best_acc = acc
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            print(f"💾 Miglior modello salvato in: {best_model_dir} con acc: {acc:.4f}")
            save_to_hf(
                best_model_dir,
                repo_id=(
                    f"MatteoBucc/passphrase-identification-"
                    f"{safe_model_name}-{DATASET_NAME}-best"
                )
            )

    # Final embeddings
    generate_embeddings(model, val_loader, save_path=best_model_dir)

if __name__ == "__main__":
    train()