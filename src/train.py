import os
import random
import numpy as np
import torch, time
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import default_data_collator, get_scheduler
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    EPOCHS, SAVE_DIR, DATASET_NAME, SEED,
    MAX_LENGTH, WARMUP_RATIO, WARMUP_STEPS,
    LR_SCHEDULER, LOGGING_STEPS,
    ACCUM_STEPS, LABEL_SMOOTHING,
    EARLY_STOPPING_PATIENCE, NUM_WORKERS, PIN_MEMORY, MODEL_NAME, HF_REPO_PREFIX
)
from data_loader import get_datasets
from model import get_model
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
            cls_emb = outputs.hidden_states[-1][:, 0, :]
            all_embeddings.append(cls_emb.cpu())
            all_labels.extend(labels)
    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.tensor(all_labels)
    os.makedirs(save_path, exist_ok=True)
    torch.save({"embeddings": all_embeddings, "labels": all_labels},
               os.path.join(save_path, "validation_embeddings.pt"))
    print(f"💾 Embedding salvati in: {save_path}")


def train():
    set_seed(SEED)
    dataset = get_datasets()
    model = get_model(hidden_dropout_prob=0.2).to(DEVICE)

    train_loader = DataLoader(
        dataset['train'], batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=default_data_collator
    )
    val_loader = DataLoader(
        dataset['validation'], batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=default_data_collator
    )

    optimizer = AdamW(
        model.parameters(), lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = (WARMUP_STEPS if WARMUP_STEPS is not None
                    else int(WARMUP_RATIO * total_steps))

    scheduler = get_scheduler(
        LR_SCHEDULER, optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    loss_fn = CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    best_val_loss = float('inf')
    no_improve = 0
    safe_name = MODEL_NAME.replace('/', '-')
    best_dir = os.path.join(SAVE_DIR, f"{safe_name}-{DATASET_NAME}-best")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        start = time.time()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(loop):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                logits = outputs.logits
                loss = loss_fn(logits, batch['labels']) / ACCUM_STEPS
            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUM_STEPS
            if (step + 1) % LOGGING_STEPS == 0:
                loop.set_postfix(loss=total_loss / ((step+1)*BATCH_SIZE))

        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} — Train Loss: {avg_train_loss:.4f} — Time: {time.time()-start:.1f}s")

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                val_loss += loss_fn(logits, batch['labels']).item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch['labels'].cpu().tolist())

        avg_val_loss = val_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(f"🧪 Val — Loss: {avg_val_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}\n")

        # Early stopping & best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOPPING_PATIENCE:
                print("⏹ Early stopping")
                break

        if acc > best_acc:
            best_acc = acc
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            save_to_hf(best_dir, repo_id=f"{HF_REPO_PREFIX}-{safe_name}-{DATASET_NAME}-best")
            print(f"💾 Best model saved: {best_dir} (acc: {acc:.4f})")

    # Embeddings finali
    generate_embeddings(model, val_loader, save_path=best_dir)


if __name__ == '__main__':
    train()