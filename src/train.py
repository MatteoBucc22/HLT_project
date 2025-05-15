import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed
)
from tqdm.auto import tqdm
import time
import datetime
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter

from data_loader import get_datasets, MAX_LENGTH  # assume MAX_LENGTH used inside
from model import get_model, MODEL_NAME
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, SAVE_DIR, DATASET_NAME, WEIGHT_DECAY, WARMUP_STEPS, PATIENCE, GRAD_CLIP_NORM
from hf_utils import save_to_hf


def generate_embeddings(model, dataloader, save_path):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔍 Generating Embeddings"):
            labels = batch["labels"]
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model.base_model(**batch, output_hidden_states=True, return_dict=True)

            # Usa il [CLS] token embedding dall'ultimo hidden state
            cls_embeddings = outputs.hidden_states[-1][:, 0, :]  # [batch_size, hidden_dim]
            all_embeddings.append(cls_embeddings.cpu())
            all_labels.extend(labels)

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.tensor(all_labels)

    os.makedirs(save_path, exist_ok=True)
    torch.save(
        {"embeddings": all_embeddings, "labels": all_labels},
        os.path.join(save_path, "validation_embeddings.pt")
    )
    print(f"💾 Embedding di validazione salvati in: {save_path}/validation_embeddings.pt")


def train():
    dataset = get_datasets()
    base_model = get_model().to(DEVICE)
# -----------------------------
#  Setting seeds for reproducibility
# -----------------------------
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------------
#  Initialize TensorBoard
# -----------------------------
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, 'runs', ts))

# -----------------------------
#  Load Data
# -----------------------------
print("🔄 Loading datasets...")
datasets = get_datasets()
train_dataset = datasets['train']
val_dataset = datasets['validation']
print(f" » Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

dataloader_args = dict(
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True,
    collate_fn=default_data_collator
)
train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_args)
val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_args)

# -----------------------------
#  Load Base Model and apply LoRA
# -----------------------------
print(f"🔄 Loading base model: {MODEL_NAME}")
base_model = get_model().to(DEVICE)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# Multi-GPU support
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# -----------------------------
#  Prepare optimizer, scheduler, and loss
# -----------------------------
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps
)

# Early stopping
best_f1 = 0.0
epochs_no_improve = 0

# -----------------------------
#  Training Loop
# -----------------------------
for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    model.train()
    train_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
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

    acc = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds)
    precision = precision_score(val_labels, val_preds)
    recall = recall_score(val_labels, val_preds)

    print(f"🧪 Val — Acc: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}\n")
    writer.add_scalar('Metrics/val_accuracy', acc, epoch)
    writer.add_scalar('Metrics/val_f1', f1, epoch)
    writer.add_scalar('Metrics/val_precision', precision, epoch)
    writer.add_scalar('Metrics/val_recall', recall, epoch)

    # -------------------------
    #  Checkpoint & Early Stopping
    # -------------------------
    model_to_save = model.module if hasattr(model, 'module') else model
    if f1 > best_f1:
        best_f1 = f1
        epochs_no_improve = 0
        best_dir = os.path.join(SAVE_DIR, 'best_model')
        os.makedirs(best_dir, exist_ok=True)
        model_to_save.save_pretrained(best_dir)
        print(f"✔️ Best model saved with F1: {best_f1:.4f} at {best_dir}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"⏸️ Early stopping triggered. No improvement in {PATIENCE} epochs.")
            break

    # Save LoRA adapter every 2 epochs
    if epoch % 2 == 0:
        adapter_dir = os.path.join(SAVE_DIR, f"lora_epoch_{epoch}")
        os.makedirs(adapter_dir, exist_ok=True)
        model_to_save.save_pretrained(adapter_dir)
        save_to_hf(adapter_dir,
                   repo_id=f"MatteoBucc/passphrase-identification-{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}-epoch-{epoch}")
        print(f"🔖 LoRA adapter epoch {epoch} saved at {adapter_dir}")

# -----------------------------
#  Final Save
# -----------------------------
final_dir = os.path.join(SAVE_DIR, f"lora_final_{ts}")
# ensure directory exists
os.makedirs(final_dir, exist_ok=True)
model_to_save.save_pretrained(final_dir)
torch.save(model.state_dict(), os.path.join(final_dir, f"model_full_{ts}.pth"))
print(f"✔️ Final LoRA adapter and full model saved at {final_dir}")
save_to_hf(final_dir,
           repo_id=f"MatteoBucc/passphrase-identification-{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}-final")

    # ⏬ Salva embeddings per l'ensemble
    generate_embeddings(model, val_loader, save_path=adapter_dir_final)

writer.close()

