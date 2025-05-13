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

from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter

from data_loader import get_datasets, MAX_LENGTH  # assume MAX_LENGTH used inside
from model import get_model, MODEL_NAME
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, SAVE_DIR, DATASET_NAME, WEIGHT_DECAY, WARMUP_STEPS, PATIENCE, GRAD_CLIP_NORM
from hf_utils import save_to_hf

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
datasets = get_datasets(max_length=MAX_LENGTH)
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

# Loss will come from model outputs (CrossEntropy for SEQ_CLS)
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

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        loop.set_postfix(loss=train_loss / (batch_idx + 1), lr=scheduler.get_last_lr()[0])

    avg_train_loss = train_loss / len(train_loader)
    epoch_time = time.time() - start_time
    print(f"📈 Epoch {epoch} — Train Loss: {avg_train_loss:.4f} — Time: {epoch_time:.1f}s")
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

    # -------------------------
    #  Validation
    # -------------------------
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=1)
            val_preds.extend(preds.cpu().tolist())
            val_labels.extend(batch['labels'].cpu().tolist())

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
    # Save best model
    if f1 > best_f1:
        best_f1 = f1
        epochs_no_improve = 0
        best_dir = os.path.join(SAVE_DIR, 'best_model')
        os.makedirs(best_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, 'module') else model
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
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(adapter_dir)
        save_to_hf(adapter_dir,
                   repo_id=f"MatteoBucc/passphrase-identification-{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}-epoch-{epoch}")
        print(f"🔖 LoRA adapter epoch {epoch} saved at {adapter_dir}")

# -----------------------------
#  Final Save
# -----------------------------
final_dir = os.path.join(SAVE_DIR, f"lora_final_{ts}")
os.makedirs(final_dir, exist_ok=True)
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(final_dir)
torch.save(model.state_dict(), os.path.join(final_dir, f"model_full_{ts}.pth"))
print(f"✔️ Final LoRA adapter and full model saved at {final_dir}")
save_to_hf(final_dir,
           repo_id=f"MatteoBucc/passphrase-identification-{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}-final")

writer.close()
