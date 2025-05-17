import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import default_data_collator, get_scheduler
from tqdm.auto import tqdm
import time
from sklearn.metrics import accuracy_score, f1_score

from data_loader import get_datasets
from model import get_model, MODEL_NAME
from config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    EPOCHS, SAVE_DIR, DATASET_NAME, SEED,
    MAX_LENGTH, WARMUP_RATIO, WARMUP_STEPS,
    LR_SCHEDULER, LOGGING_STEPS, HIDDEN_DROPOUT
)
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

    # Carica dataset e modello
    dataset = get_datasets(tokenizer_max_length=MAX_LENGTH)
    model = get_model(
        model_name=MODEL_NAME,
        hidden_dropout_prob=HIDDEN_DROPOUT
    ).to(DEVICE)

    train_loader = DataLoader(
        dataset['train'], batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True,
        collate_fn=default_data_collator
    )
    val_loader = DataLoader(
        dataset['validation'], batch_size=BATCH_SIZE,
        num_workers=4, pin_memory=True,
        collate_fn=default_data_collator
    )

    # Ottimizzatore con weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Calcolo warmup steps
    total_steps = EPOCHS * len(train_loader)
    if WARMUP_STEPS is None:
        warmup_steps = int(WARMUP_RATIO * total_steps)
    else:
        warmup_steps = WARMUP_STEPS

    # Scheduler: linear o cosine
    scheduler = get_scheduler(
        LR_SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Gradient accumulation per simulare batch più grandi
    accum_steps = 2
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0
    safe_name = MODEL_NAME.replace('/', '-')
    best_dir = os.path.join(SAVE_DIR, f"{safe_name}-{DATASET_NAME}-best")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}", unit='batch')

        optimizer.zero_grad()
        for step, batch in enumerate(loop):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            if (step + 1) % LOGGING_STEPS == 0:
                loop.set_postfix(
                    loss=total_loss / ((step+1) * BATCH_SIZE)
                )

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} — Avg Train Loss: {avg_loss:.4f} — Time: {time.time() - start_time:.1f}s")

        # Validazione
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch['labels'].cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(f"🧪 Validation — Accuracy: {acc:.4f} | F1 Score: {f1:.4f}\n")

        # Salvataggio best model
        if acc > best_acc:
            best_acc = acc
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            print(f"💾 Miglior modello salvato in: {best_dir} (acc: {acc:.4f})")
            save_to_hf(
                best_dir,
                repo_id=f"MatteoBucc/passphrase-identification-{safe_name}-{DATASET_NAME}-best"
            )

    # Embeddings finali
    generate_embeddings(model, val_loader, save_path=best_dir)

if __name__ == '__main__':
    train()
