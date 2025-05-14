import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import default_data_collator, get_scheduler, RobertaConfig, RobertaForSequenceClassification
from tqdm.auto import tqdm
import time
import datetime
from sklearn.metrics import accuracy_score, f1_score

from peft import get_peft_model, LoraConfig, TaskType

from data_loader import get_datasets
from model import get_model, MODEL_NAME
from config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    SAVE_DIR,
    DATASET_NAME,
    SEED,
    LR_SCHEDULER,
    WARMUP_RATIO,
    WARMUP_STEPS,
    HIDDEN_DROPOUT,
    LORA_DROPOUT,
    LOGGING_STEPS,
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
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔍 Generating Embeddings"):
            labels = batch["labels"]
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model.base_model(**batch, output_hidden_states=True, return_dict=True)
            cls_emb = outputs.hidden_states[-1][:, 0, :]
            all_embeddings.append(cls_emb.cpu())
            all_labels.extend(labels)

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.tensor(all_labels)
    os.makedirs(save_path, exist_ok=True)
    torch.save(
        {"embeddings": all_embeddings, "labels": all_labels},
        os.path.join(save_path, "validation_embeddings.pt")
    )
    print(f"💾 Embeddings salvati in: {save_path}/validation_embeddings.pt")


def train():
    set_seed(SEED)

    # Carica dataset
    dataset = get_datasets()

    # Overwrite dropout interno di RoBERTa
    base_cfg = RobertaConfig.from_pretrained(
        MODEL_NAME,
        hidden_dropout_prob=HIDDEN_DROPOUT,
        attention_probs_dropout_prob=HIDDEN_DROPOUT
    )
    base_model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=base_cfg)
    base_model.to(DEVICE)

    # Config LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query", "value"]
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # DataLoader
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

    # Optimizer con weight decay
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Scheduler
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_STEPS if WARMUP_STEPS is not None else int(WARMUP_RATIO * total_steps)
    scheduler = get_scheduler(
        name=LR_SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    use_amp = True
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    safe_name = MODEL_NAME.replace("/", "-")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for step, batch in enumerate(loop, 1):
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

            scheduler.step()
            running_loss += loss.item()

            if LOGGING_STEPS and step % LOGGING_STEPS == 0:
                lr = scheduler.get_last_lr()[0]
                avg = running_loss / step
                print(f"  Step {step:4d} — loss: {avg:.4f} — lr: {lr:.2e}")

            loop.set_postfix(loss=running_loss / step)

        epoch_time = time.time() - t0
        avg_loss = running_loss / len(train_loader)
        print(f"\n🏷️  Epoch {epoch} — Train Loss: {avg_loss:.4f} — Time: {epoch_time:.1f}s")

        # Validation
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                out = model(**batch)
                p = out.logits.argmax(dim=1)
                preds.extend(p.cpu().tolist())
                labels.extend(batch["labels"].cpu().tolist())

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        print(f"🧪 Validation — Acc: {acc:.4f} | F1: {f1:.4f}\n")

        # Salvataggio checkpoint LoRA
        if epoch % 2 == 0:
            ckpt_dir = os.path.join(SAVE_DIR, f"{safe_name}-{DATASET_NAME}_ep{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            print(f"✔️  LoRA adapter salvato in: {ckpt_dir}")
            save_to_hf(ckpt_dir, repo_id=f"MatteoBucc/{safe_name}-{DATASET_NAME}-ep{epoch}")

    # Salvataggio finale
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_dir = os.path.join(SAVE_DIR, f"{safe_name}-{DATASET_NAME}_lora_{timestamp}")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    print(f"✔️  LoRA finale salvato in: {final_dir}")

    # Full model .pth
    pth_file = f"{safe_name}-{DATASET_NAME}_full_{timestamp}.pth"
    pth_path = os.path.join(SAVE_DIR, pth_file)
    torch.save(model.state_dict(), pth_path)
    print(f"✔️  State dict salvato in: {pth_path}")

    save_to_hf(final_dir, repo_id=f"MatteoBucc/{safe_name}-{DATASET_NAME}-final")

    # Embeddings per ensemble
    generate_embeddings(model, val_loader, save_path=final_dir)


if __name__ == "__main__":
    train()
