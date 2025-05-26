import os
import torch
import time
import datetime
import numpy as np
import random

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    SAVE_DIR,
    DATASET_NAME,
    MODEL_NAME,
    TOTAL_TRAIN_STEPS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    GRAD_CLIP_NORM
)
from model import get_model
from data_loader import get_datasets
from hf_utils import save_to_hf

# reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_embeddings(model, dataloader, save_path, repo_id=None):
    model.eval()
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔍 Generating Embeddings"):
            labels = batch.pop("labels")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model.base_model(**batch, output_hidden_states=True, return_dict=True)
            cls_emb = outputs.hidden_states[-1][:, 0, :]
            all_embeddings.append(cls_emb.cpu())
            all_labels.extend(labels.tolist())

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.tensor(all_labels)
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, "validation_embeddings.pt")
    torch.save({"embeddings": all_embeddings, "labels": all_labels}, file_path)
    print(f"💾 Embedding di validazione salvati in: {file_path}")

    if repo_id:
        print(f"⏫ Caricamento embeddings su Hugging Face: {repo_id}")
        # save_to_hf(save_path, repo_id=repo_id)  # Commentato: disabilitato upload HF
        print("✔️ Embeddings salvati localmente (upload HF disabilitato)")


def train(resume_from=None, start_epoch=0):
    set_seed()
    
    # Verifico che il device sia disponibile
    print(f"🖥️  Usando device: {DEVICE}")
    if DEVICE == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA non disponibile, switching to CPU")
        
    dataset = get_datasets()
    base_model = get_model().to(DEVICE)

    # load or init PEFT model
    if resume_from and os.path.isdir(resume_from):
        print(f"📦 Caricamento modello da checkpoint: {resume_from}")
        model = PeftModel.from_pretrained(base_model, resume_from)
    else:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            target_modules=["query", "value"]
        )
        model = get_peft_model(base_model, peft_config)

    model.to(DEVICE)
    model.print_trainable_parameters()

    train_loader = DataLoader(
        dataset['train'], batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=default_data_collator
    )
    val_loader = DataLoader(
        dataset['validation'], batch_size=BATCH_SIZE,
        num_workers=4, pin_memory=True, collate_fn=default_data_collator
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY
    )

    # total steps and scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=TOTAL_TRAIN_STEPS
    )

    scaler = torch.cuda.amp.GradScaler()
    global_step = 0

    model.train()
    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0.0
        start_time = time.time()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()

            # gradient clipping
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step >= TOTAL_TRAIN_STEPS:
                break

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch+1} — Avg Loss: {avg_loss:.4f} — Time: {elapsed:.1f}s")

        # validation
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                out = model(**batch)
                pred = out.logits.argmax(dim=1)
                preds.extend(pred.cpu().tolist())
                labels.extend(batch['labels'].cpu().tolist())
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        print(f"🧪 Validation — Acc: {acc:.4f} | F1: {f1:.4f}\n")
        model.train()

        if global_step >= TOTAL_TRAIN_STEPS:
            break

        # checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            ckpt_dir = os.path.join(SAVE_DIR, f"{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}_ep{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            # save_to_hf(ckpt_dir, repo_id=f"{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}-ep{epoch+1}")  # Commentato: disabilitato upload HF
            print(f"✔️ Checkpoint epoch {epoch+1} salvato localmente in {ckpt_dir}")
    
    # final save
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_dir = os.path.join(SAVE_DIR, f"{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}_final_{ts}")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    torch.save(model.state_dict(), os.path.join(final_dir, "model_state.pth"))
    print(f"💾 Modello finale salvato in: {final_dir}")
    
    # Salvataggio embedding senza upload HF
    # generate_embeddings(model, val_loader, final_dir, repo_id=f"{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}-embeddings-{ts}")  # Commentato: disabilitato upload HF
    generate_embeddings(model, val_loader, final_dir, repo_id=None)  # Solo salvataggio locale
    print("✔️ Training completato! Modello e embeddings salvati localmente.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    args = parser.parse_args()
    train(resume_from=args.resume_from, start_epoch=args.start_epoch)
