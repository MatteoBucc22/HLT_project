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
from .config_miniLM_qqp import (
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
from .model_miniLM_qqp import get_model
from .data_loader_miniLM_qqp import get_datasets
from hf_utils import save_to_hf


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def train(resume_from=None, start_epoch=0):
    set_seed()
    
        
    dataset = get_datasets()
    base_model = get_model().to(DEVICE)

    if resume_from and os.path.isdir(resume_from):
        print(f"Caricamento modello da checkpoint: {resume_from}")
        model = PeftModel.from_pretrained(base_model, resume_from)
    else:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=16,                             
            lora_alpha=32,                     
            lora_dropout=0.1,                  
            target_modules=["query", "value", "key", "dense"]  
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

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=TOTAL_TRAIN_STEPS
    )

    scaler = torch.cuda.amp.GradScaler()
    global_step = 0
    
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0

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
        print(f"🧪 Validation — Acc: {acc:.4f} | F1: {f1:.4f}")
        
        if acc > best_val_acc:
            best_val_acc = acc
            patience_counter = 0
            print(f"Nuova migliore accuracy: {acc:.4f} - Modello salvato!")

            best_model_dir = os.path.join(SAVE_DIR, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping! Migliore accuracy: {best_val_acc:.4f}")
                break
        
        model.train()

        if global_step >= TOTAL_TRAIN_STEPS:
            break

        if (epoch + 1) % 2 == 0:
            ckpt_dir = os.path.join(SAVE_DIR, f"{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}_ep{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            save_to_hf(ckpt_dir, repo_id=f"{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}-ep{epoch+1}")  
            print(f"✔️ Checkpoint epoch {epoch+1} salvato localmente in {ckpt_dir}")
    

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_dir = os.path.join(SAVE_DIR, f"{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}_final_{ts}")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    torch.save(model.state_dict(), os.path.join(final_dir, "model_state.pth"))
    print(f"💾 Modello finale salvato in: {final_dir}")
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    args = parser.parse_args()
    train(resume_from=args.resume_from, start_epoch=args.start_epoch)