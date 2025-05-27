# train.py
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
from data_loader_miniLM_mrpc import get_datasets
from model_miniLM_mrpc import get_model, MODEL_NAME
from config_miniLM_mrpc import (DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, SAVE_DIR, DATASET_NAME, SEED,
                   WEIGHT_DECAY, GRADIENT_CLIPPING, WARMUP_RATIO, LR_SCHEDULER, PATIENCE, MIN_DELTA)
from hf_utils import save_to_hf  # NOTA: funzione modificata per salvataggio solo locale

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

    # Optimizer ottimizzato con weight decay
    optimizer = AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    num_training_steps = EPOCHS * len(train_loader)
    num_warmup_steps = int(WARMUP_RATIO * num_training_steps)

    scheduler = get_scheduler(
        LR_SCHEDULER,  # Usa il scheduler dalla configurazione
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    scaler = torch.cuda.amp.GradScaler()
    
    # Early stopping per migliore generalization
    best_acc = 0.0
    best_f1 = 0.0
    patience_counter = 0
    safe_model_name = MODEL_NAME.replace("/", "-")
    best_model_dir = os.path.join(SAVE_DIR, f"{safe_model_name}-{DATASET_NAME}-best")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        start = time.time()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for batch_idx, batch in enumerate(loop):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss

            # Gradient clipping per stabilità
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            
            # Progress bar con learning rate
            current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else LEARNING_RATE
            loop.set_postfix({
                'loss': f"{total_loss / (batch_idx + 1):.4f}",
                'lr': f"{current_lr:.2e}"
            })

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{EPOCHS} — Avg Train Loss: {avg_loss:.4f} — Time: {time.time() - start:.1f}s")

        # Validation con metriche dettagliate
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc="Validation", leave=False)
            for batch in val_loop:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                
                val_loss += outputs.loss.item() if outputs.loss is not None else 0
                preds = outputs.logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())

        avg_val_loss = val_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')
        
        print(f"🧪 Validation — Loss: {avg_val_loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
        # Early stopping basato su accuracy + F1
        combined_metric = 0.7 * acc + 0.3 * f1  # Peso maggiore all'accuracy
        best_combined = 0.7 * best_acc + 0.3 * best_f1
        
        if combined_metric > best_combined + MIN_DELTA:
            print(f"🏆 NUOVO MIGLIOR MODELLO! Acc: {acc:.4f} | F1: {f1:.4f} (combined: {combined_metric:.4f})")
            best_acc = acc
            best_f1 = f1
            patience_counter = 0
            
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            print(f"💾 Salvato in: {best_model_dir}")
            
            # NOTA: save_to_hf ora salva solo localmente (upload HF disabilitato)
            save_to_hf(
                best_model_dir,
                repo_id=(
                    f"MatteoBucc/passphrase-identification-"
                    f"{safe_model_name}-{DATASET_NAME}-best"
                )
            )
        else:
            patience_counter += 1
            print(f"📉 Nessun miglioramento. Patience: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print(f"🛑 Early stopping! Migliore accuracy: {best_acc:.4f} | F1: {best_f1:.4f}")
                break

    # Final embeddings per il modello migliore
    generate_embeddings(model, val_loader, save_path=best_model_dir)
    
    # ✅ SOLO IL MODELLO MIGLIORE È SALVATO
    print(f"🏆 Training completato! Miglior modello salvato in: {best_model_dir}")
    print(f"📊 Migliore accuracy raggiunta: {best_acc:.4f}")
    print(f"📊 Migliore F1 score raggiunto: {best_f1:.4f}")
    print(f"📊 Score combinato: {0.7 * best_acc + 0.3 * best_f1:.4f}")

if __name__ == "__main__":
    train()