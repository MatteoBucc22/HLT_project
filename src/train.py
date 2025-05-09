import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import default_data_collator
from tqdm.auto import tqdm
import time
import datetime

from peft import get_peft_config, get_peft_model, TaskType, PeftType

from data_loader import get_datasets
from model import get_model, MODEL_NAME
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, SAVE_DIR, DATASET_NAME
from hf_utils import save_to_hf

def train():
    # Carica dataset e modello base
    dataset = get_datasets()
    base_model = get_model().to(DEVICE)

    # CONFIGURA AutoLoRA via get_peft_config
    peft_config = get_peft_config(
        model=base_model,
        peft_type=PeftType.LORA,       # tipo di adapter
        task_type=TaskType.SEQ_CLS,    # sequence classification
        inference_mode=False,          # training mode
        init_r=8,                      # rank iniziale
        max_r=32,                      # rank massimo consentito
        lora_alpha=32,
        lora_dropout=0.1,
        growth_factor=2.0,             # fattore di crescita del rank
        threshold=0.01,                # soglia di “importanza” per aumentare r
        update_every=100,              # ogni quanti step r viene adattato
        target_modules=["query", "value"]
    )

    # Applica il configuration AutoLoRA al modello
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Prepara i DataLoader
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

    # Optimizer: solo parametri LoRA + classifier
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    # Mixed precision
    use_amp = True
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    from sklearn.metrics import accuracy_score, f1_score

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
        print(f"\nEpoch {epoch+1} — Avg Train Loss: {avg_loss:.4f} — Time: {epoch_time:.1f}s")

        # VALIDATION
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

        if (epoch + 1) % 2 == 0:
            adapter_dir_epoch = os.path.join(SAVE_DIR, f"{MODEL_NAME}-{DATASET_NAME}_epoch_{epoch+1}")
            os.makedirs(adapter_dir_epoch, exist_ok=True)
            model.save_pretrained(adapter_dir_epoch)
            save_to_hf(adapter_dir_epoch,
                       repo_id=f"MatteoBucc/passphrase-identification-{MODEL_NAME}-{DATASET_NAME}-epoch-{epoch+1}")

    # Salvataggio finale dell’adapter LoRA
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_dir_final = os.path.join(SAVE_DIR, f"{MODEL_NAME}-{DATASET_NAME}_lora_adapter_{ts}")
    model.save_pretrained(adapter_dir_final)
    print(f"✔️  LoRA adapter finale salvato in: {adapter_dir_final}")

    # (Opzionale) salva anche lo state_dict completo
    pth_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}-{DATASET_NAME}_cross_encoder_{ts}.pth")
    torch.save(model.state_dict(), pth_path)
    print(f"✔️ Modello cross‑encoder salvato in: {pth_path}")

    # Upload su Hugging Face Hub
    save_to_hf(adapter_dir_final,
               repo_id=f"MatteoBucc/passphrase-identification-{MODEL_NAME}-{DATASET_NAME}-final")

if __name__ == "__main__":
    train()
