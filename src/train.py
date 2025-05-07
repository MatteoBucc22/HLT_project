import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import default_data_collator
from tqdm.auto import tqdm
import time
import datetime


from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from data_loader import get_datasets
from model import get_model
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, SAVE_DIR
from hf_utils import save_to_hf


def train():
    # Carica dataset e modello
    dataset = get_datasets()
    base_model = get_model().to(DEVICE)

    # CONFIGURA LORA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"]
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()  # per controllare quanti parametri si addestrano

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

    # Ottimizzatore (aggiorna SOLO parametri LoRA + classifier)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    # Mixed precision CUDA
    use_amp = True
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    from sklearn.metrics import accuracy_score, f1_score  # <-- aggiungi all'inizio del file se non già presente

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

        # 🔍 VALIDATION
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
            save_to_hf(model, adapter_dir=f"epoch_{epoch+1}")


    # Salvataggio del solo LoRA adapter
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_dir = os.path.join(SAVE_DIR, f"lora_adapter_{ts}")
    os.makedirs(adapter_dir, exist_ok=True)

    model.save_pretrained(adapter_dir)
    print(f"✔️  LoRA adapter salvato in: {adapter_dir}")

    # Salvataggio opzionale anche del modello intero come .pth
    pth_name = f"cross_encoder_qqp_{ts}.pth"
    pth_path = os.path.join(SAVE_DIR, pth_name)
    torch.save(model.state_dict(), pth_path)
    print(f"✔️ Modello cross‑encoder salvato in: {pth_path}")

    # Upload su Hugging Face Hub
    save_to_hf(adapter_dir, repo_id="MatteoBucc/passphrase-identification")




if __name__ == "__main__":
    train()
