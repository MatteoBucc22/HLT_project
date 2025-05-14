import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import default_data_collator
from tqdm.auto import tqdm
import time
import datetime
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from peft import get_peft_model, LoraConfig, TaskType

from data_loader import get_datasets
from model import get_model, MODEL_NAME
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, SAVE_DIR, DATASET_NAME
from hf_utils import save_to_hf


def generate_embeddings(model, dataloader, save_path, repo_id=None):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔍 Generating Embeddings"):
            labels = batch["labels"]
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model.base_model(**batch, output_hidden_states=True, return_dict=True)

            cls_embeddings = outputs.hidden_states[-1][:, 0, :]  # [batch_size, hidden_dim]
            all_embeddings.append(cls_embeddings.cpu())
            all_labels.extend(labels)

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.tensor(all_labels)

    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, "validation_embeddings.pt")
    torch.save(
        {"embeddings": all_embeddings, "labels": all_labels},
        file_path
    )
    print(f"💾 Embedding di validazione salvati in: {file_path}")

    # Upload embeddings to Hugging Face
    if repo_id:
        print(f"⏫ Caricamento embeddings su Hugging Face: {repo_id}")
        save_to_hf(save_path, repo_id=repo_id)
        print("✔️ Embeddings caricati su Hugging Face")


def train():
    dataset = get_datasets()
    base_model = get_model().to(DEVICE)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"]
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

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

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    use_amp = True
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

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

        # Save LoRA adapter every 2 epochs
        if (epoch + 1) % 2 == 0:
            adapter_dir_epoch = os.path.join(SAVE_DIR, f"{MODEL_NAME}-{DATASET_NAME}_epoch_{epoch+1}")
            os.makedirs(adapter_dir_epoch, exist_ok=True)
            model.save_pretrained(adapter_dir_epoch)
            print(f"✔️  LoRA adapter (epoch {epoch+1}) salvato in: {adapter_dir_epoch}")
            save_to_hf(adapter_dir_epoch, repo_id=f"MatteoBucc/passphrase-identification-{MODEL_NAME}-{DATASET_NAME}-epoch-{epoch+1}")

    # Save final adapter
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_dir_final = os.path.join(SAVE_DIR, f"{MODEL_NAME}-{DATASET_NAME}_lora_adapter_{ts}")
    os.makedirs(adapter_dir_final, exist_ok=True)
    model.save_pretrained(adapter_dir_final)
    print(f"✔️  LoRA adapter finale salvato in: {adapter_dir_final}")

    # Save full model weights
    pth_name = f"{MODEL_NAME}-{DATASET_NAME}_cross_encoder_{ts}.pth"
    pth_path = os.path.join(SAVE_DIR, pth_name)
    torch.save(model.state_dict(), pth_path)
    print(f"✔️ Modello cross‑encoder salvato in: {pth_path}")

    # Upload final adapter
    save_to_hf(adapter_dir_final, repo_id=f"MatteoBucc/passphrase-identification-{MODEL_NAME}-{DATASET_NAME}-final")

    # Save and upload embeddings for ensemble
    generate_embeddings(
        model,
        val_loader,
        save_path=adapter_dir_final,
        repo_id=f"MatteoBucc/passphrase-identification-{MODEL_NAME}-{DATASET_NAME}-embeddings-{ts}"
    )


if __name__ == "__main__":
    train()
