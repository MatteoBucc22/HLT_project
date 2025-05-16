import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import default_data_collator, AutoTokenizer, AutoConfig
from tqdm.auto import tqdm
import time
import datetime
from sklearn.metrics import accuracy_score, f1_score

from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from data_loader import get_datasets
from model import get_model, MODEL_NAME
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, SAVE_DIR, DATASET_NAME, SEED
from hf_utils import save_to_hf

# Impostazione dei seed per riproducibilità
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_worker_seed(worker_id):
    # Assicura seed anche per i worker dei DataLoader
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def generate_embeddings(model, dataloader, save_path, repo_id=None):
    """
    Genera e salva embeddings CLS dell'ultimo layer su validation set,
    e carica su HF se repo_id specificato.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔍 Generating Embeddings"):
            labels = batch["labels"].cpu().tolist()
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}
            outputs = model.base_model(**batch, output_hidden_states=True, return_dict=True)

            cls_embeddings = outputs.hidden_states[-1][:, 0, :]
            all_embeddings.append(cls_embeddings.cpu())
            all_labels.extend(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.tensor(all_labels)

    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, "validation_embeddings.pt")
    torch.save({"embeddings": all_embeddings, "labels": all_labels}, file_path)
    print(f"💾 Embeddings salvati in: {file_path}")

    if repo_id:
        print(f"⏫ Caricamento embeddings su HF: {repo_id}")
        save_to_hf(save_path, repo_id=repo_id)
        print("✔️ Embeddings caricati su HF")


def train():
    # Carico dataset e modello
    dataset = get_datasets()
    base_model = get_model().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Configurazione PEFT LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"]
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # DataLoader con semina dei worker
    train_loader = DataLoader(
        dataset["train"], batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=default_data_collator,
        worker_init_fn=set_worker_seed
    )
    val_loader = DataLoader(
        dataset["validation"], batch_size=BATCH_SIZE,
        num_workers=4, pin_memory=True, collate_fn=default_data_collator,
        worker_init_fn=set_worker_seed
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )
    use_amp = True
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Training loop
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
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

        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f} — Time: {elapsed:.1f}s")

        # Validazione
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels.extend(batch["labels"].cpu().tolist())
                batch = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}
                outputs = model(**batch)
                batch_preds = outputs.logits.argmax(dim=1).cpu().tolist()
                preds.extend(batch_preds)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        print(f"📊 Validation — Acc: {acc:.4f} | F1: {f1:.4f}\n")

        # Salvataggio adapter ogni 2 epoche
        if (epoch + 1) % 2 == 0:
            adapter_dir = os.path.join(SAVE_DIR, f"adapter_epoch_{epoch+1}")
            os.makedirs(adapter_dir, exist_ok=True)
            model.save_pretrained(adapter_dir)
            save_to_hf(adapter_dir, repo_id=f"MatteoBucc/{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}-adapter-epoch-{epoch+1}")
            print(f"✔️ Adapter epoch {epoch+1} salvato in: {adapter_dir}\n")

    # Merge LoRA + base per cross-encoder completo
    merged = PeftModel.from_pretrained(base_model, SAVE_DIR)
    merged = merged.merge_and_unload()

    # Salvataggio modello completo
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_dir = os.path.join(SAVE_DIR, f"{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}-full-{ts}")
    os.makedirs(full_dir, exist_ok=True)

    # Config
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=merged.config.num_labels)
    config.id2label = merged.config.id2label
    config.label2id = merged.config.label2id
    config.save_pretrained(full_dir)

    # Tokenizer & Pesi
    tokenizer.save_pretrained(full_dir)
    merged.save_pretrained(full_dir)

    print(f"✔️ Modello completo salvato in: {full_dir}")
    save_to_hf(full_dir, repo_id=f"MatteoBucc/{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}-full")

    # Generazione embeddings per ensemble
    generate_embeddings(
        merged,
        val_loader,
        save_path=full_dir,
        repo_id=f"MatteoBucc/{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}-embeddings-{ts}"
    )


if __name__ == "__main__":
    train()

