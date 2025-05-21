import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import default_data_collator
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import ContrastiveLoss, MultipleNegativesRankingLoss, SoftmaxLoss, CoSENTLoss
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.util import cos_sim
from tqdm.auto import tqdm
import time
import datetime
from random import sample


from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from datasets import load_dataset
from model import get_model, MODEL_NAME  # Importa anche MODEL_NAME
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, SAVE_DIR, DATASET_NAME

from sklearn.metrics import accuracy_score, f1_score  # <-- aggiungi all'inizio del file se non già presente

from sentence_transformers.losses import ContrastiveLoss
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from hf_utils import save_to_hf

def train(model_name, dataset_name, element_name):
    # Define model
    ## Step 1: use an existing language model
    word_embedding_model = Transformer(model_name)

    ## Step 2: use a pool function over the token embeddings
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension(), 
                                pooling_mode = 'cls',
                                pooling_mode_cls_token=True, 
                                pooling_mode_mean_tokens = False)

    ## Join steps 1 and 2 using the modules argument
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # CONFIGURA LORA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"]
    )
    model.add_adapter(peft_config)

    dataset = load_dataset("glue", dataset_name)

    # Format training data
    train_examples = []
    for example in dataset['train']:
        train_examples.append(InputExample(texts=[example[element_name+'1'], example[element_name+'2']], label=float(example['label'])))

    train_dataloader = DataLoader(sample(train_examples, 5000) if len(train_examples) > 5000 else train_examples, shuffle=True, batch_size=4)

    train_loss = ContrastiveLoss(model=model)
    # (anchor, positive), (anchor, positive, negative)
    mnrl_loss = MultipleNegativesRankingLoss(model)
    # (sentence_A, sentence_B) + class
    softmax_loss = SoftmaxLoss(model, model.get_sentence_embedding_dimension(), 3)
    # (sentence_A, sentence_B) + score
    cosent_loss = CoSENTLoss(model)

    # Format evaluation data
    sentences1 = []
    sentences2 = []
    scores = []
    for example in dataset['validation']:
        sentences1.append(example[element_name+'1'])
        sentences2.append(example[element_name+'2'])
        scores.append(float(example['label']))
    
    evaluator = BinaryClassificationEvaluator(sentences1[:500], sentences2[:500], scores[:500])

    # Start training
    model.fit(
        train_objectives=[(train_dataloader, train_loss)], 
        evaluator=evaluator,
        evaluation_steps=500,
        epochs=EPOCHS, 
        warmup_steps=0,
        output_path='./sentence_transformer/',
        weight_decay=0.01,
        optimizer_params={'lr': LEARNING_RATE},
        save_best_model=True,
        show_progress_bar=True,
    )

    model.save(f"../outputs/{model_name}-{dataset_name}")
"""
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
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    train_loss = ContrastiveLoss(model=model)

    # Format evaluation data
    sentences1 = []
    sentences2 = []
    scores = []
    for example in dataset['validation']:
        #print(example)
        sentences1.append(example['question1'])
        sentences2.append(example['question2'])
        scores.append(float(example['labels']))

    evaluator = BinaryClassificationEvaluator(sentences1, sentences2, scores)

    # Start training
    model.fit(
        train_objectives=[(train_loader, train_loss)], 
        evaluator=evaluator,
        evaluation_steps=500,
        epochs=1, 
        warmup_steps=0,
        output_path='./sentence_transformer/',
        weight_decay=0.01,
        optimizer_params={'lr': 0.00004},
        save_best_model=True,
        show_progress_bar=True,
    )

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
        for batch_idx, batch in enumerate(loop):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
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
            adapter_dir_epoch = os.path.join(SAVE_DIR, f"{MODEL_NAME}-{DATASET_NAME}_epoch_{epoch+1}")
            os.makedirs(adapter_dir_epoch, exist_ok=True)
            model.save_pretrained(adapter_dir_epoch)
            print(f"✔️  LoRA adapter (epoch {epoch+1}) salvato in: {adapter_dir_epoch}")
            save_to_hf(adapter_dir_epoch, repo_id=f"MatteoBucc/passphrase-identification-{MODEL_NAME}-{DATASET_NAME}-epoch-{epoch+1}")
 

    # Salvataggio del solo LoRA adapter finale
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_dir_final = os.path.join(SAVE_DIR, f"{MODEL_NAME}-{DATASET_NAME}_lora_adapter_{ts}")
    os.makedirs(adapter_dir_final, exist_ok=True)

    model.save_pretrained(adapter_dir_final)
    print(f"✔️  LoRA adapter finale salvato in: {adapter_dir_final}")

    # Salvataggio opzionale anche del modello intero come .pth
    pth_name = f"{MODEL_NAME}-{DATASET_NAME}_cross_encoder_qqp_{ts}.pth"
    pth_path = os.path.join(SAVE_DIR, pth_name)
    torch.save(model.state_dict(), pth_path)
    print(f"✔️ Modello cross‑encoder salvato in: {pth_path}")

    # Upload su Hugging Face Hub dell'adapter finale
    save_to_hf(adapter_dir_final, repo_id=f"MatteoBucc/passphrase-identification-{MODEL_NAME}-{DATASET_NAME}-final") """


if __name__ == "__main__":
    #train('distilroberta-base', 'mrpc', 'sentence')
    train('distilroberta-base', 'qqp', 'question')