# train.py
import os
from sentence_transformers import losses, evaluation, SentencesDataset
from torch.utils.data import DataLoader
from config import (MODEL_NAME, BATCH_SIZE, EPOCHS,
                    LEARNING_RATE, SAVE_DIR, DEVICE,
                    EVAL_STEPS, WARMUP_RATIO, HUB_MODEL_ID)
from src.data_loader import get_examples
from src.model import get_siamese_model
import numpy as np
from adapter_hub import save_to_hf, generate_embeddings  # Assicurati di installare adapter_hub


def main():
    # Directory di output
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Carica dati
    train_examples, dev_examples = get_examples()

    # Inizializza modello siamese SBERT
    model = get_siamese_model()

    # DataLoader per train e dev
    train_dataset = SentencesDataset(train_examples, model.tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    
    dev_dataset = SentencesDataset(dev_examples, model.tokenizer)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=BATCH_SIZE)

    # Loss e evaluator
    train_loss = losses.CosineSimilarityLoss(model)
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        dev_examples, name='dev-eval', show_progress_bar=True
    )

    # Calcolo warmup steps
    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)

    # Training
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=EPOCHS,
        evaluation_steps=EVAL_STEPS,
        warmup_steps=warmup_steps,
        output_path=SAVE_DIR,
        optimizer_params={'lr': LEARNING_RATE}
    )

    # Upload finale su HuggingFace con adapter
    adapter_dir_final = os.path.join(SAVE_DIR, "final_adapter")
    # Supponendo che il modello SBERT sia stato salvato in SAVE_DIR
    save_to_hf(adapter_dir_final, repo_id=
                f"MatteoBucc/passphrase-identification-{MODEL_NAME}-{DATASET_NAME}-final")

    # ⏬ Salva embeddings per l'ensemble
    generate_embeddings(model, dev_dataloader, save_path=adapter_dir_final)

    print("Salvataggi completati: adapter e embeddings.")

if __name__ == '__main__':
    main()