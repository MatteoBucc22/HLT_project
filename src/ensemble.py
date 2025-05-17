import os
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Configurazione dei modelli: tipo "peft" per adapter, tipo "full" per repo con modello completo
MODEL_INFOS = {
    "roberta-qqp": {
        "type": "peft",
        "base": "roberta-base",
        "adapter": "MatteoBucc/passphrase-identification-roberta-base-qqp-final"
    },
    "minilm-qqp": {
        "type": "peft",
        "base": "sentence-transformers/all-MiniLM-L6-v2",
        "adapter": "MatteoBucc/sentence-transformers-all-MiniLM-L6-v2-qqp-adapter-epoch-4"
    },
    "roberta-mrpc": {
        "type": "full",
        # repo contiene config.json e model.safetensors
        "model_repo": "MatteoBucc/passphrase-identification-roberta-base-mrpc-best"
    },
    "minilm-mrpc": {
        "type": "full",
        "model_repo": "MatteoBucc/passphrase-identification-sentence-transformers-all-MiniLM-L6-v2-mrpc-best"
    }
}

def predict_with_peft(base_model_name, adapter_name, pairs, device="cuda", batch_size=16):
    """
    Inference per modelli LoRA (PEFT).
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name).to(device)
    model = PeftModel.from_pretrained(base_model, adapter_name).eval()

    all_probs = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        inputs = tokenizer(
            [p[0] for p in batch],
            [p[1] for p in batch],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        del inputs, logits, probs
        torch.cuda.empty_cache()

    return np.vstack(all_probs)


def predict_with_full(model_repo, pairs, device="cuda", batch_size=16):
    """
    Inference per modelli fine-tuned salvati come modello completo su HF.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = AutoModelForSequenceClassification.from_pretrained(model_repo).to(device).eval()

    all_probs = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        inputs = tokenizer(
            [p[0] for p in batch],
            [p[1] for p in batch],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        del inputs, logits, probs
        torch.cuda.empty_cache()

    return np.vstack(all_probs)


def ensemble_predict(pairs, weights=None, device="cuda"):
    """
    Calcola ensemble su tutti i modelli definiti in MODEL_INFOS.
    """
    n_models = len(MODEL_INFOS)
    if weights is None:
        weights = {k: 1/n_models for k in MODEL_INFOS}

    weighted_probs = []
    for name, info in MODEL_INFOS.items():
        if info["type"] == "peft":
            probs = predict_with_peft(info["base"], info["adapter"], pairs, device)
        else:
            # full model
            probs = predict_with_full(info["model_repo"], pairs, device)
        weighted_probs.append(weights[name] * probs)

    avg_probs = np.sum(weighted_probs, axis=0)
    preds = np.argmax(avg_probs, axis=1)
    return preds, avg_probs


if __name__ == "__main__":
    # Carico validation set
    qqp_ds = load_dataset("glue", "qqp", split="validation")
    mrpc_ds = load_dataset("glue", "mrpc", split="validation")

    qqp_pairs = [(ex["question1"], ex["question2"]) for ex in qqp_ds]
    qqp_labels = np.array(qqp_ds["label"])

    mrpc_pairs = [(ex["sentence1"], ex["sentence2"]) for ex in mrpc_ds]
    mrpc_labels = np.array(mrpc_ds["label"])

    mixed_pairs = qqp_pairs + mrpc_pairs
    mixed_labels = np.concatenate([qqp_labels, mrpc_labels])

    # Valutazioni
    for split_name, pairs, labels in [
        ("QQP", qqp_pairs, qqp_labels),
        ("MRPC", mrpc_pairs, mrpc_labels),
        ("Mixed", mixed_pairs, mixed_labels)
    ]:
        preds, _ = ensemble_predict(pairs)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        print(f"=== {split_name} ===")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}\n")
