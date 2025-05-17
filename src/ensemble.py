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
        "adapter": "MatteoBucc/passphrase-identification-roberta-base-qqp-final",
        # peso iniziale più alto per RoBERTa
        "weight": 0.3
    },
    "minilm-qqp": {
        "type": "peft",
        "base": "sentence-transformers/all-MiniLM-L6-v2",
        "adapter": "MatteoBucc/sentence-transformers-all-MiniLM-L6-v2-qqp-adapter-epoch-4",
        "weight": 0.2
    },
    "roberta-mrpc": {
        "type": "full",
        "base": "roberta-base",
        "model_repo": "MatteoBucc/passphrase-identification-roberta-base-mrpc-best",
        "weight": 0.3
    },
    "minilm-mrpc": {
        "type": "full",
        "base": "sentence-transformers/all-MiniLM-L6-v2",
        "model_repo": "MatteoBucc/passphrase-identification-sentence-transformers-all-MiniLM-L6-v2-mrpc-best",
        "weight": 0.2
    }
}


def predict_with_peft(base_model_name, adapter_name, pairs, device="cuda", batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name).to(device)
    model = PeftModel.from_pretrained(base_model, adapter_name).eval()

    all_probs = []
    total = len(pairs)
    for i in range(0, total, batch_size):
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


def predict_with_full(base_model_name, model_repo, pairs, device="cuda", batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_repo).to(device).eval()

    all_probs = []
    total = len(pairs)
    for i in range(0, total, batch_size):
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


def ensemble_predict(pairs, device="cuda"):
    # Estrai i pesi e normalizzali
    weights = {name: info.get("weight", 1.0) for name, info in MODEL_INFOS.items()}
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    weighted_probs = []
    for name, info in MODEL_INFOS.items():
        print(f"--- Inference con modello: {name} ---")
        if info["type"] == "peft":
            probs = predict_with_peft(info["base"], info["adapter"], pairs, device)
        else:
            probs = predict_with_full(info["base"], info["model_repo"], pairs, device)
        weighted = weights[name] * probs
        weighted_probs.append(weighted)
        print(f"Pesi: {name} -> {weights[name]:.2f}\n")

    print("--- Calcolo ensemble ---")
    avg_probs = np.sum(weighted_probs, axis=0)
    preds = np.argmax(avg_probs, axis=1)
    return preds, avg_probs


if __name__ == "__main__":
    qqp_ds = load_dataset("glue", "qqp", split="validation")
    mrpc_ds = load_dataset("glue", "mrpc", split="validation")

    qqp_pairs = [(ex["question1"], ex["question2"]) for ex in qqp_ds]
    qqp_labels = np.array(qqp_ds["label"])

    mrpc_pairs = [(ex["sentence1"], ex["sentence2"]) for ex in mrpc_ds]
    mrpc_labels = np.array(mrpc_ds["label"])

    mixed_pairs = qqp_pairs + mrpc_pairs
    mixed_labels = np.concatenate([qqp_labels, mrpc_labels])

    for split_name, pairs, labels in [
        ("QQP", qqp_pairs, qqp_labels),
        ("MRPC", mrpc_pairs, mrpc_labels),
        ("Mixed", mixed_pairs, mixed_labels)
    ]:
        print(f"===== Valutazione Split: {split_name} =====")
        preds, _ = ensemble_predict(pairs)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        print(f"=== {split_name} ===")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}\n")
