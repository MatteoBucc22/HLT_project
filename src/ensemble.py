import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from peft import PeftModel

# Configurazione dei modelli su Hugging Face
MODEL_INFOS = {
    "roberta-qqp": {
        "adapter": "MatteoBucc/passphrase-identification-roberta-base-qqp-final",
        "base": "roberta-base",
        "tokenizer": None,
        "model": None
    },
    "minilm-qqp": {
        "adapter": "MatteoBucc/sentence-transformers-all-MiniLM-L6-v2-qqp-adapter-epoch-4",
        "base": "sentence-transformers/all-MiniLM-L6-v2",
        "tokenizer": None,
        "model": None
    }
}

# Caricamento di tokenizers e modelli base + PEFT adapter
for key, info in MODEL_INFOS.items():
    # Carica tokenizer e modello base
    info["tokenizer"] = AutoTokenizer.from_pretrained(info["base"])
    base_model = AutoModelForSequenceClassification.from_pretrained(info["base"])

    # Carica adapter PEFT
    info["model"] = PeftModel.from_pretrained(base_model, info["adapter"]).eval()

@torch.no_grad()
def predict_single(model, tokenizer, sentences):
    """
    sentences: list of tuples (sent1, sent2)
    returns: numpy array shape (len(sentences), num_labels)
    """
    inputs = tokenizer(
        [s[0] for s in sentences],
        [s[1] for s in sentences],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probs

def ensemble_predict(sentences, weights=None):
    """
    sentences: list of tuples (sent1, sent2)
    weights: dict with per-model weights (default uniform)
    returns: (preds, avg_probs)
    """
    n = len(MODEL_INFOS)
    if weights is None:
        weights = {k: 1/n for k in MODEL_INFOS}

    all_probs = []
    for key, info in MODEL_INFOS.items():
        probs = predict_single(info["model"], info["tokenizer"], sentences)
        all_probs.append(weights[key] * probs)

    avg_probs = np.sum(all_probs, axis=0)
    preds = np.argmax(avg_probs, axis=1)
    return preds, avg_probs

def load_task(task_name):
    """Load validation split for QQP or MRPC and return pairs and labels"""
    ds = load_dataset("glue", task_name, split="validation")
    if task_name == "qqp":
        pairs = [(ex["question1"], ex["question2"]) for ex in ds]
    else:  # mrpc
        pairs = [(ex["sentence1"], ex["sentence2"]) for ex in ds]
    labels = np.array(ds["label"])
    return pairs, labels

def evaluate(pairs, labels, name):
    preds, _ = ensemble_predict(pairs)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(f"=== Evaluation on {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

if __name__ == "__main__":
    # Carico dataset
    qqp_pairs, qqp_labels = load_task("qqp")
    mrpc_pairs, mrpc_labels = load_task("mrpc")

    # Mix di entrambi
    mixed_pairs = qqp_pairs + mrpc_pairs
    mixed_labels = np.concatenate([qqp_labels, mrpc_labels])

    # Valutazioni
    evaluate(qqp_pairs, qqp_labels, "QQP")
    evaluate(mrpc_pairs, mrpc_labels, "MRPC")
    evaluate(mixed_pairs, mixed_labels, "Mixed QQP+MRPC")
