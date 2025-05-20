import os
import torch
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from config import DEVICE, BATCH_SIZE

# Configurazione dei modelli
MODEL_INFOS = {
    "roberta-qqp": {"type": "peft", "base": "roberta-base", "adapter": "MatteoBucc/passphrase-identification-roberta-base-qqp-final"},
    "minilm-qqp": {"type": "peft", "base": "sentence-transformers/all-MiniLM-L6-v2", "adapter": "MatteoBucc/sentence-transformers-all-MiniLM-L6-v2-qqp-adapter-epoch-4"},
    "roberta-mrpc": {"type": "full", "base": "roberta-base", "model_repo": "MatteoBucc/passphrase-identification-roberta-base-mrpc-best"},
    "minilm-mrpc": {"type": "full", "base": "sentence-transformers/all-MiniLM-L6-v2", "model_repo": "MatteoBucc/passphrase-identification-sentence-transformers-all-MiniLM-L6-v2-mrpc-best"}
}


def predict_probs(info, pairs):
    """Restituisce le probabilità per un modello singolo"""
    tok = AutoTokenizer.from_pretrained(info.get('base'))
    if info['type'] == 'peft':
        base = AutoModelForSequenceClassification.from_pretrained(info['base']).to(DEVICE)
        model = PeftModel.from_pretrained(base, info['adapter']).eval()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(info['model_repo']).to(DEVICE).eval()

    all_probs = []
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i: i + BATCH_SIZE]
        inputs = tok([p[0] for p in batch], [p[1] for p in batch],
                     padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        torch.cuda.empty_cache()
    return np.vstack(all_probs)


def compute_dynamic_weights(probs_list, true_labels):
    """Calcola pesi dinamici in base alle performance sui dati di validazione"""
    # per ogni modello, calcola F1 sulle probabilità -> peso proporzionale
    f1s = []
    preds_list = [np.argmax(p, axis=1) for p in probs_list]
    for preds in preds_list:
        f1s.append(f1_score(true_labels, preds))
    f1s = np.array(f1s)
    weights = f1s / f1s.sum()
    return weights


def ensemble_with_dynamic_weights(pairs, labels):
    # Split validation per pesi
    p_train, p_val, y_train, y_val = train_test_split(pairs, labels, test_size=0.3, random_state=42)
    # Calcola probabilità su split train per stacking, e su val per pesi
    probs_train = [predict_probs(info, p_train) for info in MODEL_INFOS.values()]
    # Predizioni su val per dynamic weights
    probs_val = [predict_probs(info, p_val) for info in MODEL_INFOS.values()]
    weights = compute_dynamic_weights(probs_val, y_val)
    print("Pesi dinamici:", weights)

    # Costruisci feature matrix per stacking: concatenazione delle probs
    X_stack = np.hstack(probs_train)
    meta = LogisticRegression(max_iter=1000)
    meta.fit(X_stack, y_train)

    # Inferenza finale sul test intero (train+val)
    all_pairs = p_train + p_val
    all_probs = [predict_probs(info, all_pairs) for info in MODEL_INFOS.values()]
    # Ensemble pesato
    weighted = sum(w * p for w, p in zip(weights, all_probs))
    preds_ens = np.argmax(weighted, axis=1)
    # Stacking
    X_meta = np.hstack(all_probs)
    preds_stack = meta.predict(X_meta)

    # Metriche
    print("Ensemble dynamic weights:", accuracy_score(y_train+y_val, preds_ens), f1_score(y_train+y_val, preds_ens))
    print("Stacking metaclassificatore:", accuracy_score(y_train+y_val, preds_stack), f1_score(y_train+y_val, preds_stack))


if __name__ == "__main__":
    qqp = load_dataset("glue", "qqp", split="validation")
    mrpc = load_dataset("glue", "mrpc", split="validation")
    pairs = [(ex['question1'], ex['question2']) for ex in qqp] + [(ex['sentence1'], ex['sentence2']) for ex in mrpc]
    labels = np.array(qqp['label'] + mrpc['label'])
    ensemble_with_dynamic_weights(pairs, labels)
