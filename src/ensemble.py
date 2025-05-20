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
    tokenizer = AutoTokenizer.from_pretrained(info.get('base'))
    if info['type'] == 'peft':
        base_model = AutoModelForSequenceClassification.from_pretrained(info['base']).to(DEVICE)
        model = PeftModel.from_pretrained(base_model, info['adapter']).eval()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(info['model_repo']).to(DEVICE).eval()

    all_probs = []
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i: i + BATCH_SIZE]
        inputs = tokenizer(
            [p[0] for p in batch],
            [p[1] for p in batch],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        torch.cuda.empty_cache()
    return np.vstack(all_probs)


def compute_dynamic_weights(probs_list, true_labels):
    """Calcola pesi dinamici in base alle performance sui dati di validazione"""
    f1s = []
    for probs in probs_list:
        preds = np.argmax(probs, axis=1)
        f1s.append(f1_score(true_labels, preds))
    f1s = np.array(f1s)
    return f1s / f1s.sum()


def evaluate_ensemble_and_stacking(pairs, labels):
    # Split per dynamic weights e stacking
    pairs_train, pairs_val, y_train, y_val = train_test_split(
        pairs, labels, test_size=0.3, random_state=42
    )

    # Calcola probabilità
    probs_train = [predict_probs(info, pairs_train) for info in MODEL_INFOS.values()]
    probs_val = [predict_probs(info, pairs_val) for info in MODEL_INFOS.values()]
    weights = compute_dynamic_weights(probs_val, y_val)

    # Stacking metaclassificatore
    X_stack = np.hstack(probs_train)
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(X_stack, y_train)

    # Inference su train+val
    all_pairs = pairs_train + pairs_val
    all_labels = np.concatenate([y_train, y_val])
    all_probs = [predict_probs(info, all_pairs) for info in MODEL_INFOS.values()]

    # Ensemble pesato
    weighted_probs = sum(w * p for w, p in zip(weights, all_probs))
    preds_ens = np.argmax(weighted_probs, axis=1)

    # Stacking
    X_meta = np.hstack(all_probs)
    preds_stack = meta_clf.predict(X_meta)

    # Metriche
    return {
        'dynamic': {
            'accuracy': accuracy_score(all_labels, preds_ens),
            'f1': f1_score(all_labels, preds_ens)
        },
        'stacking': {
            'accuracy': accuracy_score(all_labels, preds_stack),
            'f1': f1_score(all_labels, preds_stack)
        }
    }

if __name__ == '__main__':
    results = {}
    # Carica e valuta su QQP
    qqp = load_dataset('glue', 'qqp', split='validation')
    qqp_pairs = [(ex['question1'], ex['question2']) for ex in qqp]
    qqp_labels = np.array(qqp['label'])
    results['QQP'] = evaluate_ensemble_and_stacking(qqp_pairs, qqp_labels)

    # Carica e valuta su MRPC
    mrpc = load_dataset('glue', 'mrpc', split='validation')
    mrpc_pairs = [(ex['sentence1'], ex['sentence2']) for ex in mrpc]
    mrpc_labels = np.array(mrpc['label'])
    results['MRPC'] = evaluate_ensemble_and_stacking(mrpc_pairs, mrpc_labels)

    # Mixed
    mixed_pairs = qqp_pairs + mrpc_pairs
    mixed_labels = np.concatenate([qqp_labels, mrpc_labels])
    results['Mixed'] = evaluate_ensemble_and_stacking(mixed_pairs, mixed_labels)

    # Stampa risultati
    for split, res in results.items():
        dyn = res['dynamic']
        stk = res['stacking']
        print(f"===== {split} =====")
        print(f"Ensemble dynamic weights - Accuracy: {dyn['accuracy']:.4f}, F1: {dyn['f1']:.4f}")
        print(f"Stacking metaclassificatore - Accuracy: {stk['accuracy']:.4f}, F1: {stk['f1']:.4f}\n")
