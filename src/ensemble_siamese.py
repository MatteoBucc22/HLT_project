import os
import torch
import numpy as np
import joblib
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
from hf_utils import save_to_hf
from config import DEVICE, BATCH_SIZE

# Configurazione dei modelli (Sentence-BERT Siamese)
MODEL_INFOS = {
    "siamese-qqp": {"type": "siamese", "path": "../models/distilroberta-base-qqp/"},
    "siamese-mrpc": {"type": "siamese", "path": "../models/distilroberta-base-mrpc/"}
}

ARTIFACT_DIR = "ensemble_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def predict_probs_siamese(model_path, pairs):
    model = SentenceTransformer(model_path, device=DEVICE)

    probs = []
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i:i + BATCH_SIZE]
        texts1 = [p[0] for p in batch]
        texts2 = [p[1] for p in batch]

        emb1 = model.encode(texts1, convert_to_tensor=True, device=DEVICE)
        emb2 = model.encode(texts2, convert_to_tensor=True, device=DEVICE)

        sims = util.cos_sim(emb1, emb2).diagonal()  # Similarità coseno tra i pari
        sims = (sims + 1) / 2  # scala da [-1, 1] → [0, 1]
        probs_batch = torch.stack([1 - sims, sims], dim=1).cpu().numpy()
        probs.append(probs_batch)
    
    return np.vstack(probs)

def predict_probs(info, pairs):
    if info['type'] == 'siamese':
        return predict_probs_siamese(info['path'], pairs)
    else:
        raise ValueError(f"Tipo modello non supportato: {info['type']}")

def compute_dynamic_weights(probs_list, true_labels):
    f1s = []
    for probs in probs_list:
        preds = np.argmax(probs, axis=1)
        f1s.append(f1_score(true_labels, preds))
    f1s = np.array(f1s)
    return f1s / f1s.sum()

def evaluate_ensemble_and_stacking(pairs, labels):
    pairs_train, pairs_val, y_train, y_val = train_test_split(pairs, labels, test_size=0.3, random_state=42)

    probs_train = [predict_probs(info, pairs_train) for info in MODEL_INFOS.values()]
    probs_val = [predict_probs(info, pairs_val) for info in MODEL_INFOS.values()]
    weights = compute_dynamic_weights(probs_val, y_val)

    # Salvataggio pesi dinamici
    weights_path = os.path.join(ARTIFACT_DIR, "dynamic_weights.npy")
    np.save(weights_path, weights)

    # Stacking metaclassificatore
    X_stack = np.hstack(probs_train)
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(X_stack, y_train)

    # Salvataggio del meta-classificatore
    clf_path = os.path.join(ARTIFACT_DIR, "stacking_meta_clf.joblib")
    joblib.dump(meta_clf, clf_path)

    all_pairs = pairs_train + pairs_val
    all_labels = np.concatenate([y_train, y_val])
    all_probs = [predict_probs(info, all_pairs) for info in MODEL_INFOS.values()]

    weighted_probs = sum(w * p for w, p in zip(weights, all_probs))
    preds_ens = np.argmax(weighted_probs, axis=1)

    X_meta = np.hstack(all_probs)
    preds_stack = meta_clf.predict(X_meta)

    return {
        'dynamic': {'accuracy': accuracy_score(all_labels, preds_ens), 'f1': f1_score(all_labels, preds_ens)},
        'stacking': {'accuracy': accuracy_score(all_labels, preds_stack), 'f1': f1_score(all_labels, preds_stack)}
    }

if __name__ == '__main__':
    results = {}
    qqp = load_dataset('glue', 'qqp', split='validation')
    qqp_pairs = [(ex['question1'], ex['question2']) for ex in qqp]
    qqp_labels = np.array(qqp['label'])
    results['QQP'] = evaluate_ensemble_and_stacking(qqp_pairs, qqp_labels)

    mrpc = load_dataset('glue', 'mrpc', split='validation')
    mrpc_pairs = [(ex['sentence1'], ex['sentence2']) for ex in mrpc]
    mrpc_labels = np.array(mrpc['label'])
    results['MRPC'] = evaluate_ensemble_and_stacking(mrpc_pairs, mrpc_labels)

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

    # Upload su Hugging Face
    save_to_hf(ARTIFACT_DIR, repo_id="MatteoBucc/ensemble-artifacts", commit_msg="Ensemble dynamic weights e stacking artifacts")
