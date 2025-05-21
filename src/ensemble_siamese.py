import os
import torch
import numpy as np
import joblib
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from torch.nn.functional import cosine_similarity
from hf_utils import save_to_hf
from config import DEVICE, BATCH_SIZE

# Configurazione dei modelli Siamese (PEFT adapter + base)
MODEL_INFOS = {
    "siamese-qqp": {
        "type": "siamese",
        "base": "distilroberta-base",
        "adapter_path": "/mnt/data/siamese_models/distilroberta-base-qqp/"
    },
    "siamese-mrpc": {
        "type": "siamese",
        "base": "distilroberta-base",
        "adapter_path": "/mnt/data/siamese_models/distilroberta-base-mrpc/"
    }
}

ARTIFACT_DIR = "ensemble_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def predict_probs_siamese(info, pairs):
    """
    Carica il tokenizer e l’adapter PEFT su base AutoModel,
    estrae CLS token embeddings e calcola cosine similarity → probabilità.
    """
    tokenizer = AutoTokenizer.from_pretrained(info["adapter_path"])
    # Carico il base model e poi applico l’adapter
    base_model = AutoModel.from_pretrained(info["base"]).to(DEVICE)
    model = PeftModel.from_pretrained(base_model, info["adapter_path"]).to(DEVICE).eval()

    all_probs = []
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i : i + BATCH_SIZE]
        texts1 = [p[0] for p in batch]
        texts2 = [p[1] for p in batch]

        # tokenizziamo separatamente
        inputs1 = tokenizer(texts1, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        inputs2 = tokenizer(texts2, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            # estraggo gli embeddings CLS dal base_model (senza passare labels)
            emb1 = model.base_model(**inputs1).last_hidden_state[:, 0]
            emb2 = model.base_model(**inputs2).last_hidden_state[:, 0]

            # cosine similarity fra i due embeddings
            sims = cosine_similarity(emb1, emb2, dim=1)  # [-1,1]
            sims = (sims + 1) / 2  # → [0,1]

            # Probabilità [not-paraphrase, paraphrase]
            probs_batch = torch.stack([1 - sims, sims], dim=1).cpu().numpy()

        all_probs.append(probs_batch)
        torch.cuda.empty_cache()

    return np.vstack(all_probs)


def predict_probs(info, pairs):
    if info["type"] == "siamese":
        return predict_probs_siamese(info, pairs)
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
    pairs_train, pairs_val, y_train, y_val = train_test_split(
        pairs, labels, test_size=0.3, random_state=42
    )

    # 1) predizioni sui train e val per ogni modello
    probs_train = [predict_probs(info, pairs_train) for info in MODEL_INFOS.values()]
    probs_val   = [predict_probs(info, pairs_val)   for info in MODEL_INFOS.values()]

    # 2) calcolo pesi dinamici da validation
    weights = compute_dynamic_weights(probs_val, y_val)
    np.save(os.path.join(ARTIFACT_DIR, "dynamic_weights.npy"), weights)

    # 3) stacking metaclassificatore
    X_stack = np.hstack(probs_train)
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(X_stack, y_train)
    joblib.dump(meta_clf, os.path.join(ARTIFACT_DIR, "stacking_meta_clf.joblib"))

    # 4) valutazione su tutti i dati (train+val)
    all_pairs  = pairs_train + pairs_val
    all_labels = np.concatenate([y_train, y_val])
    all_probs  = [predict_probs(info, all_pairs) for info in MODEL_INFOS.values()]

    # ensemble weighted
    weighted = sum(w * p for w, p in zip(weights, all_probs))
    preds_ens  = np.argmax(weighted, axis=1)
    # stacking
    X_meta    = np.hstack(all_probs)
    preds_stack = meta_clf.predict(X_meta)

    return {
        "dynamic":  {"accuracy": accuracy_score(all_labels, preds_ens),
                     "f1":       f1_score(all_labels, preds_ens)},
        "stacking": {"accuracy": accuracy_score(all_labels, preds_stack),
                     "f1":       f1_score(all_labels, preds_stack)}
    }


if __name__ == "__main__":
    results = {}

    qqp   = load_dataset("glue", "qqp", split="validation")
    qqp_pairs  = [(ex["question1"], ex["question2"]) for ex in qqp]
    qqp_labels = np.array(qqp["label"])
    results["QQP"] = evaluate_ensemble_and_stacking(qqp_pairs, qqp_labels)

    mrpc   = load_dataset("glue", "mrpc", split="validation")
    mrpc_pairs  = [(ex["sentence1"], ex["sentence2"]) for ex in mrpc]
    mrpc_labels = np.array(mrpc["label"])
    results["MRPC"] = evaluate_ensemble_and_stacking(mrpc_pairs, mrpc_labels)

    # Mixed
    mixed_pairs  = qqp_pairs + mrpc_pairs
    mixed_labels = np.concatenate([qqp_labels, mrpc_labels])
    results["Mixed"] = evaluate_ensemble_and_stacking(mixed_pairs, mixed_labels)

    # Stampa risultati
    for split, res in results.items():
        d = res["dynamic"]
        s = res["stacking"]
        print(f"===== {split} =====")
        print(f"Ensemble dynamic weights - Accuracy: {d['accuracy']:.4f}, F1: {d['f1']:.4f}")
        print(f"Stacking metaclassificatore - Accuracy: {s['accuracy']:.4f}, F1: {s['f1']:.4f}\n")

    # Upload su Hugging Face
    save_to_hf(
        ARTIFACT_DIR,
        repo_id="MatteoBucc/ensemble-artifacts_siamese",
        commit_msg="Ensemble dynamic weights e stacking artifacts"
    )
