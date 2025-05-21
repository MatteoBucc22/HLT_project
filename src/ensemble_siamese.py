import os
import torch
import numpy as np
import joblib
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from peft import PeftModel
from hf_utils import save_to_hf
from config import DEVICE, BATCH_SIZE

# Configurazione dei modelli Siamese (PEFT adapter + pooling)
MODEL_INFOS = {
    "siamese-qqp": {
        "type": "siamese",
        "adapter_path": "../models/distilroberta-base-qqp/"
    },
    "siamese-mrpc": {
        "type": "siamese",
        "adapter_path": "../models/distilroberta-base-mrpc/"
    }
}

ARTIFACT_DIR = "ensemble_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def predict_probs_siamese(info, pairs):
    """
    Carica la pipeline completa SentenceTransformer (tokenizer, transformer, pooling)
    e sostituisce solo il transformer con la versione PEFT + base_model.
    Restituisce probabilità [non-parafrasi, parafrasi].
    """
    # 1) Carica pipeline ST (tokenizer + transformer + pooling)
    model = SentenceTransformer(info["adapter_path"], device=DEVICE)
    
    # 2) Estrai il modulo Transformer e sostituisci il suo auto_model con PEFT
    transformer_module = model._first_module()  # solitamente è il TransformerModule
    peft_base = PeftModel.from_pretrained(transformer_module.auto_model, info["adapter_path"])
    peft_base.to(DEVICE)
    transformer_module.auto_model = peft_base
    model.eval()

    all_probs = []
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i : i + BATCH_SIZE]
        texts1 = [p[0] for p in batch]
        texts2 = [p[1] for p in batch]

        # 3) Ottieni embeddings (include pooling definito in 1_Pooling/)
        emb1 = model.encode(texts1, convert_to_tensor=True, device=DEVICE)
        emb2 = model.encode(texts2, convert_to_tensor=True, device=DEVICE)

        # 4) Cosine similarity → da [-1,1] a [0,1]
        sims = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
        sims = (sims + 1) / 2

        # 5) Probabilità: [not-paraphrase, paraphrase]
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

    # 1) Predizioni su train e val per ciascun modello
    probs_train = [predict_probs(info, pairs_train) for info in MODEL_INFOS.values()]
    probs_val   = [predict_probs(info, pairs_val)   for info in MODEL_INFOS.values()]

    # 2) Calcolo pesi dinamici (F1-weighted)
    weights = compute_dynamic_weights(probs_val, y_val)
    np.save(os.path.join(ARTIFACT_DIR, "dynamic_weights.npy"), weights)

    # 3) Stacking metaclassificatore
    X_stack = np.hstack(probs_train)
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(X_stack, y_train)
    joblib.dump(meta_clf, os.path.join(ARTIFACT_DIR, "stacking_meta_clf.joblib"))

    # 4) Valutazione su train+val
    all_pairs  = pairs_train + pairs_val
    all_labels = np.concatenate([y_train, y_val])
    all_probs  = [predict_probs(info, all_pairs) for info in MODEL_INFOS.values()]

    # Ensemble dinamico
    weighted = sum(w * p for w, p in zip(weights, all_probs))
    preds_ens = np.argmax(weighted, axis=1)

    # Stacking
    X_meta      = np.hstack(all_probs)
    preds_stack = meta_clf.predict(X_meta)

    return {
        "dynamic":  {
            "accuracy": accuracy_score(all_labels, preds_ens),
            "f1":       f1_score(all_labels, preds_ens)
        },
        "stacking": {
            "accuracy": accuracy_score(all_labels, preds_stack),
            "f1":       f1_score(all_labels, preds_stack)
        }
    }


if __name__ == "__main__":
    results = {}

    # QQP
    qqp       = load_dataset("glue", "qqp", split="validation")
    qqp_pairs = [(ex["question1"], ex["question2"]) for ex in qqp]
    qqp_lbls  = np.array(qqp["label"])
    results["QQP"] = evaluate_ensemble_and_stacking(qqp_pairs, qqp_lbls)

    # MRPC
    mrpc       = load_dataset("glue", "mrpc", split="validation")
    mrpc_pairs = [(ex["sentence1"], ex["sentence2"]) for ex in mrpc]
    mrpc_lbls  = np.array(mrpc["label"])
    results["MRPC"] = evaluate_ensemble_and_stacking(mrpc_pairs, mrpc_lbls)

    # Mixed
    mixed_pairs  = qqp_pairs + mrpc_pairs
    mixed_labels = np.concatenate([qqp_lbls, mrpc_lbls])
    results["Mixed"] = evaluate_ensemble_and_stacking(mixed_pairs, mixed_labels)

    # Stampa risultati
    for split, res in results.items():
        d, s = res["dynamic"], res["stacking"]
        print(f"===== {split} =====")
        print(f"Ensemble dynamic weights - Accuracy: {d['accuracy']:.4f}, F1: {d['f1']:.4f}")
        print(f"Stacking metaclassificatore - Accuracy: {s['accuracy']:.4f}, F1: {s['f1']:.4f}\n")

    # Upload su Hugging Face
    save_to_hf(
        ARTIFACT_DIR,
        repo_id="MatteoBucc/ensemble-artifacts_siamese",
        commit_msg="Ensemble dynamic weights e stacking artifacts"
    )
