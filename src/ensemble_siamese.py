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
from torch.nn.functional import cosine_similarity, normalize
from hf_utils import save_to_hf
from config import DEVICE, BATCH_SIZE

# Configurazione dei modelli Siamese (LoRA adapter + manual pooling)
MODEL_INFOS = {
    "siamese-qqp": {
        "base": "distilroberta-base",
        "adapter_path": "../models/distilroberta-base-qqp/"
    },
    "siamese-mrpc": {
        "base": "distilroberta-base",
        "adapter_path": "../models/distilroberta-base-mrpc/"
    }
}

ARTIFACT_DIR = "ensemble_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

from sklearn.metrics import f1_score

def find_best_threshold(sims, labels, num_steps=101):
    """
    sims: array shape (N,) di similarità [0,1] per la classe positiva
    labels: array binaria shape (N,)
    Restituisce (best_tau, best_f1)
    """
    best_tau, best_f1 = 0.0, 0.0
    for tau in np.linspace(0, 1, num_steps):
        preds = (sims >= tau).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return best_tau, best_f1



def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def predict_probs_siamese(info, pairs):
    # 1) tokenizer e modello base
    tokenizer = AutoTokenizer.from_pretrained(info["adapter_path"])
    base_model = AutoModel.from_pretrained(info["base"]).to(DEVICE)

    # 2) carica LoRA adapter e mergia nei pesi
    peft_model = PeftModel.from_pretrained(base_model, info["adapter_path"])
    base_model = peft_model.merge_and_unload().to(DEVICE).eval()

    all_probs = []
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i : i + BATCH_SIZE]
        texts1 = [p[0] for p in batch]
        texts2 = [p[1] for p in batch]

        inputs1 = tokenizer(texts1, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        inputs2 = tokenizer(texts2, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            # 3) estrai hidden_states
            out1 = base_model(**inputs1, return_dict=True).last_hidden_state
            out2 = base_model(**inputs2, return_dict=True).last_hidden_state

            # 4) applica pooling mean + L2 norm
            emb1 = mean_pool(out1, inputs1.attention_mask)
            emb2 = mean_pool(out2, inputs2.attention_mask)
            emb1 = normalize(emb1, p=2, dim=1)
            emb2 = normalize(emb2, p=2, dim=1)

            # 5) cosine similarity → [0,1]
            sims = cosine_similarity(emb1, emb2, dim=1)
            sims = (sims + 1) / 2

            # 6) probabilità [non-parafrasi, parafrasi]
            probs = torch.stack([1 - sims, sims], dim=1).cpu().numpy()

        all_probs.append(probs)
        torch.cuda.empty_cache()

    return np.vstack(all_probs)


def predict_probs(info, pairs):
    return predict_probs_siamese(info, pairs)


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

    # 1) predizioni train & val
    probs_train = [predict_probs(info, pairs_train) for info in MODEL_INFOS.values()]
    probs_val   = [predict_probs(info, pairs_val)   for info in MODEL_INFOS.values()]

    # 2) calcolo pesi dinamici (F1-weighted)
    weights = compute_dynamic_weights(probs_val, y_val)
    np.save(os.path.join(ARTIFACT_DIR, "dynamic_weights.npy"), weights)

    # 3) stacking standard
    X_stack_tr = np.hstack(probs_train)
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(X_stack_tr, y_train)
    joblib.dump(meta_clf, os.path.join(ARTIFACT_DIR, "stacking_meta_clf.joblib"))

    # 4) ensemble dinamico: combina tutte le predizioni
    all_pairs  = pairs_train + pairs_val
    all_labels = np.concatenate([y_train, y_val])
    probs_all  = [predict_probs(info, all_pairs) for info in MODEL_INFOS.values()]
    weighted   = sum(w * p for w, p in zip(weights, probs_all))  # shape (N_all, 2)

    # 5) calibrazione soglia sul validation
    sims_val = weighted[: len(pairs_val), 1]  # prendiamo solo la prob. della classe “paraphrase”
    best_tau, best_f1 = find_best_threshold(sims_val, y_val)
    print(f"[Calibration] Best τ = {best_tau:.2f}, F1_val = {best_f1:.4f}")

    # 6) predizioni ensemble con τ ottimale
    sims_all = weighted[:, 1]
    preds_ens = (sims_all >= best_tau).astype(int)

    # 7) stacking standard su train+val
    X_stack_all = np.hstack(probs_all)
    preds_stack = meta_clf.predict(X_stack_all)

    return {
        "dynamic": {
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
    qqp      = load_dataset("glue", "qqp", split="validation")
    qqp_p, qqp_l = [(ex["question1"], ex["question2"]) for ex in qqp], np.array(qqp["label"])
    results["QQP"] = evaluate_ensemble_and_stacking(qqp_p, qqp_l)

    # MRPC
    mrpc      = load_dataset("glue", "mrpc", split="validation")
    mrpc_p, mrpc_l = [(ex["sentence1"], ex["sentence2"]) for ex in mrpc], np.array(mrpc["label"])
    results["MRPC"] = evaluate_ensemble_and_stacking(mrpc_p, mrpc_l)

    # Mixed
    mixed_p  = qqp_p + mrpc_p
    mixed_l  = np.concatenate([qqp_l, mrpc_l])
    results["Mixed"] = evaluate_ensemble_and_stacking(mixed_p, mixed_l)

    # stampa
    for split, res in results.items():
        d, s = res["dynamic"], res["stacking"]
        print(f"===== {split} =====")
        print(f"Ensemble dynamic weights - Accuracy: {d['accuracy']:.4f}, F1: {d['f1']:.4f}")
        print(f"Stacking metaclassificatore - Accuracy: {s['accuracy']:.4f}, F1: {s['f1']:.4f}\n")

    # upload Hugging Face
    save_to_hf(
        ARTIFACT_DIR,
        repo_id="MatteoBucc/ensemble-artifacts_siamese",
        commit_msg="Merged LoRA adapters + ensemble artifacts"
    )
