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

# Configurazione dei modelli Siamese (PEFT adapter + manual pooling)
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


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    token_embeddings: [bs, seq_len, hidden]
    attention_mask:    [bs, seq_len]
    """
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()  # [bs,seq,hidden]
    summed = torch.sum(token_embeddings * mask, dim=1)                            # [bs,hidden]
    counts = mask.sum(dim=1)                                                      # [bs,hidden]
    return summed / counts.clamp(min=1e-9)


def predict_probs_siamese(info, pairs):
    # 1) tokenizer & modello PEFT
    tokenizer = AutoTokenizer.from_pretrained(info["adapter_path"])
    base_model = AutoModel.from_pretrained(info["base"]).to(DEVICE)
    model = PeftModel.from_pretrained(base_model, info["adapter_path"]).to(DEVICE).eval()

    all_probs = []
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i : i + BATCH_SIZE]
        texts1 = [p[0] for p in batch]
        texts2 = [p[1] for p in batch]

        inputs1 = tokenizer(texts1, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        inputs2 = tokenizer(texts2, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            # 2) estrazione hidden_states
            out1 = model(**inputs1, return_dict=True).last_hidden_state
            out2 = model(**inputs2, return_dict=True).last_hidden_state

            # 3) pooling mean + L2 norm
            emb1 = mean_pool(out1, inputs1.attention_mask)
            emb2 = mean_pool(out2, inputs2.attention_mask)
            emb1 = normalize(emb1, p=2, dim=1)
            emb2 = normalize(emb2, p=2, dim=1)

            # 4) cosine similarity → [−1,1] → [0,1]
            sims = cosine_similarity(emb1, emb2, dim=1)
            sims = (sims + 1) / 2

            # 5) probabilità [not-paraphrase, paraphrase]
            probs_batch = torch.stack([1 - sims, sims], dim=1).cpu().numpy()

        all_probs.append(probs_batch)
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

    # 2) pesi dinamici (F1-weighted)
    weights = compute_dynamic_weights(probs_val, y_val)
    np.save(os.path.join(ARTIFACT_DIR, "dynamic_weights.npy"), weights)

    # 3) stacking metaclassificatore
    X_stack = np.hstack(probs_train)
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(X_stack, y_train)
    joblib.dump(meta_clf, os.path.join(ARTIFACT_DIR, "stacking_meta_clf.joblib"))

    # 4) valutazione su train+val
    all_pairs  = pairs_train + pairs_val
    all_labels = np.concatenate([y_train, y_val])
    all_probs  = [predict_probs(info, all_pairs) for info in MODEL_INFOS.values()]

    weighted = sum(w * p for w, p in zip(weights, all_probs))
    preds_ens  = np.argmax(weighted, axis=1)
    X_meta     = np.hstack(all_probs)
    preds_stack = meta_clf.predict(X_meta)

    return {
        "dynamic":  {"accuracy": accuracy_score(all_labels, preds_ens),
                     "f1":       f1_score(all_labels, preds_ens)},
        "stacking": {"accuracy": accuracy_score(all_labels, preds_stack),
                     "f1":       f1_score(all_labels, preds_stack)}
    }


if __name__ == "__main__":
    results = {}

    # QQP
    qqp        = load_dataset("glue", "qqp", split="validation")
    qqp_pairs  = [(ex["question1"], ex["question2"]) for ex in qqp]
    qqp_labels = np.array(qqp["label"])
    results["QQP"] = evaluate_ensemble_and_stacking(qqp_pairs, qqp_labels)

    # MRPC
    mrpc        = load_dataset("glue", "mrpc", split="validation")
    mrpc_pairs  = [(ex["sentence1"], ex["sentence2"]) for ex in mrpc]
    mrpc_labels = np.array(mrpc["label"])
    results["MRPC"] = evaluate_ensemble_and_stacking(mrpc_pairs, mrpc_labels)

    # Mixed
    mixed_pairs  = qqp_pairs + mrpc_pairs
    mixed_labels = np.concatenate([qqp_labels, mrpc_labels])
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
