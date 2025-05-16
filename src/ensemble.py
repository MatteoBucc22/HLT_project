import os
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configurazione dei modelli full su HF
MODEL_INFOS = {
    "roberta-qqp": {
        "hf_name": "MatteoBucc/passphrase-identification-roberta-base-qqp-final",
    },
    "minilm-qqp": {
        "hf_name": "MatteoBucc/sentence-transformers-all-MiniLM-L6-v2-qqp-adapter-epoch-4",
    }
}

def predict_single_full(model_name, sentences, device="cuda", batch_size=32):
    """
    Carica on-the-fly tokenizer e modello full fine-tuned,
    inferisce le probabilità in batch e rilascia memoria.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(device)

    all_probs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(
            [s[0] for s in batch],
            [s[1] for s in batch],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        del inputs, logits
        torch.cuda.empty_cache()

    all_probs = np.concatenate(all_probs, axis=0)

    del model, tokenizer
    torch.cuda.empty_cache()

    return all_probs

def ensemble_predict(sentences, weights=None, device="cuda"):
    n = len(MODEL_INFOS)
    if weights is None:
        weights = {k: 1/n for k in MODEL_INFOS}

    all_probs = []
    for key, info in MODEL_INFOS.items():
        probs = predict_single_full(info["hf_name"], sentences, device)
        all_probs.append(weights[key] * probs)

    avg = np.sum(all_probs, axis=0)
    preds = np.argmax(avg, axis=1)
    return preds, avg

if __name__ == "__main__":
    # Carico validation set
    qqp_ds = load_dataset("glue", "qqp", split="validation")
    mrpc_ds = load_dataset("glue", "mrpc", split="validation")

    qqp_pairs = [(ex["question1"], ex["question2"]) for ex in qqp_ds]
    qqp_labels = np.array(qqp_ds["label"])

    mrpc_pairs = [(ex["sentence1"], ex["sentence2"]) for ex in mrpc_ds]
    mrpc_labels = np.array(mrpc_ds["label"])

    # Mixed
    mixed_pairs = qqp_pairs + mrpc_pairs
    mixed_labels = np.concatenate([qqp_labels, mrpc_labels])

    # Valutazioni
    for name, pairs, labels in [
        ("QQP", qqp_pairs, qqp_labels),
        ("MRPC", mrpc_pairs, mrpc_labels),
        ("Mixed", mixed_pairs, mixed_labels)
    ]:
        preds, _ = ensemble_predict(pairs)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        print(f"=== {name} ===")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}\n")