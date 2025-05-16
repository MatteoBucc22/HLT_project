import os
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Definisci qui i tuoi repository HF
MODEL_INFOS = {
    "roberta-qqp": {
        "hf_name": "MatteoBucc/passphrase-identification-roberta-base-qqp-full-20250516_123000",  # full model
        "base": "roberta-base"
    },
    "minilm-qqp": {
        "hf_name": "MatteoBucc/passphrase-identification-all-MiniLM-L6-v2-qqp-full-20250516_111648",
        "base": "sentence-transformers/all-MiniLM-L6-v2"
    }
}

def predict_single_adapter(adapter_info, sentences, device="cuda"):
    """
    Carica on-the-fly tokenizer, base-model e LoRA adapter,
    inferisce le probabilità e rilascia memoria.
    """
    # 1) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_info["base"])
    # 2) modello base
    base = AutoModelForSequenceClassification.from_pretrained(adapter_info["base"])  
    # 3) applichi LoRA adapter
    model = PeftModel.from_pretrained(base, adapter_info["hf_name"]).eval().to(device)

    # Tokenizzazione e inferenza
    inputs = tokenizer(
        [s[0] for s in sentences],
        [s[1] for s in sentences],
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Libera memoria
    del model, base, tokenizer, inputs, logits
    torch.cuda.empty_cache()

    return probs

def ensemble_predict(sentences, weights=None, device="cuda"):
    n = len(MODEL_INFOS)
    if weights is None:
        weights = {k: 1/n for k in MODEL_INFOS}

    all_probs = []
    for key, info in MODEL_INFOS.items():
        probs = predict_single_adapter(info, sentences, device)
        all_probs.append(weights[key] * probs)

    avg = np.sum(all_probs, axis=0)
    preds = np.argmax(avg, axis=1)
    return preds, avg

if __name__ == "__main__":
    # Carica validation set
    qqp_ds = load_dataset("glue", "qqp", split="validation")
    mrpc_ds = load_dataset("glue", "mrpc", split="validation")

    qqp_pairs = [(ex["question1"], ex["question2"]) for ex in qqp_ds]
    qqp_labels = np.array(qqp_ds["label"])

    mrpc_pairs = [(ex["sentence1"], ex["sentence2"]) for ex in mrpc_ds]
    mrpc_labels = np.array(mrpc_ds["label"])

    # Mixed
    mixed_pairs = qqp_pairs + mrpc_pairs
    mixed_labels = np.concatenate([qqp_labels, mrpc_labels])

    # Valuta
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
