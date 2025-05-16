import os
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

def predict_single_adapter(adapter_repo: str, base_model: str, sentences: list[tuple[str,str]], device: str = "cuda") -> np.ndarray:
    """
    Carica on-the-fly il tokenizer, il modello base e l'adapter LoRA,
    esegue inferenza sulle coppie di frasi e rilascia memoria.

    Returns:
        probs: numpy array di shape (len(sentences), num_labels)
    """
    # Carica tokenizer e modello base
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForSequenceClassification.from_pretrained(base_model)
    # Applica LoRA adapter
    model = PeftModel.from_pretrained(base, adapter_repo).eval().to(device)

    # Tokenizza e inferisci
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

    # Pulisci
    del model, base, tokenizer, inputs, logits
    torch.cuda.empty_cache()
    return probs


def ensemble_predict(sentences: list[tuple[str,str]], device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """
    Esegue l'ensemble in streaming caricando ogni adapter + base di volta in volta.

    Returns:
        preds: array di predizioni (label id)
        avg_probs: array di probabilità medie
    """
    # Definisci qui i tuoi adapter e base_model
    MODEL_INFOS = {
        "roberta-qqp": {
            "adapter_repo": "MatteoBucc/passphrase-identification-roberta-base-qqp-final",
            "base_model": "roberta-base"
        },
        "minilm-qqp": {
            "adapter_repo": "MatteoBucc/passphrase-identification-all-MiniLM-L6-v2-qqp-final",
            "base_model": "sentence-transformers/all-MiniLM-L6-v2"
        },
        # aggiungi altri modelli se necessario
    }
    n = len(MODEL_INFOS)
    weights = {k: 1.0/n for k in MODEL_INFOS}

    all_probs = []
    for key, info in MODEL_INFOS.items():
        probs = predict_single_adapter(
            adapter_repo=info["adapter_repo"],
            base_model=info["base_model"],
            sentences=sentences,
            device=device
        )
        all_probs.append(weights[key] * probs)

    avg_probs = np.sum(all_probs, axis=0)
    preds = np.argmax(avg_probs, axis=1)
    return preds, avg_probs


def evaluate_task(pairs: list[tuple[str,str]], labels: np.ndarray, name: str, device: str = "cuda"):
    preds, _ = ensemble_predict(pairs, device)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(f"=== {name} ===")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}\n")


if __name__ == "__main__":
    # Carica validation split di QQP e MRPC
    qqp = load_dataset("glue", "qqp", split="validation")
    mrpc = load_dataset("glue", "mrpc", split="validation")

    qqp_pairs = [(ex["question1"], ex["question2"]) for ex in qqp]
    qqp_labels = np.array(qqp["label"])

    mrpc_pairs = [(ex["sentence1"], ex["sentence2"]) for ex in mrpc]
    mrpc_labels = np.array(mrpc["label"])

    # Mixed
    mixed_pairs = qqp_pairs + mrpc_pairs
    mixed_labels = np.concatenate([qqp_labels, mrpc_labels])

    # Valutazioni
    evaluate_task(qqp_pairs, qqp_labels, "QQP", device="cuda")
    evaluate_task(mrpc_pairs, mrpc_labels, "MRPC", device="cuda")
    evaluate_task(mixed_pairs, mixed_labels, "Mixed QQP+MRPC", device="cuda")
