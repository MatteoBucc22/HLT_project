import os
import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from hf_utils import save_to_hf

# Device and batch settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32

# Directory for saving artifacts
ARTIFACT_DIR = "ensemble_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Model information for inference
MODEL_INFOS = {
    "roberta-qqp": {
        "type": "peft",
        "base": "roberta-base",
        "adapter": "/kaggle/working/HLT_project/outputs/roberta-base-qqp_lora_adapter_20250527_122916"
    },
    "minilm-qqp": {
        "type": "peft",
        "base": "sentence-transformers/all-MiniLM-L6-v2",
        "adapter": "/kaggle/working/HLT_project/outputs/sentence-transformers-all-MiniLM-L6-v2-qqp_final_20250527_110735"
    },
    "roberta-mrpc": {
        "type": "full",
        "base": "roberta-base",
        "model_repo": "/kaggle/working/HLT_project/outputs/roberta-base-mrpc-best"
    },
    "minilm-mrpc": {
        "type": "full",
        "base": "sentence-transformers/all-MiniLM-L12-v2",
        "model_repo": "/kaggle/working/HLT_project/outputs/sentence-transformers-all-MiniLM-L12-v2-mrpc-best"
    }
}

# Helper to plot and save confusion matrices
def plot_and_save_cm(cm, title, filename):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]),
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.colorbar()
    plt.tight_layout()
    path = os.path.join(ARTIFACT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {path}")

# Predict probabilities for a list of sentence pairs
def predict_probs(info, pairs):
    tokenizer = AutoTokenizer.from_pretrained(info.get('base'))
    if info['type'] == 'peft':
        base_model = AutoModelForSequenceClassification.from_pretrained(
            info['base']
        ).to(device)
        model = PeftModel.from_pretrained(base_model, info['adapter']).eval()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            info['model_repo']
        ).to(device).eval()

    all_probs = []
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i: i + BATCH_SIZE]
        inputs = tokenizer([p[0] for p in batch],
                           [p[1] for p in batch],
                           padding=True, truncation=True,
                           return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        torch.cuda.empty_cache()
    return np.vstack(all_probs)

# Compute dynamic weights based on validation F1
def compute_dynamic_weights(probs_list, true_labels):
    f1s = []
    for probs in probs_list:
        preds = np.argmax(probs, axis=1)
        f1s.append(f1_score(true_labels, preds))
    f1s = np.array(f1s)
    return f1s / f1s.sum()

# Run evaluation, ensembling, stacking, and save confusion matrices
def evaluate_ensemble_and_stacking(pairs, labels, split_name=""):
    # Split
    pairs_train, pairs_val, y_train, y_val = train_test_split(
        pairs, labels, test_size=0.3, random_state=42
    )

    # Predict probabilities
    probs_train = [predict_probs(info, pairs_train) for info in MODEL_INFOS.values()]
    probs_val   = [predict_probs(info, pairs_val)   for info in MODEL_INFOS.values()]

    # Dynamic weighting
    weights = compute_dynamic_weights(probs_val, y_val)
    np.save(os.path.join(ARTIFACT_DIR, f"dynamic_weights_{split_name}.npy"), weights)

    # Stacking meta-classifier
    X_stack = np.hstack(probs_train)
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(X_stack, y_train)
    joblib.dump(meta_clf, os.path.join(ARTIFACT_DIR, f"stacking_meta_clf_{split_name}.joblib"))

    # Combine train+val for final metrics
    all_pairs  = pairs_train + pairs_val
    all_labels = np.concatenate([y_train, y_val])
    all_probs  = [predict_probs(info, all_pairs) for info in MODEL_INFOS.values()]

    # Ensemble predictions & confusion matrix
    weighted_probs = sum(w * p for w, p in zip(weights, all_probs))
    preds_ens = np.argmax(weighted_probs, axis=1)
    cm_ens = confusion_matrix(all_labels, preds_ens)
    plot_and_save_cm(
        cm_ens,
        f"{split_name.upper()} Dynamic Weights",
        f"cm_{split_name}_dynamic.png"
    )

    # Stacking predictions & confusion matrix
    X_meta     = np.hstack(all_probs)
    preds_stack = meta_clf.predict(X_meta)
    cm_stack    = confusion_matrix(all_labels, preds_stack)
    plot_and_save_cm(
        cm_stack,
        f"{split_name.upper()} Stacking",
        f"cm_{split_name}_stacking.png"
    )

    return {
        'dynamic': {
            'accuracy': accuracy_score(all_labels, preds_ens),
            'f1':       f1_score(all_labels, preds_ens),
            'confusion_matrix': cm_ens
        },
        'stacking': {
            'accuracy': accuracy_score(all_labels, preds_stack),
            'f1':       f1_score(all_labels, preds_stack),
            'confusion_matrix': cm_stack
        }
    }

if __name__ == '__main__':
    results = {}

    # QQP
    qqp = load_dataset('glue', 'qqp', split='validation')
    qqp_pairs  = [(ex['question1'], ex['question2']) for ex in qqp]
    qqp_labels = np.array(qqp['label'])
    results['QQP'] = evaluate_ensemble_and_stacking(qqp_pairs, qqp_labels, split_name="qqp")

    # MRPC
    mrpc = load_dataset('glue', 'mrpc', split='validation')
    mrpc_pairs  = [(ex['sentence1'], ex['sentence2']) for ex in mrpc]
    mrpc_labels = np.array(mrpc['label'])
    results['MRPC'] = evaluate_ensemble_and_stacking(mrpc_pairs, mrpc_labels, split_name="mrpc")

    # Mixed
    mixed_pairs  = qqp_pairs + mrpc_pairs
    mixed_labels = np.concatenate([qqp_labels, mrpc_labels])
    results['Mixed'] = evaluate_ensemble_and_stacking(mixed_pairs, mixed_labels, split_name="mixed")

    # Summary printout
    for split, res in results.items():
        dyn = res['dynamic']
        stk = res['stacking']
        print(f"===== {split} =====")
        print(f"Dynamic Ensemble — Acc: {dyn['accuracy']:.4f}, F1: {dyn['f1']:.4f}")
        print(f"Stacking        — Acc: {stk['accuracy']:.4f}, F1: {stk['f1']:.4f}")

    # Push artifacts to Hugging Face Hub
    save_to_hf(
        ARTIFACT_DIR,
        repo_id="MatteoBucc/ensemble-artifacts",
        commit_msg="Added confusion matrices and plots for ensemble and stacking"
    )