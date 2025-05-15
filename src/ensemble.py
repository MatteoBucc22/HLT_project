import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from typing import List
import joblib
import argparse

def load_embeddings(paths: List[str], split="validation"):
    """Carica embeddings (validation/test) da più modelli"""
    all_embeddings = []
    labels = None

    for path in paths:
        file_path = os.path.join(path, f"{split}_embeddings.pt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File mancante: {file_path}")

        data = torch.load(file_path)
        all_embeddings.append(data["embeddings"])
        if labels is None:
            labels = data["labels"]
        else:
            assert torch.equal(labels, data["labels"]), "Le etichette non combaciano"

    return all_embeddings, labels


def ensemble_average(embeddings_list):
    stacked = torch.stack(embeddings_list)
    return stacked.mean(dim=0)


def ensemble_concat_classifier(embeddings_list, labels):
    X = torch.cat(embeddings_list, dim=1).numpy()
    y = labels.numpy()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    preds = clf.predict(X)
    return clf, preds


def save_classifier(clf, path):
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, "ensemble_classifier.joblib")
    joblib.dump(clf, model_path)
    print(f"💾 Classificatore salvato in: {model_path}")


def download_from_huggingface(repo_ids: List[str]):
    from huggingface_hub import snapshot_download

    local_paths = []
    for repo_id in repo_ids:
        print(f"⬇️ Scarico da Hugging Face: {repo_id}")
        path = snapshot_download(repo_id=repo_id, allow_patterns="*embeddings.pt")
        local_paths.append(path)
    return local_paths


def run_ensemble(
    embedding_dirs: List[str],
    split: str = "validation",
    strategy: str = "mean",
    save_dir: str = None,
    upload_to_hf: bool = False,
    hf_repo_id: str = None,
):
    embeddings_list, labels = load_embeddings(embedding_dirs, split=split)

    if strategy == "mean":
        X = ensemble_average(embeddings_list).numpy()
        y = labels.numpy()
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        preds = clf.predict(X)
    elif strategy == "concat":
        clf, preds = ensemble_concat_classifier(embeddings_list, labels)
        X = torch.cat(embeddings_list, dim=1).numpy()
        y = labels.numpy()
    else:
        raise ValueError("Strategia non supportata: usa 'mean' o 'concat'")

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    print(f"✅ Ensemble ({strategy}) su {split} — Accuracy: {acc:.4f} | F1: {f1:.4f}")

    if save_dir:
        save_classifier(clf, save_dir)

        # Salva le metriche
        with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
            f.write(f"Split: {split}\nStrategy: {strategy}\nAccuracy: {acc:.4f}\nF1: {f1:.4f}\n")

        if upload_to_hf:
            from huggingface_hub import login, create_repo, upload_folder
            login()  # Puoi usare login(token="hf_xxx") se vuoi autenticarti automaticamente
            create_repo(hf_repo_id, exist_ok=True)
            upload_folder(
                repo_id=hf_repo_id,
                folder_path=save_dir,
                path_in_repo=".",
                commit_message="Upload ensemble classifier"
            )
            print(f"📤 Classificatore caricato su Hugging Face Hub: {hf_repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegui ensemble su embedding salvati")
    parser.add_argument("--paths", nargs="+", help="Percorsi locali contenenti gli embedding (.pt)")
    parser.add_argument("--hf_paths", nargs="+", help="Repo Hugging Face contenenti gli embedding (.pt)")
    parser.add_argument("--split", default="validation", choices=["validation", "test"], help="Split da usare")
    parser.add_argument("--strategy", default="mean", choices=["mean", "concat"], help="Strategia ensemble")
    parser.add_argument("--save_dir", default=None, help="Cartella dove salvare il classificatore")
    parser.add_argument("--upload_to_hf", action="store_true", help="Upload su Hugging Face Hub")
    parser.add_argument("--hf_repo_id", default=None, help="Repo Hugging Face (es. username/nome_modello)")

    args = parser.parse_args()

    if not args.paths and not args.hf_paths:
        raise ValueError("Devi specificare almeno uno tra --paths o --hf_paths")

    if args.hf_paths:
        args.paths = download_from_huggingface(args.hf_paths)

    run_ensemble(
        embedding_dirs=args.paths,
        split=args.split,
        strategy=args.strategy,
        save_dir=args.save_dir,
        upload_to_hf=args.upload_to_hf,
        hf_repo_id=args.hf_repo_id
    )
