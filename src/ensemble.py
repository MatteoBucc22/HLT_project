import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from transformers import AutoTokenizer
import os

# Percorsi dei due file di embedding (entrambi su QQP)
EMBEDDING_PATH_1 = "MatteoBucc/passphrase-identification-roberta-base-qqp-embeddings-20250515_135729"
EMBEDDING_PATH_2 = "MatteoBucc/passphrase-identification-sentence-transformers-all-MiniLM-L6-v2-qqp-embeddings-20250515_141045"

# Scarica automaticamente da Hugging Face
embedding_1 = torch.load(f"https://huggingface.co/{EMBEDDING_PATH_1}/resolve/main/validation_embeddings.pt", map_location="cpu")
embedding_2 = torch.load(f"https://huggingface.co/{EMBEDDING_PATH_2}/resolve/main/validation_embeddings.pt", map_location="cpu")

# Assicura che le dimensioni corrispondano
assert embedding_1["embeddings"].shape[0] == embedding_2["embeddings"].shape[0], "Dimension mismatch"
assert torch.equal(embedding_1["labels"], embedding_2["labels"]), "Labels mismatch"

# Estrai embeddings e normalizzali (opzionale ma utile per cosine similarity)
emb1 = torch.nn.functional.normalize(embedding_1["embeddings"], dim=1)
emb2 = torch.nn.functional.normalize(embedding_2["embeddings"], dim=1)

# Ensemble: media semplice
ensemble_embeddings = (emb1 + emb2) / 2

# Classificatore: soglia su cosine similarity (paraphrase detection)
cosine_similarities = torch.nn.functional.cosine_similarity(ensemble_embeddings, torch.zeros_like(ensemble_embeddings))
threshold = 0.0  # Puoi ottimizzare questa soglia
preds = (cosine_similarities > threshold).long()

# Valutazione
labels = embedding_1["labels"]
acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds)

print(f"✅ ENSEMBLE QQP — Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
