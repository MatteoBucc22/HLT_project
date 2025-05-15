
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import os
import io
import requests

# Percorsi agli embeddings su Hugging Face
roberta_url = (
    "https://huggingface.co/MatteoBucc/"
    "passphrase-identification-roberta-base-qqp-embeddings-20250515_135729/"
    "resolve/main/validation_embeddings.pt"
)
minilm_url = (
    "https://huggingface.co/MatteoBucc/"
    "passphrase-identification-sentence-transformers-all-MiniLM-L6-v2-qqp-embeddings-20250515_141045/"
    "resolve/main/validation_embeddings.pt"
)

def download_and_load_embedding(url):
    # Scarica il file in memoria
    response = requests.get(url)
    response.raise_for_status()
    buffer = io.BytesIO(response.content)
    # Carica solo i pesi, per sicurezza
    data = torch.load(buffer, map_location="cpu", weights_only=True)
    return data

# Scarica e carica embeddings
print("Downloading Roberta embeddings...")
roberta_data = download_and_load_embedding(roberta_url)
print("Downloading MiniLM embeddings...")
minilm_data = download_and_load_embedding(minilm_url)

# Verifica consistenza
assert len(roberta_data['labels']) == len(minilm_data['labels']), (
    "I dataset devono avere lo stesso numero di esempi"
)

# Estrai embeddings e label
e1 = roberta_data["embeddings"]
e2 = minilm_data["embeddings"]
labels = roberta_data["labels"]

# Normalizzazione
emb1 = F.normalize(e1, dim=1)
emb2 = F.normalize(e2, dim=1)

# Ensemble: media delle embeddings
ensemble_emb = (emb1 + emb2) / 2

# Centroidi delle classi
pos_centroid = ensemble_emb[labels == 1].mean(dim=0)
neg_centroid = ensemble_emb[labels == 0].mean(dim=0)

# Calcolo similarità e predizioni
sim_pos = F.cosine_similarity(ensemble_emb, pos_centroid.unsqueeze(0))
sim_neg = F.cosine_similarity(ensemble_emb, neg_centroid.unsqueeze(0))
preds = (sim_pos > sim_neg).long().tolist()

true_labels = labels.tolist()

# Metriche
tacc = accuracy_score(true_labels, preds)
tf1 = f1_score(true_labels, preds)

print("\n🧪 Ensemble Roberta + MiniLM su QQP")
print(f"Accuracy: {tacc:.4f}")
print(f"F1 Score: {tf1:.4f}")

