
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import os
import io
import requests

# URL per gli embeddings su Hugging Face
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

# Funzione per scaricare e caricare embeddings
def download_and_load_embedding(url):
    response = requests.get(url)
    response.raise_for_status()
    buffer = io.BytesIO(response.content)
    data = torch.load(buffer, map_location="cpu", weights_only=True)
    return data

print("Downloading Roberta embeddings...")
roberta_data = download_and_load_embedding(roberta_url)
print("Downloading MiniLM embeddings...")
minilm_data = download_and_load_embedding(minilm_url)

# Verifica consistenza
assert len(roberta_data['labels']) == len(minilm_data['labels']), (
    "I dataset devono avere lo stesso numero di esempi"
)

# Estrai embeddings e labels
emb1 = F.normalize(roberta_data["embeddings"], dim=1)   # [N, 768]
emb2 = F.normalize(minilm_data["embeddings"], dim=1)     # [N, 384]
labels = roberta_data["labels"]                         # [N]

def compute_centroids(embeddings, labels):
    pos_centroid = embeddings[labels == 1].mean(dim=0)
    neg_centroid = embeddings[labels == 0].mean(dim=0)
    return pos_centroid, neg_centroid

# Centroidi per ogni spazio di embedding
pos1, neg1 = compute_centroids(emb1, labels)
pos2, neg2 = compute_centroids(emb2, labels)

# Calcola similarità e predizioni combinate
# Pesi, se vuoi dare maggiore importanza a uno dei modelli
w1 = 0.8
w2 = 0.2

sims_pos = w1 * F.cosine_similarity(emb1, pos1.unsqueeze(0)) \
         + w2 * F.cosine_similarity(emb2, pos2.unsqueeze(0))
sims_neg = w1 * F.cosine_similarity(emb1, neg1.unsqueeze(0)) \
         + w2 * F.cosine_similarity(emb2, neg2.unsqueeze(0))

preds = (sims_pos > sims_neg).long().tolist()
true_labels = labels.tolist()

# Calcolo metriche
tacc = accuracy_score(true_labels, preds)
tf1 = f1_score(true_labels, preds)

print("\n🧪 Ensemble Roberta + MiniLM su QQP usando similarità centroidi")
print(f"Accuracy: {tacc:.4f}")
print(f"F1 Score: {tf1:.4f}")

