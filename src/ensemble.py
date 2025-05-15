import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import os

# Percorsi agli embeddings
roberta_path = "MatteoBucc/passphrase-identification-roberta-base-qqp-embeddings-20250515_135729"
minilm_path = "MatteoBucc/passphrase-identification-sentence-transformers-all-MiniLM-L6-v2-qqp-embeddings-20250515_141045"

# Caricamento degli embeddings
roberta_data = torch.load(torch.hub.download_url_to_file(os.path.join(f"https://huggingface.co/{roberta_path}/resolve/main/validation_embeddings.pt"), "roberta.pt"))
minilm_data = torch.load(torch.hub.download_url_to_file(os.path.join(f"https://huggingface.co/{minilm_path}/resolve/main/validation_embeddings.pt"), "minilm.pt"))

# Verifica consistenza
assert len(roberta_data['labels']) == len(minilm_data['labels']), "I dataset devono avere lo stesso numero di esempi"

# Estrai embeddings e label
emb1 = roberta_data["embeddings"]
emb2 = minilm_data["embeddings"]
labels = roberta_data["labels"]

# Normalizzazione (cosine similarity benefit)
emb1 = F.normalize(emb1, dim=1)
emb2 = F.normalize(emb2, dim=1)

# Ensemble: media delle embeddings
ensemble_emb = (emb1 + emb2) / 2

# Classificatore lineare semplice (k-NN o simile, qui: soglia sulla distanza coseno rispetto a centroide positivo)
pos_mean = ensemble_emb[labels == 1].mean(dim=0)
neg_mean = ensemble_emb[labels == 0].mean(dim=0)

# Predizione: maggiore similarità a quale centroide?
sim_to_pos = F.cosine_similarity(ensemble_emb, pos_mean.unsqueeze(0))
sim_to_neg = F.cosine_similarity(ensemble_emb, neg_mean.unsqueeze(0))
preds = (sim_to_pos > sim_to_neg).long()

# Metriche
acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds)

print(f"\n🧪 Ensemble Roberta + MiniLM su QQP")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")