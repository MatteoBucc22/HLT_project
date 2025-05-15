import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import warnings

# Nasconde il warning di sicurezza di torch.load
warnings.filterwarnings("ignore", category=FutureWarning)

# Percorsi locali ai file scaricati
embedding_file_1 = "roberta_embeddings.pt"
embedding_file_2 = "minilm_embeddings.pt"

# Carica gli embeddings (dizionari con 'embeddings' e 'labels')
embedding_1 = torch.load(embedding_file_1, map_location="cpu")
embedding_2 = torch.load(embedding_file_2, map_location="cpu")

# Estrai tensor e label
embeddings_1 = embedding_1["embeddings"]  # Roberta
embeddings_2 = embedding_2["embeddings"]  # MiniLM
labels = embedding_1["labels"]            # Le etichette sono uguali in entrambi

# Calcola logits da cosine similarity
def cosine_logits(embeddings):
    sim = F.cosine_similarity(embeddings[:, 0], embeddings[:, 1])
    return torch.stack([1 - sim, sim], dim=1)  # classe 0 e classe 1

logits_1 = cosine_logits(embeddings_1)
logits_2 = cosine_logits(embeddings_2)

# Ensemble con soft voting
ensemble_logits = (logits_1 + logits_2) / 2
preds = ensemble_logits.argmax(dim=1).numpy()

# Metriche
labels = np.array(labels)
acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds)

print(f"\n🔎 ENSEMBLE RESULTS (Roberta + MiniLM on QQP):")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
