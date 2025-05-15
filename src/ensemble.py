import os
import torch
import requests

def download_embedding_file(repo, filename, output_path):
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    response = requests.get(url)
    if response.status_code != 200:
        raise FileNotFoundError(f"Impossibile scaricare {url}")
    with open(output_path, "wb") as f:
        f.write(response.content)

# Percorsi Hugging Face
EMBEDDING_PATH_1 = "MatteoBucc/passphrase-identification-roberta-base-qqp-embeddings-20250515_135729"
EMBEDDING_PATH_2 = "MatteoBucc/passphrase-identification-sentence-transformers-all-MiniLM-L6-v2-qqp-embeddings-20250515_141045"

# Percorsi locali
embedding_file_1 = "validation_embeddings_1.pt"
embedding_file_2 = "validation_embeddings_2.pt"

# Scarica i file se non esistono già
if not os.path.exists(embedding_file_1):
    download_embedding_file(EMBEDDING_PATH_1, "validation_embeddings.pt", embedding_file_1)
if not os.path.exists(embedding_file_2):
    download_embedding_file(EMBEDDING_PATH_2, "validation_embeddings.pt", embedding_file_2)

# Ora puoi caricare
embedding_1 = torch.load(embedding_file_1, map_location="cpu")
embedding_2 = torch.load(embedding_file_2, map_location="cpu")
